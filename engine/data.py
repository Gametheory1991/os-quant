"""Data adapters for the os-quant research engine.

Free/keyless sources only:
  - Yahoo Finance chart API (https://query1.finance.yahoo.com/v8/finance/chart/)
    for daily OHLCV + adjusted closes. No key, honest user-agent.
  - FRED (https://api.stlouisfed.org) for macro series — needs FRED_API_KEY in
    the environment; every function degrades gracefully when it is absent.

HTTP etiquette mirrors the os-bloom collector: an honest user-agent, ~0.6s
pacing between Yahoo calls, one retry with backoff on HTTP 429.

All price data is cached locally as parquet (data/cache/<TICKER>.parquet) so
backtests are reproducible without re-hitting the network. Adjusted closes
are the primary price: they account for splits AND distributions, which is
what a total-return backtest must use.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

log = logging.getLogger(__name__)

USER_AGENT = "os-quant/0.1 (+https://github.com/Gametheory1991/os-quant)"
YAHOO_PACE_SECONDS = 0.6
FRED_API_KEY_ENV = "FRED_API_KEY"

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
UNIVERSE_DIR = Path(__file__).resolve().parent.parent / "universes"


# ---------------------------------------------------------------------------
# low-level HTTP
# ---------------------------------------------------------------------------
def _http_get(url: str, timeout: int = 30, _retried: bool = False) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 429 and not _retried:
            log.warning("HTTP 429 for %s — backing off 15s and retrying once", url[:80])
            time.sleep(15)
            return _http_get(url, timeout=timeout, _retried=True)
        raise RuntimeError(f"HTTP {exc.code} for {url[:120]}") from None


# ---------------------------------------------------------------------------
# Yahoo Finance
# ---------------------------------------------------------------------------
def fetch_yahoo_daily(
    ticker: str,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Daily OHLCV + adjusted close for one ticker.

    Uses explicit period1/period2 (Unix timestamps) because Yahoo downsamples
    ``range=max`` responses to monthly bars. Returns a DataFrame indexed by
    date with columns: open, high, low, close, adjclose, volume, dividends.
    """
    if start is None:
        start = date(1990, 1, 1)
    if end is None:
        end = date.today()
    p1 = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp())
    # end-exclusive; add a day so the last session is included
    p2 = int(datetime(end.year, end.month, end.day, tzinfo=timezone.utc).timestamp()) + 86400
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{urllib.parse.quote(ticker)}"
        f"?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit"
    )
    payload = json.loads(_http_get(url))
    result = (payload.get("chart") or {}).get("result")
    if not result:
        err = ((payload.get("chart") or {}).get("error") or {}).get("description", "no result")
        raise RuntimeError(f"Yahoo returned no data for {ticker}: {err}")
    r = result[0]
    idx = pd.to_datetime(r["timestamp"], unit="s", utc=True).tz_convert(None).normalize()
    q = (r.get("indicators") or {}).get("quote", [{}])[0]
    adj = (r.get("indicators") or {}).get("adjclose", [{}])[0].get("adjclose", [])
    df = pd.DataFrame(
        {
            "open": q.get("open"),
            "high": q.get("high"),
            "low": q.get("low"),
            "close": q.get("close"),
            "adjclose": adj if len(adj) == len(idx) else q.get("close"),
            "volume": q.get("volume"),
        },
        index=idx,
    )
    df.index.name = "date"
    # dividends, if any
    divs = pd.Series(0.0, index=df.index)
    for ev in ((r.get("events") or {}).get("dividends") or {}).values():
        d = pd.Timestamp(datetime.fromtimestamp(ev["date"], tz=timezone.utc).date())
        if d in divs.index:
            divs.loc[d] = float(ev.get("amount", 0.0))
    df["dividends"] = divs.values
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def fetch_yahoo_batch(tickers: list[str], start: date | None = None,
                      end: date | None = None) -> dict[str, pd.DataFrame]:
    """Fetch several tickers with polite pacing between calls."""
    out: dict[str, pd.DataFrame] = {}
    for i, t in enumerate(tickers):
        if i:
            time.sleep(YAHOO_PACE_SECONDS)
        try:
            out[t] = fetch_yahoo_daily(t, start, end)
            log.info("yahoo %s: %d rows (%s -> %s)", t, len(out[t]),
                     out[t].index[0].date(), out[t].index[-1].date())
        except Exception as exc:  # noqa: BLE001 — per-ticker isolation
            log.warning("yahoo %s failed: %s", t, exc)
    return out


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
def cache_path(ticker: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{ticker.upper()}.parquet"


def load_or_fetch(ticker: str, start: date | None = None,
                  end: date | None = None, refresh: bool = False) -> pd.DataFrame:
    """Cached Yahoo daily bars. Refreshes when forced or when the cache does
    not cover the requested end date (i.e. is more than ~1 trading day stale)."""
    path = cache_path(ticker)
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
        want_end = end or date.today()
        if df.index[-1].date() >= want_end or (want_end - df.index[-1].date()).days <= 3:
            return df
        log.info("cache for %s stale (ends %s) — refreshing", ticker, df.index[-1].date())
    df = fetch_yahoo_daily(ticker, start=start or date(1990, 1, 1), end=end)
    df.to_parquet(path)
    return df


def load_universe_prices(tickers: list[str], refresh: bool = False) -> pd.DataFrame:
    """Total-return price panel (adjusted closes) for a ticker list.

    Returns a DataFrame indexed by date, one column per ticker. Missing values
    are left as NaN — the backtester treats NaN as untradeable (zero weight,
    zero return contribution). Callers choose the common backtest start date.
    """
    frames = {}
    for i, t in enumerate(tickers):
        if i:
            time.sleep(YAHOO_PACE_SECONDS)
        frames[t] = load_or_fetch(t, refresh=refresh)["adjclose"].rename(t)
    px = pd.concat(frames, axis=1).sort_index()
    px.index = pd.to_datetime(px.index).normalize()
    return px


# ---------------------------------------------------------------------------
# FRED (optional — needs FRED_API_KEY)
# ---------------------------------------------------------------------------
def fetch_fred(series_id: str, api_key: str | None = None,
               start: date | None = None) -> pd.Series | None:
    """FRED observations. Returns None (with a log line) when no API key is
    available — the engine never hard-fails on a missing optional key."""
    api_key = api_key or os.environ.get(FRED_API_KEY_ENV)
    if not api_key:
        log.info("FRED %s skipped: %s not set", series_id, FRED_API_KEY_ENV)
        return None
    params = {
        "series_id": series_id, "api_key": api_key, "file_type": "json",
        "observation_start": (start or date(1990, 1, 1)).isoformat(),
    }
    url = "https://api.stlouisfed.org/fred/series/observations?" + urllib.parse.urlencode(params)
    payload = json.loads(_http_get(url))
    obs = payload.get("observations", [])
    s = pd.Series(
        {pd.Timestamp(o["date"]): float(o["value"]) for o in obs if o["value"] != "."},
        name=series_id,
    ).sort_index()
    return s


# ---------------------------------------------------------------------------
# universes
# ---------------------------------------------------------------------------
def load_universe(name: str) -> dict:
    """Load universes/<name>.yaml -> {tickers, sleeves, ...}."""
    path = UNIVERSE_DIR / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
