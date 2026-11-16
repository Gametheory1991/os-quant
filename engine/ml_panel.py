"""ML Phase 0 — standing daily Treasury-liquidity panel + feature store.

Builds, from data already on disk (plus two small public pulls: FRED CSVs
via curl and the Treasury FiscalData auction file), the point-in-time
feature store specified in
``goals/quant-research-backtest-engine/references/ml_engine_design.md``
(Target 1, Phase 0).

Sources
-------
- Daily Treasury TRACE detail ``ust_detail_3y.json`` (FINRA daily Treasury
  aggregates XLSX, parsed by ``update_trace_data.py``).
- Yahoo cache parquets in ``data/cache/`` (adjclose = total return):
  SHY, IEF, TLT, TIP, ^TNX, ^VIX.
- FRED fredgraph.csv via **curl only** (urllib hangs on FRED from this
  host): BAMLH0A0HYM2 (HY OAS), BAMLC0A0CM (IG OAS), T10Y2Y.  Persisted
  incrementally under ``data/ml/``.
- Primary dealer net positions ``dealer_net.json`` (PDPOSGST-TOT, weekly).
- Monthly TRACE ``trace_monthly_all.json`` (corporate leg).
- Treasury FiscalData ``auctions_query`` (one pull; v1 endpoint — the v2
  path 404s as of 2026-10-13), persisted to ``data/ml/auctions.csv``.

Confirmed 6-vector semantics (from the source XLSX header row, inspected
2026-10-13 for ts-daily-aggregates-2026-10-02.xlsx — the parser in
``update_trace_data.py`` takes cells[1:7] and drops the header):

    [0] ATS & Interdealer trades          (count)
    [1] ATS & Interdealer par value       ($ BILLIONS)
    [2] Dealer-to-Customer trades         (count)
    [3] Dealer-to-Customer par value      ($ BILLIONS)
    [4] Total trades                      (count)   = [0] + [2]
    [5] Total par value                   ($ BILLIONS) = [1] + [3]

Header text: "Par Value amounts in $Billions"; column groups are
"ATS & Interdealer" / "Dealer to Customer" / "Total", each split
Trades / Par Value.  (The source also carries a 7th data column, VWAP,
populated only on on-the-run bucket rows; the updater drops it, so the
JSON carries no price fields.)  NOTE: this corrects the pre-build
assumption of $ millions — volumes are $ billions, so ADV in $B is the
total par value [5] as stored, with NO /1000 rescaling.

Point-in-time hard rule: every feature at row t is computable from data
dated <= t, with the publication lags documented per feature below.
No centered windows, no full-sample statistics anywhere.
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"
ML_DIR = ROOT / "data" / "ml"
UST_DIR = Path.home() / "workspace" / "your_files" / "ust_data"
TRACE_DAILY_PATH = UST_DIR / "ust_detail_3y.json"
TRACE_MONTHLY_PATH = UST_DIR / "trace_monthly_all.json"
DEALER_PATH = UST_DIR / "dealer_net.json"

# ---------------------------------------------------------------------------
# Bucket maps (FINRA label -> slug) and the Phase 0 ILLIQ ETF mapping.
# Construction decision (design §1 T1a left this to Phase 0): the daily
# file carries no price fields, so the return proxy per bucket is the
# mapped ETF's daily total return (adjclose pct change).
# ---------------------------------------------------------------------------
NOMINAL_BUCKETS = {
    "<= 2 years": "le2",
    "> 2 years and <= 3 years": "b2_3",
    "> 3 years and <= 5 years": "b3_5",
    "> 5 years and <= 7 years": "b5_7",
    "> 7 years and <= 10 years": "b7_10",
    "> 10 years and <= 20 years": "b10_20",
    "> 20 years": "gt20",
}
TIPS_BUCKETS = {
    "<= 5 years": "tips_le5",
    "> 5 years and <= 10 years": "tips_5_10",
    "> 10 years": "tips_gt10",
}
BUCKET_ETF = {
    "le2": "SHY", "b2_3": "SHY",
    "b3_5": "IEF", "b5_7": "IEF", "b7_10": "IEF",
    "b10_20": "TLT", "gt20": "TLT",
    "tips_le5": "TIP", "tips_5_10": "TIP", "tips_gt10": "TIP",
}
ALL_BUCKET_SLUGS = list(NOMINAL_BUCKETS.values()) + list(TIPS_BUCKETS.values())
AGGREGATES = {  # top-level JSON key -> slug
    "nominal_coupons": "nominal",
    "tips": "tips",
    "bills": "bills",
    "frns": "frns",
}
ENTITIES = ALL_BUCKET_SLUGS + list(AGGREGATES.values()) + ["agg"]

RECON_TOL = 0.005          # 0.5% bucket/aggregate vs reported tolerance
STUB_VOL_B = 1.0           # reported total par < $1B => holiday stub day
DEALER_LAG_DAYS = 7        # Wednesday obs usable w + 7 calendar days
CORP_LAG_DAYS = 45         # monthly CORP usable month-end + 45 days

FRED_SERIES = {  # series id -> persisted filename
    "BAMLH0A0HYM2": "oas_hy.csv",
    "BAMLC0A0CM": "oas_ig.csv",
    "T10Y2Y": "t10y2y.csv",
}
AUCTIONS_CSV = ML_DIR / "auctions.csv"


# ---------------------------------------------------------------------------
# Small point-in-time-safe helpers
# ---------------------------------------------------------------------------
def trailing_zscore(s: pd.Series, window: int = 252, min_periods: int = 126) -> pd.Series:
    """Trailing z-score: (x - trailing mean) / trailing std. Past data only."""
    m = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std()
    return (s - m) / sd


def velocity(s: pd.Series, lag: int, window: int = 252, min_periods: int = 63) -> pd.Series:
    """Harry's velocity convention: the maximum ``lag``-trading-day % rise
    ending at or before t, over the trailing ``window`` days."""
    pct = s.pct_change(lag)
    return pct.rolling(window, min_periods=min_periods).max()


def _asof(index: pd.DatetimeIndex, s: pd.Series) -> pd.Series:
    """Forward-fill series ``s`` onto ``index`` using only obs dated <= t."""
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(index, method="ffill")


# ---------------------------------------------------------------------------
# FRED via curl (incremental persist)
# ---------------------------------------------------------------------------
def fetch_fred_series(series_id: str, refresh: bool = False) -> pd.Series:
    """FRED series from fredgraph.csv via curl, persisted incrementally.

    If ``data/ml/<file>`` exists and ``refresh`` is False the file is used
    as-is (no network).  Otherwise the fresh CSV is merged in: old rows
    kept, only newer/changed dates appended (dedupe on date, fresh wins).
    """
    ML_DIR.mkdir(parents=True, exist_ok=True)
    dest = ML_DIR / FRED_SERIES[series_id]
    fresh = None
    if refresh or not dest.exists():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        out = subprocess.run(
            ["curl", "-s", "--max-time", "60", url],
            capture_output=True, text=True, check=True,
        ).stdout
        tmp = ML_DIR / f".fred_tmp_{series_id}.csv"
        tmp.write_text(out)
        fresh = pd.read_csv(tmp)
        tmp.unlink()
    frames = []
    if dest.exists():
        frames.append(pd.read_csv(dest))
    if fresh is not None:
        frames.append(fresh)
    raw = pd.concat(frames, ignore_index=True)
    date_col, val_col = raw.columns[0], raw.columns[1]
    raw[val_col] = pd.to_numeric(raw[val_col], errors="coerce")
    raw = raw.dropna(subset=[val_col])
    raw[date_col] = pd.to_datetime(raw[date_col])
    raw = (raw.drop_duplicates(subset=[date_col], keep="last")
              .sort_values(date_col))
    if fresh is not None:
        raw.to_csv(dest, index=False,
                   date_format="%Y-%m-%d")
    return pd.Series(raw[val_col].values,
                     index=pd.DatetimeIndex(raw[date_col]),
                     name=series_id).sort_index()


# ---------------------------------------------------------------------------
# Auctions (one FiscalData pull; v1 — v2 path 404s as of 2026-10-13)
# ---------------------------------------------------------------------------
def fetch_auctions(refresh: bool = False) -> pd.DataFrame | None:
    """Auction records (auction_date >= 2023-01-25) persisted to CSV.

    Returns None (feature set skipped) if the pull fails.  Uses the v1
    endpoint: the v2 path in the build spec returns a 404 page from this
    host; v1 serves the identical dataset.  Subsequent builds read the
    persisted CSV unless ``refresh``.
    """
    if AUCTIONS_CSV.exists() and not refresh:
        df = pd.read_csv(AUCTIONS_CSV, parse_dates=["auction_date", "issue_date"])
        return df
    fields = ("auction_date,issue_date,security_type,security_term,"
              "offering_amt,bid_to_cover_ratio,cusip")
    url = ("https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
           "/v1/accounting/od/auctions_query"
           f"?filter=auction_date:gte:2023-01-25&sort=-auction_date"
           f"&fields={fields}&page[size]=5000&format=json")
    try:
        out = subprocess.run(["curl", "-g", "-s", "--max-time", "90", url],
                             capture_output=True, text=True, check=True).stdout
        payload = json.loads(out)
        recs = payload.get("data", [])
        if not recs:
            raise ValueError("empty auctions payload")
        df = pd.DataFrame(recs)
        for c in ("auction_date", "issue_date"):
            df[c] = pd.to_datetime(df[c])
        for c in ("offering_amt", "bid_to_cover_ratio"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        ML_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(AUCTIONS_CSV, index=False)
        return df
    except Exception as exc:  # noqa: BLE001 — auctions are optional
        log.warning("auction pull failed, proceeding without: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Loaders for on-disk inputs (all take already-loaded objects where a test
# needs to truncate them first)
# ---------------------------------------------------------------------------
def load_trace_daily(path: Path = TRACE_DAILY_PATH) -> dict:
    with open(path) as fh:
        return json.load(fh)


def load_etf_prices(tickers: list[str]) -> dict[str, pd.Series]:
    out = {}
    for t in tickers:
        df = pd.read_parquet(CACHE_DIR / f"{t}.parquet")
        s = df["adjclose"].astype(float)
        s.index = pd.to_datetime(s.index).normalize()
        out[t] = s[~s.index.duplicated(keep="last")].sort_index()
    return out


def load_dealer_tot(path: Path = DEALER_PATH) -> list[tuple[pd.Timestamp, float]]:
    with open(path) as fh:
        raw = json.load(fh)
    return [(pd.Timestamp(d), float(v)) for d, v in raw["PDPOSGST-TOT"]]


def load_monthly(path: Path = TRACE_MONTHLY_PATH) -> dict:
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------
def _vec_fields(vec: list[float]) -> dict[str, float]:
    """Decode one FINRA 6-vector (see module docstring for semantics)."""
    return {
        "ats_trades": vec[0], "ats_vol": vec[1],
        "cust_trades": vec[2], "cust_vol": vec[3],
        "adt": vec[4], "adv": vec[5],
    }


def build_panel(trace: dict) -> tuple[pd.DataFrame, dict]:
    """Daily liquidity panel: one row per TRACE trading day.

    Columns per entity slug ``s``: ``adv_<s>`` ($B), ``adt_<s>`` (trades),
    ``ticket_<s>`` ($M per trade), ``ats_share_<s>`` (ATS & Interdealer
    par / total par), and ``onrun_share_<s>`` where an on-the-run split
    exists (buckets, nominal, tips, agg).  ``agg`` is computed as
    nominal_coupons + bills + tips + frns and reconciled against the
    file's own ``total`` row.
    """
    dates = sorted(trace.keys())
    idx = pd.DatetimeIndex(pd.to_datetime(dates), name="date")
    rows: dict[str, dict[str, float]] = {d: {} for d in dates}
    integrity: dict = {
        "repairs": [], "recon_failures": [],
        "zero_volume_entity_days": [],
    }
    max_recon = 0.0
    for d in dates:
        day = trace[d]
        rec = rows[d]
        fields: dict[str, dict[str, float]] = {}
        for label, slug in {**NOMINAL_BUCKETS, **TIPS_BUCKETS}.items():
            fields[slug] = _vec_fields(day["buckets"][label]["total"])
            on = _vec_fields(day["buckets"][label]["onrun"])
            fields[slug]["onrun_vol"] = on["adv"]
        for key, slug in AGGREGATES.items():
            fields[slug] = _vec_fields(day[key])
        # nominal / tips on-run volume = sum of their bucket on-run vols
        fields["nominal"]["onrun_vol"] = sum(
            fields[s]["onrun_vol"] for s in NOMINAL_BUCKETS.values())
        fields["tips"]["onrun_vol"] = sum(
            fields[s]["onrun_vol"] for s in TIPS_BUCKETS.values())
        # computed aggregate
        agg = {k: sum(fields[s][k] for s in AGGREGATES.values())
               for k in ("ats_trades", "ats_vol", "cust_trades",
                         "cust_vol", "adt", "adv")}
        agg["onrun_vol"] = fields["nominal"]["onrun_vol"] + fields["tips"]["onrun_vol"]
        fields["agg"] = agg
        # file's own on-run vector must equal the bucket sum (parser sums
        # the on-the-run rows of the nominal + TIPS sections)
        file_onrun_vol = float(day["onrun"][5])
        if abs(file_onrun_vol - agg["onrun_vol"]) > 0.35:
            integrity["recon_failures"].append(
                {"date": d, "check": "file_onrun_vs_bucket_sum",
                 "diff": file_onrun_vol - agg["onrun_vol"]})
        reported = _vec_fields(day["total"])
        rec["adv_total_reported"] = reported["adv"]
        rec["adt_total_reported"] = reported["adt"]
        denom = reported["adv"] if reported["adv"] else np.nan
        abs_diff = abs(agg["adv"] - reported["adv"])
        diff_pct = abs_diff / denom if denom else np.nan
        max_recon = max(max_recon, float(diff_pct) if diff_pct == diff_pct else 0.0)
        rec["recon_diff_pct"] = diff_pct
        # A failure needs BOTH a relative breach and an absolute breach
        # beyond source rounding (each stored value is rounded to 0.1 $B,
        # so a 4-way sum can differ by ~0.35 $B on rounding alone; on
        # holiday-stub days the relative diff is meaningless).
        if (diff_pct == diff_pct and diff_pct > RECON_TOL
                and abs_diff > 0.35):
            integrity["recon_failures"].append(
                {"date": d, "check": "agg_vs_reported_total",
                 "diff_pct": float(diff_pct), "abs_diff_b": float(abs_diff)})
        for slug in ENTITIES:
            f = fields[slug]
            adv, adt = f["adv"], f["adt"]
            rec[f"adv_{slug}"] = adv
            rec[f"adt_{slug}"] = float(adt)
            rec[f"ticket_{slug}"] = (adv * 1000.0 / adt) if adt else np.nan
            rec[f"ats_share_{slug}"] = (f["ats_vol"] / adv) if adv else np.nan
            if "onrun_vol" in f:
                rec[f"onrun_share_{slug}"] = (f["onrun_vol"] / adv) if adv else np.nan
            if adv <= 0:
                integrity["zero_volume_entity_days"].append(
                    {"date": d, "entity": slug, "adv": adv})
    panel = pd.DataFrame.from_dict(rows, orient="index")
    panel.index = idx
    panel = panel.sort_index()
    panel["is_holiday_stub"] = panel["adv_total_reported"] < STUB_VOL_B
    integrity["recon_max_abs_diff_pct"] = max_recon
    integrity["stub_days"] = [d.strftime("%Y-%m-%d")
                              for d in panel.index[panel["is_holiday_stub"]]]
    if integrity["stub_days"]:
        integrity["repairs"].append(
            "Holiday-stub days (reported total par < $1B: "
            + ", ".join(integrity["stub_days"])
            + ") kept in the panel as reported; their ILLIQ is set to NaN "
              "in the feature store and they are masked out of ADV "
              "change/velocity computations (ratio/volume changes against "
              "a stub are not meaningful). No rows deleted.")
    # calendar integrity: missing weekdays vs business-day calendar.
    # Bond-market holidays = US federal holidays + Good Friday (SIFMA
    # closes for Good Friday even though it is not a federal holiday;
    # verified: 2024-03-29, 2025-04-18, 2026-04-03 are absent).
    bdays = pd.bdate_range(panel.index.min(), panel.index.max())
    missing = [d for d in bdays if d not in panel.index]
    try:
        from pandas.tseries.holiday import USFederalHolidayCalendar
        from dateutil.easter import easter
        hols = set(USFederalHolidayCalendar().holidays(
            bdays.min(), bdays.max()).normalize())
        for yr in range(bdays.min().year, bdays.max().year + 1):
            gf = pd.Timestamp(easter(yr)) - pd.Timedelta(days=2)
            hols.add(gf.normalize())
    except Exception:  # noqa: BLE001
        hols = set()
    integrity["missing_weekdays"] = [d.strftime("%Y-%m-%d") for d in missing]
    integrity["missing_weekdays_nonholiday"] = [
        d.strftime("%Y-%m-%d") for d in missing if d.normalize() not in hols]
    integrity["panel_days_by_year"] = {
        str(y): int(n) for y, n in panel.groupby(panel.index.year).size().items()}
    if integrity["missing_weekdays_nonholiday"]:
        integrity["repairs"].append(
            "Coverage finding (no repair possible from disk): the daily "
            "file is SPARSE in Feb-Oct 2023 — "
            f"{len(integrity['missing_weekdays_nonholiday'])} non-holiday "
            "weekdays in that window have no FINRA daily file in the "
            "saved dataset (132 panel days in 2023 vs 252 in 2024). "
            "From 2023-11-01 onward the only missing weekdays are "
            "bond-market holidays. Rows are NOT fabricated; features in "
            "the sparse window are computed on the trading days present "
            "(pct changes there span the calendar gaps).")
    integrity["duplicate_dates"] = int(
        pd.Index(dates).duplicated().sum())
    integrity["n_days"] = len(panel)
    return panel, integrity


# ---------------------------------------------------------------------------
# Monthly corporate ADV (lagged to publication reality)
# ---------------------------------------------------------------------------
def monthly_corp_adv(monthly: dict) -> pd.DataFrame:
    """Monthly CORP ADV from the monthly TRACE file.

    The parser stores the monthly file's first 8 columns:
    [ATS, Interdealer, Customer, Total] trades, then the same four par
    values in $ MILLIONS (confirmed from the source XLSX header,
    2026-10-13; the file's own Average-Daily columns are dropped by the
    parser).  ADV = total par / trading days in month, where trading
    days = weekdays minus US federal holidays (validated: Sep 2026
    gives 21 days and ADV 60,196.1 $M, exactly FINRA's published
    average-daily par for CORP that month).
    """
    from pandas.tseries.holiday import USFederalHolidayCalendar
    cal = USFederalHolidayCalendar()
    recs = []
    for ym in sorted(monthly):
        if "CORP" not in monthly[ym]:
            continue
        total_par_m = float(monthly[ym]["CORP"][7])
        start = pd.Timestamp(f"{ym}-01")
        end = start + pd.offsets.MonthEnd(0)
        weekdays = pd.bdate_range(start, end)
        hols = set(cal.holidays(start, end).normalize())
        ndays = sum(1 for d in weekdays if d.normalize() not in hols)
        adv_b = (total_par_m / ndays / 1000.0) if ndays else np.nan
        recs.append({"month": ym, "month_end": end,
                     "trading_days": ndays, "corp_adv": adv_b})
    df = pd.DataFrame(recs)
    df["corp_adv_z12m"] = (
        (df["corp_adv"] - df["corp_adv"].rolling(12, min_periods=6).mean())
        / df["corp_adv"].rolling(12, min_periods=6).std())
    df["available"] = df["month_end"] + pd.Timedelta(days=CORP_LAG_DAYS)
    return df


# ---------------------------------------------------------------------------
# Feature store
# ---------------------------------------------------------------------------
def build_features(panel: pd.DataFrame, *,
                   etf_prices: dict[str, pd.Series],
                   oas_ig: pd.Series, oas_hy: pd.Series,
                   t10y2y: pd.Series,
                   dealer: list[tuple[pd.Timestamp, float]],
                   monthly: dict,
                   auctions: pd.DataFrame | None,
                   integrity: dict) -> pd.DataFrame:
    """Panel + all Phase 0 features.  Everything trailing/lagged (PIT)."""
    fs = panel.copy()
    idx = fs.index
    stub = fs["is_holiday_stub"]

    # --- ETF total returns (own calendar; NO fill onto panel days) ------
    etf_ret: dict[str, pd.Series] = {}
    for t, px in etf_prices.items():
        etf_ret[t] = px.pct_change()
    for t in ("SHY", "IEF", "TLT", "TIP"):
        fs[f"ret_{t.lower()}"] = etf_ret[t].reindex(idx)

    # --- ILLIQ per bucket + aggregate ------------------------------------
    missing_ret: dict[str, int] = {}
    illiq_cols = []
    for slug in ALL_BUCKET_SLUGS:
        r = etf_ret[BUCKET_ETF[slug]].reindex(idx)
        adv = fs[f"adv_{slug}"]
        ill = (r.abs() / adv).where(adv > 0)
        ill = ill.mask(stub)  # repair: stub-day ratio is meaningless
        fs[f"illiq_{slug}"] = ill
        illiq_cols.append(f"illiq_{slug}")
        missing_ret[slug] = int(r.isna().sum())
    integrity["etf_missing_return_days"] = missing_ret
    w = fs[[f"adv_{s}" for s in NOMINAL_BUCKETS.values()]].copy()
    w = w.mask(stub, other=np.nan)
    il = fs[[f"illiq_{s}" for s in NOMINAL_BUCKETS.values()]].copy()
    # volume-weighted mean over buckets with a valid ILLIQ that day
    # (positional numpy: il/w column names differ, DataFrame arithmetic
    # would align on names and produce all-NaN)
    il_v, w_v = il.to_numpy(dtype=float), w.to_numpy(dtype=float)
    valid = ~np.isnan(il_v) & ~np.isnan(w_v) & (w_v > 0)
    num = np.where(valid, il_v * w_v, 0.0).sum(axis=1)
    den = np.where(valid, w_v, 0.0).sum(axis=1)
    fs["illiq_agg"] = pd.Series(
        np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan),
        index=idx)
    fs["illiq_agg_z252"] = trailing_zscore(fs["illiq_agg"])

    # --- Velocity (Harry's convention) + plain changes -------------------
    adv_clean = fs["adv_agg"].mask(stub)
    for name, series in (("illiq", fs["illiq_agg"]), ("adv", adv_clean)):
        fs[f"{name}_chg5"] = series.pct_change(5)
        fs[f"{name}_chg21"] = series.pct_change(21)
        fs[f"{name}_vel7_252"] = velocity(series, 7)
        fs[f"{name}_vel30_252"] = velocity(series, 30)

    # --- 5Y-bucket concentration -----------------------------------------
    fs["share_3_5_agg"] = fs["adv_b3_5"] / fs["adv_agg"]
    fs["share_3_5_nominal"] = fs["adv_b3_5"] / fs["adv_nominal"]

    # --- OAS (levels in percent; changes in percentage points) -----------
    for name, s in (("oas_ig", oas_ig), ("oas_hy", oas_hy)):
        v = _asof(idx, s)
        fs[name] = v
        fs[f"{name}_chg1d"] = v.diff(1)
        fs[f"{name}_chg5d"] = v.diff(5)
        fs[f"{name}_z252"] = trailing_zscore(v)

    # --- Rates & vol -------------------------------------------------------
    tnx = _asof(idx, etf_prices["^TNX"])
    fs["tnx"] = tnx
    fs["tnx_chg63"] = tnx.diff(63)
    vix = _asof(idx, etf_prices["^VIX"])
    fs["vix"] = vix
    med = vix.rolling(252, min_periods=126).median()
    fs["vix_regime_high"] = (vix > med).astype("Int64")
    fs["vix_regime_high"] = fs["vix_regime_high"].where(med.notna())
    fs["slope_t10y2y"] = _asof(idx, t10y2y)

    # --- Dealer positioning (publication-lagged) ---------------------------
    ddf = pd.DataFrame(dealer, columns=["obs_date", "value"]).sort_values("obs_date")
    ddf["chg4w"] = ddf["value"] - ddf["value"].shift(4)
    ddf["usable"] = ddf["obs_date"] + pd.Timedelta(days=DEALER_LAG_DAYS)
    fs["dealer_net_tot"] = _asof(idx, ddf.set_index("usable")["value"])
    fs["dealer_net_chg4w"] = _asof(idx, ddf.set_index("usable")["chg4w"])

    # --- Monthly corporate ADV (publication-lagged) ------------------------
    cdf = monthly_corp_adv(monthly)
    fs["corp_adv_lag"] = _asof(idx, cdf.set_index("available")["corp_adv"])
    fs["corp_adv_lag_z12m"] = _asof(idx, cdf.set_index("available")["corp_adv_z12m"])
    integrity["corp_adv_check_sep2026"] = None
    row = cdf[cdf["month"] == "2026-09"]
    if len(row):
        integrity["corp_adv_check_sep2026"] = {
            "trading_days": int(row["trading_days"].iloc[0]),
            "corp_adv_usd_m": float(row["corp_adv"].iloc[0] * 1000.0),
            "finra_published_avg_daily_par_usd_m": 60196.1,
        }

    # --- Auctions ------------------------------------------------------------
    integrity["auction_features"] = False
    if auctions is not None and len(auctions):
        a = auctions.dropna(subset=["auction_date"]).copy()
        by_day = a.groupby(a["auction_date"].dt.normalize()).agg(
            amt=("offering_amt", "sum"), n=("offering_amt", "size"))
        fs["auction_day"] = 0
        hit = fs.index.isin(by_day.index)
        fs.loc[hit, "auction_day"] = 1
        fs["auction_amt"] = 0.0
        fs.loc[hit, "auction_amt"] = (
            by_day["amt"].reindex(fs.index[hit]).values / 1e9)
        # btc_last: mean BTC of the most recent auction date <= t with a
        # reported (non-null) BTC, forward-filled. PIT: BTC is published
        # on the auction date itself.
        btc = (a.dropna(subset=["bid_to_cover_ratio"])
               .groupby(a["auction_date"].dt.normalize())["bid_to_cover_ratio"]
               .mean())
        fs["btc_last"] = _asof(idx, btc)
        # refunding week: a Mon-Sun week whose auctions include a 3-Year
        # note, a 10-Year note and a 30-Year bond (the quarterly
        # refunding pattern). The schedule is announced in advance, so
        # classifying the whole week is point-in-time legitimate.
        a["monday"] = (a["auction_date"].dt.normalize()
                        - pd.to_timedelta(a["auction_date"].dt.weekday, unit="D"))
        terms = a.groupby("monday")["security_term"].agg(lambda s: set(s.dropna()))
        refund_mondays = {m for m, ts in terms.items()
                          if {"3-Year", "10-Year", "30-Year"} <= ts}
        day_monday = (idx - pd.to_timedelta(idx.weekday, unit="D"))
        fs["refunding_week"] = pd.Series(
            [1 if m in refund_mondays else 0 for m in day_monday],
            index=idx, dtype=int)
        integrity["auction_features"] = True
        integrity["auction_records"] = int(len(a))
        integrity["refunding_weeks"] = len(refund_mondays)
    return fs


def build_all(*, trace=None, etf_prices=None, oas_ig=None, oas_hy=None,
              t10y2y=None, dealer=None, monthly=None, auctions=None,
              refresh_fred: bool = False, refresh_auctions: bool = False
              ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Assemble panel + feature store from disk (or supplied inputs)."""
    if trace is None:
        trace = load_trace_daily()
    if etf_prices is None:
        etf_prices = load_etf_prices(
            ["SHY", "IEF", "TLT", "TIP", "^TNX", "^VIX"])
    if oas_ig is None:
        oas_ig = fetch_fred_series("BAMLC0A0CM", refresh=refresh_fred)
    if oas_hy is None:
        oas_hy = fetch_fred_series("BAMLH0A0HYM2", refresh=refresh_fred)
    if t10y2y is None:
        t10y2y = fetch_fred_series("T10Y2Y", refresh=refresh_fred)
    if dealer is None:
        dealer = load_dealer_tot()
    if monthly is None:
        monthly = load_monthly()
    if auctions is None:
        auctions = fetch_auctions(refresh=refresh_auctions)
    panel, integrity = build_panel(trace)
    fs = build_features(panel, etf_prices=etf_prices, oas_ig=oas_ig,
                        oas_hy=oas_hy, t10y2y=t10y2y, dealer=dealer,
                        monthly=monthly, auctions=auctions,
                        integrity=integrity)
    return panel, fs, integrity


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ML_DIR.mkdir(parents=True, exist_ok=True)
    panel, fs, integrity = build_all(refresh_fred=True,
                                     refresh_auctions=False)
    panel.to_parquet(ML_DIR / "liquidity_panel.parquet")
    fs.to_parquet(ML_DIR / "feature_store.parquet")
    (ML_DIR / "integrity.json").write_text(json.dumps(integrity, indent=1,
                                                      default=str))
    print(f"panel: {len(panel)} rows {panel.index.min().date()} -> "
          f"{panel.index.max().date()}")
    print(f"feature store: {fs.shape[1]} columns")
    print(json.dumps(integrity, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
