"""Tests for the 2026-10-20 illiq_agg NaN incident fix.

Incident (diagnosed 2026-10-25): the Yahoo cache staleness rule in
engine.data.load_or_fetch refreshed only when the cache was more than
3 CALENDAR days old. On the Wed 2026-10-21 daily ML run, the SHY/IEF/
TLT caches still ended Mon 2026-10-19, so Tue 2026-10-20's bars were
never fetched; the panel builder reindexed the missing day as NaN and
illiq_agg came out NaN on 2026-10-20. run_ml_daily then silently fell
back to the last complete as-of date (2026-10-16) and issued 0 with
exit 0 — for four consecutive runs.

These tests pin both halves of the fix:
  1. load_or_fetch judges staleness in completed weekday sessions.
  2. run_ml_daily.nan_guard withholds issuing entirely (no fallback)
     and surfaces a failure banner when the last panel date is
     NaN-tainted; a clean panel issues normally.

Fixture-based, no network: the HTTP/fetch layer is monkeypatched.
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

import engine.data as data
import run_ml_daily
from engine import ml_harness as H


# ---------------------------------------------------------------------------
# 1. Fetcher staleness (engine.data.load_or_fetch)
# ---------------------------------------------------------------------------
def _bars(ending: date, n: int = 10) -> pd.DataFrame:
    idx = pd.bdate_range(end=ending, periods=n)
    return pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
         "adjclose": 100.5, "volume": 1000, "dividends": 0.0},
        index=idx,
    )


@pytest.fixture()
def cache_env(tmp_path, monkeypatch):
    """A cache parquet on disk + a recording fake fetcher."""
    calls: list[str] = []

    def fake_fetch(ticker, start=None, end=None):
        calls.append(ticker)
        return _bars(date(2026, 10, 23), n=30)

    monkeypatch.setattr(data, "fetch_yahoo_daily", fake_fetch)
    path = tmp_path / "TEST.parquet"
    monkeypatch.setattr(data, "cache_path", lambda t: path)
    return path, calls


def _seed_cache(path, ending: date) -> None:
    _bars(ending).to_parquet(path)


def test_fetcher_refreshes_when_a_completed_weekday_is_missing(cache_env):
    """THE incident case: cache ends Mon 2026-10-19, run on Wed
    2026-10-21. Tue 2026-10-20 is a completed session with no bar —
    the old >3-calendar-day rule skipped this fetch and produced the
    illiq_agg NaN. The new rule must refresh."""
    path, calls = cache_env
    _seed_cache(path, date(2026, 10, 19))
    data.load_or_fetch("TEST", end=date(2026, 10, 21))
    assert calls == ["TEST"]


def test_fetcher_refreshes_across_a_long_weekend_gap(cache_env):
    """Cache ends Thu 2026-10-15, run Tue 2026-10-20: Fri 10-16 and
    Mon 10-19 are completed sessions — refresh (this is the fetch that
    actually fired during the incident week)."""
    path, calls = cache_env
    _seed_cache(path, date(2026, 10, 15))
    data.load_or_fetch("TEST", end=date(2026, 10, 20))
    assert calls == ["TEST"]


def test_fetcher_does_not_refresh_over_a_plain_weekend(cache_env):
    """Cache ends Fri 2026-10-16, run Mon 2026-10-19 before the close:
    no completed weekday is missing — no refresh."""
    path, calls = cache_env
    _seed_cache(path, date(2026, 10, 16))
    df = data.load_or_fetch("TEST", end=date(2026, 10, 19))
    assert calls == []
    assert df.index[-1].date() == date(2026, 10, 16)


def test_fetcher_does_not_refresh_on_saturday(cache_env):
    path, calls = cache_env
    _seed_cache(path, date(2026, 10, 16))  # Friday
    data.load_or_fetch("TEST", end=date(2026, 10, 17))  # Saturday
    assert calls == []


def test_fetcher_does_not_refresh_when_cache_covers_last_completed_day(cache_env):
    """Cache ends Wed 2026-10-21, run Thu 2026-10-22 (Thu not yet a
    completed session) — no refresh."""
    path, calls = cache_env
    _seed_cache(path, date(2026, 10, 21))
    data.load_or_fetch("TEST", end=date(2026, 10, 22))
    assert calls == []


def test_missing_completed_weekdays_helper():
    f = data._missing_completed_weekdays
    assert f(date(2026, 10, 19), date(2026, 10, 21)) == [date(2026, 10, 20)]
    assert f(date(2026, 10, 16), date(2026, 10, 19)) == []  # Fri -> Mon
    assert f(date(2026, 10, 16), date(2026, 10, 18)) == []  # Fri -> Sun
    assert f(date(2026, 10, 15), date(2026, 10, 20)) == [
        date(2026, 10, 16), date(2026, 10, 19)]


# ---------------------------------------------------------------------------
# 2. nan_guard unit tests
# ---------------------------------------------------------------------------
def _panel_and_X(last_illiq=np.nan, last_chg5=0.1):
    idx = pd.bdate_range(end="2026-10-21", periods=5)
    panel = pd.DataFrame({"illiq_agg": [1e-5, 1.1e-5, 0.9e-5, 1.2e-5, last_illiq]},
                         index=idx)
    X = pd.DataFrame({name: [0.1] * 5 for name in H.FEATURE_NAMES}, index=idx)
    X["illiq_chg5"] = [0.1, 0.1, 0.1, 0.1, last_chg5]
    if isinstance(last_illiq, float) and np.isnan(last_illiq):
        X.loc[idx[-1], "log_illiq_agg"] = np.nan
    return panel, X


def test_guard_trips_on_nan_illiq_last_date():
    panel, X = _panel_and_X(last_illiq=np.nan)
    g = run_ml_daily.nan_guard(panel, X)
    assert g["tripped"] is True
    assert g["illiq_agg_nan"] is True
    assert "2026-10-21" in g["banner"]
    assert "illiq_agg is NaN" in g["banner"]
    assert "ISSUANCE WITHHELD" in g["banner"]


def test_guard_trips_on_incomplete_feature_row():
    """The 2026-10-19 case: illiq_agg itself is fine, but illiq_chg5 is
    NaN (its 5-day change references the 2026-10-12 holiday-stub NaN).
    Issuing must still be withheld, not silently moved to another date."""
    panel, X = _panel_and_X(last_illiq=1.3e-5, last_chg5=np.nan)
    g = run_ml_daily.nan_guard(panel, X)
    assert g["tripped"] is True
    assert g["illiq_agg_nan"] is False
    assert "illiq_chg5" in g["banner"]


def test_guard_passes_clean_last_row_despite_historical_nans():
    """Historical stub-day NaNs are by design and must NOT trip the
    guard — only the last panel date matters."""
    panel, X = _panel_and_X(last_illiq=1.3e-5)
    panel.loc[panel.index[1], "illiq_agg"] = np.nan  # a past stub day
    X.loc[X.index[1], "log_illiq_agg"] = np.nan
    g = run_ml_daily.nan_guard(panel, X)
    assert g["tripped"] is False
    assert g["banner"] is None


# ---------------------------------------------------------------------------
# 3. Runner preview: run_live with the guard, end to end (all I/O faked
#    except the real feature/label/entry machinery on synthetic data)
# ---------------------------------------------------------------------------
def _synthetic_store(n: int = 800, nan_last_illiq: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.bdate_range(end="2026-10-21", periods=n)
    illiq = 1.5e-5 * np.exp(rng.normal(0, 0.4, n))
    fs = pd.DataFrame(index=idx)
    fs["illiq_agg"] = illiq
    fs["illiq_agg_z252"] = rng.normal(0, 1, n)
    fs["illiq_chg5"] = rng.normal(0, 0.3, n)
    fs["adv_chg5"] = rng.normal(0, 0.2, n)
    fs["onrun_share_agg"] = rng.uniform(0.5, 0.75, n)
    fs["ats_share_agg"] = rng.uniform(0.45, 0.6, n)
    fs["share_3_5_agg"] = rng.uniform(0.15, 0.3, n)
    fs["oas_hy"] = 3.1 + rng.normal(0, 0.05, n).cumsum() * 0.1
    fs["oas_hy_chg5d"] = rng.normal(0, 0.05, n)
    fs["oas_hy_z252"] = rng.normal(0, 1, n)
    fs["tnx_chg63"] = rng.normal(0, 0.3, n)
    fs["vix"] = rng.uniform(13, 22, n)
    fs["slope_t10y2y"] = rng.uniform(0.1, 0.6, n)
    fs["auction_day"] = rng.integers(0, 2, n)
    if nan_last_illiq:
        fs.loc[idx[-1], "illiq_agg"] = np.nan
    return fs


def _write_ml_dir(tmp_path, fs: pd.DataFrame) -> None:
    fs.to_parquet(tmp_path / "feature_store.parquet")
    idx = pd.bdate_range(start=fs.index.min(), periods=len(fs) + 15)
    hy = pd.DataFrame({"observation_date": idx,
                       "BAMLH0A0HYM2": 3.1 + np.arange(len(idx)) * 0.001})
    ig = pd.DataFrame({"observation_date": idx,
                       "BAMLC0A0CM": 0.9 + np.arange(len(idx)) * 0.0005})
    hy.to_csv(tmp_path / "oas_hy.csv", index=False)
    ig.to_csv(tmp_path / "oas_ig.csv", index=False)
    (tmp_path / "integrity.json").write_text(
        json.dumps({"recon_failures": [], "stub_days": []}))


@pytest.fixture()
def live_env(tmp_path, monkeypatch):
    """run_live with every external effect faked; returns a recorder."""
    rec = {"appended": [], "shadow_calls": 0}

    monkeypatch.setattr(H, "ML_DIR", tmp_path)
    monkeypatch.setattr(run_ml_daily, "_refresh_trace_updater", lambda: None)
    monkeypatch.setattr(run_ml_daily, "_refresh_yahoo_cache", lambda: None)

    from engine import ml_c3_shadow, ml_panel, ml_scorecard, ml_shadow, ml_t2_shadow, ml_t4a_daily

    monkeypatch.setattr(ml_panel, "main", lambda: None)
    monkeypatch.setattr(ml_scorecard, "score_pending", lambda panel, oas: [])
    monkeypatch.setattr(ml_scorecard, "running_record", lambda: [])

    def fake_append(entries):
        rec["appended"].append(entries)
        return len(entries)

    monkeypatch.setattr(ml_scorecard, "append_entries", fake_append)

    def fake_shadow(panel, oas):
        rec["shadow_calls"] += 1
        return {"scored_count": 0, "issued_count": 8, "issued_ids": [],
                "asof": "2026-10-21"}

    monkeypatch.setattr(ml_shadow, "run_shadow", fake_shadow)
    monkeypatch.setattr(
        ml_t4a_daily, "run_t4a",
        lambda: {"scored_count": 0, "issued_count": 3, "issued_ids": [],
                 "asof": "2026-10-21"})
    monkeypatch.setattr(
        ml_t2_shadow, "run_t2",
        lambda: {"scored_count": 0, "issued_count": 0, "issued_ids": [],
                 "auctions": []})
    monkeypatch.setattr(
        ml_c3_shadow, "run_c3",
        lambda: {"scored_count": 0, "issued_count": 3, "issued_ids": [],
                 "asof": "2026-10-21"})
    return rec


def test_run_live_withholds_and_banners_on_nan(live_env, tmp_path, capsys):
    _write_ml_dir(tmp_path, _synthetic_store(nan_last_illiq=True))
    rc = run_ml_daily.run_live()
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["status"] == "WITHHELD_NAN_GUARD"
    assert out["issued_count"] == 0
    assert "ISSUANCE WITHHELD" in out["failure_banner"]
    assert "illiq_agg is NaN" in out["failure_banner"]
    assert "2026-10-21" in out["failure_banner"]
    assert out["failure_banner"] in out["integrity_flags"]
    assert out["shadow"] == {"status": "withheld_nan_guard"}
    assert live_env["appended"] == []          # nothing was issued
    assert live_env["shadow_calls"] == 0       # shadow withheld too
    assert out["t4a"]["status"] == "ok"        # unrelated lane still runs


def test_run_live_issues_normally_on_clean_panel(live_env, tmp_path, capsys):
    _write_ml_dir(tmp_path, _synthetic_store(nan_last_illiq=False))
    rc = run_ml_daily.run_live()
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["status"] == "OK"
    assert out["issued_count"] == 16
    assert len(live_env["appended"]) == 1
    assert len(live_env["appended"][0]) == 16
    assert live_env["shadow_calls"] == 1
    assert "failure_banner" not in out
