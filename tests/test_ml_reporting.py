"""Tests for Stage A reporting wiring (E2 power + E3 calibration).

Covers engine/ml_reporting.py: effective-outcome accounting, the
exact binomial alert MDE, the paired-difference Brier/MAE MDE,
reliability bins / calibration intercept-slope / expected
calibration error against hand-computed values, the standing block
built from stored scorecard entries, content preservation of
issued entries under a header refresh (additive header only,
idempotent), and the hard failure-isolation requirement: a forced
reporting failure never raises, never blocks the v1 daily pass,
and is recorded as a note in the affected file's header.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from engine import ml_reporting as R
from engine import ml_scorecard as SC

Z = float(stats.norm.ppf(0.95)) + float(stats.norm.ppf(0.80))


# ---------------------------------------------------------------------------
# E2 — pure power computations
# ---------------------------------------------------------------------------
def test_effective_outcomes_convention():
    assert R.effective_outcomes(60, 5) == 12.0
    assert R.effective_outcomes(6, 1) == 6.0
    assert R.effective_outcomes(0, 5) == 0.0


def test_mde_paired_matches_frozen_formula():
    diffs = [0.01, -0.02, 0.03, 0.0, 0.005]
    res = R.mde_paired(diffs, horizon=5)
    sd = float(np.std(diffs, ddof=1))
    expected = Z * sd / math.sqrt(len(diffs) / 5)
    assert res["n_paired"] == 5
    assert res["n_eff"] == pytest.approx(1.0)
    assert res["sd"] == pytest.approx(sd)
    assert res["mde"] == pytest.approx(expected)


def test_mde_paired_too_few_pairs():
    res = R.mde_paired([0.01], horizon=5)
    assert res["mde"] is None
    assert "fewer than 2" in res["reason"]


def test_mde_alert_power_property():
    res = R.mde_alert(base_rate=0.20, n_alert_effective=20.0)
    assert res["reason"] is None
    assert res["n_alert_eff_int"] == 20
    k = res["critical_k"]
    # The critical count itself must reject at the frozen 0.05 level.
    assert float(stats.binom.sf(k - 1, 20, 0.20)) <= 0.05
    assert float(stats.binom.sf(k - 2, 20, 0.20)) > 0.05
    # Power at the MDE hit rate is at least 0.80; a hit rate one
    # percentage point lower does not reach it (bisection tolerance).
    p1 = res["mde_hit_rate"]
    assert float(stats.binom.sf(k - 1, 20, p1)) >= 0.80
    assert float(stats.binom.sf(k - 1, 20, p1 - 0.01)) < 0.80
    assert res["mde_excess"] == pytest.approx(p1 - 0.20)


def test_mde_alert_below_one_effective_alert():
    res = R.mde_alert(base_rate=0.20, n_alert_effective=0.2)
    assert res["mde_excess"] is None
    assert "below one independent alert" in res["reason"]


def test_mde_alert_unattainable_when_no_rejection_possible():
    # Base rate 0.9 with a single alert: even a hit has p = 0.9.
    res = R.mde_alert(base_rate=0.9, n_alert_effective=1.0)
    assert res["mde_excess"] is None
    assert "no rejection is possible" in res["reason"]


# ---------------------------------------------------------------------------
# E3 — calibration computations, hand-checked
# ---------------------------------------------------------------------------
def test_reliability_bins_and_ece_handcomputed():
    probs = [0.05, 0.07, 0.15, 0.85]
    outcomes = [0, 1, 0, 1]
    bins = R.reliability_bins(probs, outcomes)
    assert [(b["lo"], b["hi"]) for b in bins] == [(0.0, 0.1), (0.1, 0.2), (0.8, 0.9)]
    b0, b1, b2 = bins
    assert b0["n"] == 2 and b0["mean_forecast"] == pytest.approx(0.06)
    assert b0["realized_frequency"] == pytest.approx(0.5)
    assert b1["n"] == 1 and b1["realized_frequency"] == 0.0
    assert b2["n"] == 1 and b2["realized_frequency"] == 1.0
    ece = R.expected_calibration_error(bins, 4)
    assert ece == pytest.approx((2 / 4) * 0.44 + (1 / 4) * 0.15 + (1 / 4) * 0.15)


def test_calibration_intercept_slope_well_calibrated():
    # Two forecast groups whose realized frequencies match the
    # forecasts exactly: intercept 0, slope 1 (up to discreteness).
    probs = [0.25] * 100 + [0.75] * 100
    outcomes = ([1] * 25 + [0] * 75) + ([1] * 75 + [0] * 25)
    fit = R.calibration_intercept_slope(probs, outcomes)
    assert fit["reason"] is None
    assert fit["intercept"] == pytest.approx(0.0, abs=0.05)
    assert fit["slope"] == pytest.approx(1.0, abs=0.05)


def test_calibration_single_class_not_computable():
    fit = R.calibration_intercept_slope([0.2, 0.4, 0.6, 0.8], [0, 0, 0, 0])
    assert fit["intercept"] is None and fit["slope"] is None
    assert "single class" in fit["reason"]


def test_calibration_constant_forecast_not_computable():
    fit = R.calibration_intercept_slope([0.3] * 6, [0, 1, 0, 1, 0, 1])
    assert fit["slope"] is None
    assert "constant" in fit["reason"]


# ---------------------------------------------------------------------------
# Block construction from stored entries
# ---------------------------------------------------------------------------
def _entry_block(pid_date, target, model, prediction, seq, outcome, error,
                 target_descr=None):
    feats = {name: 0.5 for name in
             ["log_illiq_agg", "illiq_agg_z252", "illiq_chg5", "adv_chg5",
              "onrun_share_agg", "ats_share_agg", "share_3_5_agg", "oas_hy",
              "oas_hy_chg5d", "oas_hy_z252", "tnx_chg63", "vix",
              "slope_t10y2y", "auction_day"]}
    feats["illiq_t"] = 1.5e-05
    e = {
        "prediction_id": f"{pid_date}-{target}-{seq:02d}",
        "issued_at": "2026-10-13T23:00:00-04:00",
        "target": target_descr or f"{target}: test target",
        "horizon": "5 trading days",
        "model_version": model,
        "inputs_snapshot": {
            "asof": pid_date,
            "features": feats,
            "data_vintage": {
                "feature_store_through": pid_date,
                "feature_store_rows": 600,
                "oas_through": pid_date,
                "sample_rows": 500,
                "sample_through": pid_date,
            },
        },
        "prediction": prediction,
        "outcome": "pending",
        "error": "pending",
    }
    block = SC.format_entry(e)
    block = block.replace("- outcome: pending", f"- outcome: {outcome}")
    block = block.replace("- error: pending", f"- error: {error}")
    return block


def _synthetic_scorecard_text() -> str:
    blocks = []
    seq = 0
    # Four settled days: climatology at the base rate 0.25, a
    # logistic model with varied probabilities, outcomes 0/1/0/1.
    days = ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04"]
    events = [0, 1, 0, 1]
    logi_probs = [0.20, 0.60, 0.30, 0.70]
    for d, y, lp in zip(days, events, logi_probs):
        seq += 1
        blocks.append(_entry_block(
            d, "T1b_HY", "baseline:climatology",
            {"probability": 0.25, "threshold_pp": 0.09}, seq,
            f"widening_pp=0.01, event={y}", f"brier={(0.25 - y) ** 2!r}",
            target_descr="T1b_HY: HY OAS widens by >= threshold over t+1..t+5"))
        seq += 1
        blocks.append(_entry_block(
            d, "T1b_HY", "logistic:v1-frozen14",
            {"probability": lp, "threshold_pp": 0.09}, seq,
            f"widening_pp=0.01, event={y}", f"brier={(lp - y) ** 2!r}",
            target_descr="T1b_HY: HY OAS widens by >= threshold over t+1..t+5"))
    return "# Synthetic scorecard\n\n" + "\n\n".join(blocks) + "\n"


def _entries_segment(text: str) -> str:
    lines = text.splitlines(keepends=True)
    idx = next(i for i, ln in enumerate(lines) if SC._ID_RE.match(ln.rstrip("\n")))
    return "".join(lines[idx:])


def test_build_block_power_and_calibration_lines():
    entries = SC.parse_entries(_synthetic_scorecard_text())
    block = "\n".join(R.build_reporting_block(entries))
    # Power: settled + effective (4 settled / 5-day horizon = 0.8).
    assert "T1b_HY / logistic:v1-frozen14: settled predictions = 4" in block
    assert "effective independent outcomes = 0.8" in block
    assert "Brier leg: minimum detectable Brier improvement over climatology" in block
    assert "Alert leg (T1b_HY / logistic:v1-frozen14)" in block
    # The MDE exceeds the (tiny) v1 logistic backtest Brier effect,
    # and the block must state that plainly.
    assert "This MDE exceeds the corresponding backtest effect" in block
    # Calibration: descriptive, effective count attached, ECE with
    # its binning, and never a graduation leg.
    assert "descriptive; settled = 4, effective independent outcomes = 0.8" in block
    assert "Expected calibration error" in block
    assert "10 equal-width bins on [0, 1]" in block
    assert "not a graduation leg" in block
    assert "zero trials" in block


def test_refresh_preserves_entries_and_is_idempotent(tmp_path):
    path = tmp_path / "scorecard.md"
    path.write_text(_synthetic_scorecard_text())
    before_segment = _entries_segment(path.read_text())
    before_hash = hashlib.sha256(before_segment.encode()).hexdigest()

    res = R.refresh_scorecard_reporting(path)
    assert res["status"] == "ok"
    text1 = path.read_text()
    assert _entries_segment(text1) == before_segment
    assert hashlib.sha256(_entries_segment(text1).encode()).hexdigest() == before_hash
    assert text1.count(R.BEGIN_MARKER) == 1 and text1.count(R.END_MARKER) == 1
    assert "- ml_reporting_status: ok" in text1
    assert "- ml_reporting_error: none" in text1

    res2 = R.refresh_scorecard_reporting(path)
    assert res2["status"] == "ok"
    assert path.read_text() == text1  # idempotent: no growth, no drift


# ---------------------------------------------------------------------------
# Failure isolation (hard requirement)
# ---------------------------------------------------------------------------
def test_refresh_failure_isolated_and_noted_in_header(tmp_path, monkeypatch):
    path = tmp_path / "scorecard.md"
    path.write_text(_synthetic_scorecard_text())
    assert R.refresh_scorecard_reporting(path)["status"] == "ok"
    good_text = path.read_text()
    good_block = good_text.split(R.BEGIN_MARKER)[1].split(R.END_MARKER)[0]

    def _boom(entries):
        raise RuntimeError("calibration boom (test)")

    monkeypatch.setattr(R, "build_reporting_block", _boom)
    res = R.refresh_scorecard_reporting(path)  # must not raise
    assert res["status"] == "failed"
    assert "calibration boom (test)" in res["error"]
    text = path.read_text()
    # The failure is recorded as a note in the file header...
    header = text.split("## 2026-")[0]
    assert "- ml_reporting_status: failed" in header
    assert "calibration boom (test)" in header
    # ...the pre-existing block is left untouched...
    assert text.split(R.BEGIN_MARKER)[1].split(R.END_MARKER)[0] == good_block
    # ...and every issued entry is byte-unchanged.
    assert _entries_segment(text) == _entries_segment(good_text)


def test_run_live_reporting_failure_isolated(monkeypatch, tmp_path, capsys):
    import run_ml_daily
    from engine import ml_panel, ml_shadow, ml_t4a_daily as D

    path = tmp_path / "prediction_scorecard.md"
    path.write_text(_synthetic_scorecard_text())
    monkeypatch.setattr(SC, "SCORECARD_PATH", path)

    monkeypatch.setattr(run_ml_daily, "_refresh_trace_updater", lambda: None)
    monkeypatch.setattr(run_ml_daily, "_refresh_yahoo_cache", lambda: None)
    monkeypatch.setattr(ml_panel, "main", lambda: None)
    monkeypatch.setattr(run_ml_daily, "build_scorecard_entries",
                        lambda panel, oas, asof=None: [])
    monkeypatch.setattr(SC, "score_pending", lambda panel, oas: [])
    monkeypatch.setattr(SC, "append_entries", lambda entries: 0)
    monkeypatch.setattr(SC, "running_record", lambda: [])
    monkeypatch.setattr(ml_shadow, "run_shadow", lambda panel, oas: {
        "scored_count": 0, "issued_count": 0, "issued_ids": [],
        "asof": "2026-10-23", "reporting": {"status": "ok"}})
    monkeypatch.setattr(D, "run_t4a", lambda: {
        "scored_count": 0, "issued_count": 0, "issued_ids": [],
        "asof": "2026-10-23", "reporting": {"status": "ok"}})
    monkeypatch.setattr(D, "record_t4a_error", lambda msg: None)

    from engine import ml_c3_shadow, ml_t2_shadow

    monkeypatch.setattr(ml_t2_shadow, "run_t2", lambda: {
        "scored_count": 0, "issued_count": 0, "issued_ids": [], "auctions": []})
    monkeypatch.setattr(ml_t2_shadow, "record_t2_error", lambda msg: None)
    monkeypatch.setattr(ml_c3_shadow, "run_c3", lambda: {
        "scored_count": 0, "issued_count": 0, "issued_ids": [],
        "asof": "2026-10-23"})
    monkeypatch.setattr(ml_c3_shadow, "record_c3_error", lambda msg: None)

    def _boom(entries):
        raise RuntimeError("reporting boom (test)")

    monkeypatch.setattr(R, "build_reporting_block", _boom)

    rc = run_ml_daily.run_live()
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["reporting"]["status"] == "failed"
    assert "reporting boom (test)" in summary["reporting"]["error"]
    header = path.read_text().split("## 2026-")[0]
    assert "- ml_reporting_status: failed" in header
    assert "reporting boom (test)" in header
    # Issued entries in the v1 scorecard are byte-unchanged.
    assert _entries_segment(path.read_text()) == _entries_segment(
        _synthetic_scorecard_text())
