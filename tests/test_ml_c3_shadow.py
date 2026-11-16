"""Tests for the ML Stage C-3 leg V-b shadow forward scorecard
(approved by Harry 2026-10-27).

Covers: c3s- ID parsing through the broadened scorecard regex; the
batch shape (3 entries — the leg V-b REDUCED ridge on
|label_ret5| plus its two frozen baselines, trailing mean and
seasonal climatology for absolute returns; no gradient-boosting
entry anywhere); the inputs snapshot carrying exactly the frozen
18 REDUCED features; the forward baselines coinciding with the
backtest's expanding-baseline rule on in-sample positions;
duplicate append refusal; scoring that waits for the 5-day
horizon to mature and grades the absolute return; the grading
window arithmetic (Target 4a's, reused); and the failure
isolation of run_ml_daily step 5f (a C-3 failure never breaks
the v1 pass).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine import ml_scorecard as SC
from engine import ml_c3_shadow as S
from engine import ml_stage_c3 as C3
from engine import ml_t4a as T
from engine import ml_t4a_daily as D

REDUCED = list(T.REDUCED_FEATURES)
FORECAST_BLOCK = {"fc_hdd7", "fc_cdd7", "fc_dd_anom7", "fc_rev"}


def _synthetic_panel(n: int = 900, seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    close = 3.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    lab = np.full(n, np.nan)
    lab[:-5] = np.log(close[5:] / close[:-5])
    panel = pd.DataFrame({"date": dates, "ng_close": close,
                          "label_ret5": lab, "label_ret5_pct": 100 * lab,
                          "roll_window": False})
    for f in REDUCED:
        panel[f] = rng.normal(size=n)
    return panel


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return _synthetic_panel()


@pytest.fixture(scope="module")
def entries(panel):
    return S.build_c3_entries(panel)


# ---------------------------------------------------------------------------
# ID parsing / batch shape / frozen snapshot
# ---------------------------------------------------------------------------
def test_c3_id_parsing(entries):
    ents, meta = entries
    e = ents[0]
    assert e["prediction_id"].startswith("c3s-")
    text = "# header\n\n" + S.format_entry_c3(e) + "\n"
    parsed = SC.parse_entries(text)
    assert len(parsed) == 1
    p = parsed[0]
    assert p["prediction_id"] == e["prediction_id"]
    assert p["target_key"] == "C3-Vb"
    assert p["asof"] == meta["asof"]
    assert p["outcome"] == "pending"


def test_batch_shape_three_way_no_gbm(entries):
    ents, _meta = entries
    assert len(ents) == 3
    assert [e["model_version"] for e in ents] == [
        S.RIDGE_VERSION, S.TRAIL_VERSION, S.SEAS_VERSION]
    assert not any("gbm" in e["model_version"] for e in ents)
    assert [e["prediction_id"].rsplit("-", 1)[1] for e in ents] == ["01", "02", "03"]
    assert set(ents[0]["prediction"]) == {"point", "alpha"}
    # A move-size prediction is non-negative by construction of the
    # baselines; the ridge point is a fitted value (not asserted).
    assert ents[1]["prediction"] >= 0.0
    assert ents[2]["prediction"] >= 0.0


def test_snapshot_exactly_the_frozen_18(entries):
    ents, _meta = entries
    for e in ents:
        feats = e["inputs_snapshot"]["features"]
        assert list(feats) == REDUCED
        assert not (set(feats) & FORECAST_BLOCK)
        assert "roll_window" in e["inputs_snapshot"]
        assert "season" in e["inputs_snapshot"]


# ---------------------------------------------------------------------------
# Forward baselines == the backtest's expanding-baseline rule
# ---------------------------------------------------------------------------
def test_forward_baselines_match_backtest_rule(panel):
    ext = D.build_extended(panel).copy()
    ext["y_abs"] = ext["label_ret5"].abs()
    sample = C3.build_vb_sample(panel)
    for pos in (100, 400):
        trail_fwd, seas_fwd = S._forward_baselines(ext, pos)
        trail_bt = C3._expanding_baseline(sample, np.array([pos]), seasonal=False)[0]
        seas_bt = C3._expanding_baseline(sample, np.array([pos]), seasonal=True)[0]
        assert trail_fwd == trail_bt
        assert seas_fwd == seas_bt


# ---------------------------------------------------------------------------
# Header criteria / grading window
# ---------------------------------------------------------------------------
def test_header_and_grading_window(tmp_path):
    path = tmp_path / "scorecard_c3.md"
    S.ensure_c3_file(first_asof="2026-10-23", path=path)
    text = path.read_text()
    assert "Grading happens once, after 63 trading days of issuance" in text
    assert "First issuance: 2026-10-23" in text
    assert "Expected 63rd issuance day: 2027-01-26" in text
    assert "seasonal-climatology baseline's at grading" in text
    assert "gradient-boosting configuration FAILED" in text
    assert "- last_c3_run: none" in text


# ---------------------------------------------------------------------------
# Append / duplicate refusal / scoring maturity on the absolute label
# ---------------------------------------------------------------------------
def test_duplicate_append_refused(entries, tmp_path):
    ents, meta = entries
    path = tmp_path / "scorecard_c3.md"
    S.ensure_c3_file(first_asof=meta["asof"], path=path)
    assert S.append_entries_c3(ents, path=path) == 3
    before = path.read_text()
    with pytest.raises(ValueError, match="duplicate prediction_id"):
        S.append_entries_c3(ents, path=path)
    assert path.read_text() == before
    assert len(SC.parse_entries(path.read_text())) == 3


def test_scoring_waits_for_maturity_and_grades_absolute_return(entries, panel, tmp_path):
    ents, meta = entries
    path = tmp_path / "scorecard_c3.md"
    S.ensure_c3_file(first_asof=meta["asof"], path=path)
    S.append_entries_c3(ents, path=path)
    assert S.score_pending_c3(panel, path=path) == []

    extra_dates = pd.bdate_range(panel["date"].iloc[-1] + pd.Timedelta(days=1), periods=6)
    rng = np.random.default_rng(7)
    extra_close = panel["ng_close"].iloc[-1] * np.exp(np.cumsum(rng.normal(0, 0.02, 6)))
    extra = pd.DataFrame({"date": extra_dates, "ng_close": extra_close,
                          "roll_window": False})
    for f in REDUCED:
        extra[f] = rng.normal(size=6)
    ext_panel = pd.concat([panel, extra], ignore_index=True)
    closes = ext_panel["ng_close"].to_numpy(dtype=float)
    lab = np.full(len(ext_panel), np.nan)
    lab[:-5] = np.log(closes[5:] / closes[:-5])
    ext_panel["label_ret5"] = lab
    ext_panel["label_ret5_pct"] = 100 * lab

    scored = S.score_pending_c3(ext_panel, path=path)
    assert len(scored) == 3
    parsed = SC.parse_entries(path.read_text())
    signed = float(ext_panel.set_index("date")["label_ret5"].loc[pd.Timestamp(meta["asof"])])
    realized_abs = abs(signed)
    rec = {r["model_version"]: r for r in S.running_record_c3(path=path)}
    assert set(rec) == {S.RIDGE_VERSION, S.TRAIL_VERSION, S.SEAS_VERSION}
    for e in parsed:
        point = SC._prediction_point(e["fields"]["prediction"])
        assert float(e["outcome"]) == realized_abs
        assert rec[e["model_version"]]["value"] == pytest.approx(abs(point - realized_abs))


# ---------------------------------------------------------------------------
# Failure isolation: run_ml_daily step 5f must never break the v1 pass
# ---------------------------------------------------------------------------
def test_run_live_c3_failure_isolated(monkeypatch, tmp_path, capsys):
    import run_ml_daily
    from engine import ml_panel, ml_shadow, ml_t2_shadow, ml_t4a_daily

    path = tmp_path / "prediction_scorecard.md"
    path.write_text(Path(SC.SCORECARD_PATH).read_text())
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
        "asof": "2026-10-23"})
    monkeypatch.setattr(ml_t4a_daily, "run_t4a", lambda: {
        "scored_count": 0, "issued_count": 0, "issued_ids": [],
        "asof": "2026-10-23"})
    monkeypatch.setattr(ml_t4a_daily, "record_t4a_error", lambda msg: None)
    monkeypatch.setattr(ml_t2_shadow, "run_t2", lambda: {
        "scored_count": 0, "issued_count": 0, "issued_ids": [], "auctions": []})
    monkeypatch.setattr(ml_t2_shadow, "record_t2_error", lambda msg: None)

    def _c3_boom():
        raise RuntimeError("c3 boom (test)")

    monkeypatch.setattr(S, "run_c3", _c3_boom)
    monkeypatch.setattr(S, "record_c3_error", lambda msg: None)

    rc = run_ml_daily.run_live()
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["shadow"]["status"] == "ok"
    assert summary["t4a"]["status"] == "ok"
    assert summary["t2"]["status"] == "ok"
    assert summary["c3"]["status"] == "failed"
    assert "c3 boom" in summary["c3"]["error"]
