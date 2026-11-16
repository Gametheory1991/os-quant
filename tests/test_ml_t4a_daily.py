"""Tests for ML Target 4a daily forward issuance (approved by Harry
2026-10-24).

Covers: t4a- ID parsing through the broadened scorecard regex; the
v1/shadow scorecards still parsing to their entry counts; the batch
shape (3 entries — graduated REDUCED ridge + climatology +
zero-change); the inputs snapshot carrying exactly the frozen 18
REDUCED features (forecast block absent); the forward climatology
coinciding exactly with the spec's ml_t4a.climatology_predict on
in-sample dates; duplicate append refusal; scoring that waits for
the 5-day horizon to mature; and the failure-isolation of
run_ml_daily step 5c (a T4a failure never breaks the v1 pass).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine import ml_scorecard as SC
from engine import ml_t4a as T
from engine import ml_t4a_daily as D

REDUCED = list(T.REDUCED_FEATURES)
FORECAST_BLOCK = {"fc_hdd7", "fc_cdd7", "fc_dd_anom7", "fc_rev"}


def _synthetic_panel(n: int = 900, seed: int = 13) -> pd.DataFrame:
    """A panel-shaped frame (build_t4a column layout) with a seeded
    random-walk close and all REDUCED features populated. Only the
    label/doY mechanics matter to the functions under test."""
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
    ents, meta = D.build_t4a_entries(panel)
    return ents, meta


# ---------------------------------------------------------------------------
# ID parsing / no collateral change to the T1 + shadow scorecards
# ---------------------------------------------------------------------------
def test_t4a_id_parsing(entries):
    ents, meta = entries
    e = ents[0]
    assert e["prediction_id"].startswith("t4a-")
    text = "# header\n\n" + D.format_entry_t4a(e) + "\n"
    parsed = SC.parse_entries(text)
    assert len(parsed) == 1
    p = parsed[0]
    assert p["prediction_id"] == e["prediction_id"]
    assert p["target_key"] == "T4a-NG-5d"
    assert p["asof"] == meta["asof"]
    assert p["outcome"] == "pending"


def test_v1_and_shadow_scorecards_parse_unchanged():
    # Read-only: the broadened regex must count exactly the headings
    # each file's own convention counts (v1 IDs, v2s- IDs).
    t1 = Path(SC.SCORECARD_PATH).read_text()
    assert len(SC.parse_entries(t1)) == len(
        re.findall(r"^## \d{4}-\d{2}-\d{2}-\S+\s*$", t1, flags=re.M)
    )
    shadow_path = (
        Path.home() / "workspace/goals/quant-research-backtest-engine"
        "/knowledge/prediction_scorecard_v2_shadow.md"
    )
    sh = shadow_path.read_text()
    assert len(SC.parse_entries(sh)) == len(
        re.findall(r"^## v2s-\d{4}-\d{2}-\d{2}-\S+\s*$", sh, flags=re.M)
    )


# ---------------------------------------------------------------------------
# Batch shape + frozen snapshot
# ---------------------------------------------------------------------------
def test_batch_shape_three_way(entries):
    ents, _meta = entries
    assert len(ents) == 3
    assert [e["model_version"] for e in ents] == [
        D.RIDGE_VERSION, D.CLIM_VERSION, D.ZERO_VERSION,
    ]
    assert [e["prediction_id"].rsplit("-", 1)[1] for e in ents] == ["01", "02", "03"]
    assert all(e["prediction_id"].startswith("t4a-") for e in ents)
    assert ents[2]["prediction"] == 0.0  # zero-change
    assert set(ents[0]["prediction"]) == {"point", "alpha"}


def test_snapshot_exactly_the_frozen_18(entries):
    ents, _meta = entries
    for e in ents:
        feats = e["inputs_snapshot"]["features"]
        assert list(feats) == REDUCED  # exactly the frozen 18, spec order
        assert not (set(feats) & FORECAST_BLOCK)


# ---------------------------------------------------------------------------
# Forward climatology == the spec's climatology on in-sample dates
# ---------------------------------------------------------------------------
def test_climatology_forward_matches_spec(panel):
    sample = T.build_sample(panel, REDUCED)
    ext = D.build_extended(panel)
    y = sample[T.TARGET].to_numpy(dtype=float)
    for pos in (100, 400, len(sample) - 1):
        spec_pred = T.climatology_predict(sample, np.array([pos]))[0]
        fwd_pred = D.climatology_forward(ext, sample.index[pos])
        assert fwd_pred == spec_pred  # exact: identical eligible set + mean


# ---------------------------------------------------------------------------
# Append / duplicate refusal / scoring maturity
# ---------------------------------------------------------------------------
def test_duplicate_append_refused(entries, tmp_path):
    ents, meta = entries
    path = tmp_path / "scorecard_t4a.md"
    D.ensure_t4a_file(first_asof=meta["asof"], path=path)
    assert D.append_entries_t4a(ents, path=path) == 3
    before = path.read_text()
    with pytest.raises(ValueError, match="duplicate prediction_id"):
        D.append_entries_t4a(ents, path=path)
    assert path.read_text() == before
    assert len(SC.parse_entries(path.read_text())) == 3


def test_scoring_waits_for_maturity(entries, panel, tmp_path):
    ents, meta = entries
    path = tmp_path / "scorecard_t4a.md"
    D.ensure_t4a_file(first_asof=meta["asof"], path=path)
    D.append_entries_t4a(ents, path=path)
    # The batch's as-of date is the panel's last row: label NaN → nothing scores.
    assert D.score_pending_t4a(panel, path=path) == []
    assert all(e["outcome"] == "pending" for e in SC.parse_entries(path.read_text()))

    # Extend the panel by 6 trading days so the batch's label realizes.
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

    scored = D.score_pending_t4a(ext_panel, path=path)
    assert len(scored) == 3
    parsed = SC.parse_entries(path.read_text())
    realized = float(ext_panel.set_index("date")["label_ret5"].loc[pd.Timestamp(meta["asof"])])
    rec = {r["model_version"]: r for r in D.running_record_t4a(path=path)}
    assert set(rec) == {D.RIDGE_VERSION, D.CLIM_VERSION, D.ZERO_VERSION}
    for e in parsed:
        point = SC._prediction_point(e["fields"]["prediction"])
        assert float(e["outcome"]) == realized
        assert rec[e["model_version"]]["n_scored"] == 1
        assert rec[e["model_version"]]["value"] == pytest.approx(abs(point - realized))


# ---------------------------------------------------------------------------
# Grading window arithmetic
# ---------------------------------------------------------------------------
def test_grading_window_from_first_issuance():
    w = D.grading_window("2026-10-23")  # a Friday
    assert w["first_issuance"] == "2026-10-23"
    d63 = pd.Timestamp(w["issuance_day_63"])
    assert d63 > pd.Timestamp("2026-10-23")
    assert pd.Timestamp(w["last_batch_matures"]) > d63
    assert w["window_start"] == w["last_batch_matures"]
    assert pd.Timestamp(w["window_end"]) > pd.Timestamp(w["window_start"])
    # 2026-10-23 + 62 trading days (weekdays minus federal holidays)
    # lands on Tuesday 2027-01-26; maturity 5 trading days later is
    # Tuesday 2027-02-02.
    assert w["issuance_day_63"] == "2027-01-26"
    assert w["last_batch_matures"] == "2027-02-02"


# ---------------------------------------------------------------------------
# Failure isolation: run_ml_daily step 5c must never break the v1 pass
# ---------------------------------------------------------------------------
def test_run_live_t4a_failure_isolated(monkeypatch, tmp_path, capsys):
    import run_ml_daily
    from engine import ml_harness as H
    from engine import ml_panel, ml_shadow

    # Stage A reporting isolation: run_live step 5d
    # (_run_reporting_step) refreshes the v1 scorecard header via
    # ml_reporting.refresh_scorecard_reporting(ml_scorecard.
    # SCORECARD_PATH), resolving the path from the module attribute
    # at call time. Redirect it to a temporary copy of the real
    # scorecard — the convention already used by
    # test_ml_reporting.test_run_live_reporting_failure_isolated —
    # so every reporting write in this pass lands in tmp_path and
    # the real v1 scorecard is never written. The mocked
    # score_pending / append_entries / running_record below never
    # execute their real bodies, so their default-path arguments
    # are not exercised.
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

    def _shadow_ok(panel, oas):
        return {"scored_count": 0, "issued_count": 0, "issued_ids": [],
                "asof": "2026-10-23"}

    monkeypatch.setattr(ml_shadow, "run_shadow", _shadow_ok)

    def _t4a_boom():
        raise RuntimeError("t4a boom (test)")

    monkeypatch.setattr(D, "run_t4a", _t4a_boom)
    monkeypatch.setattr(D, "record_t4a_error", lambda msg: None)

    from engine import ml_c3_shadow, ml_t2_shadow

    monkeypatch.setattr(ml_t2_shadow, "run_t2", lambda: {
        "scored_count": 0, "issued_count": 0, "issued_ids": [], "auctions": []})
    monkeypatch.setattr(ml_t2_shadow, "record_t2_error", lambda msg: None)
    monkeypatch.setattr(ml_c3_shadow, "run_c3", lambda: {
        "scored_count": 0, "issued_count": 0, "issued_ids": [],
        "asof": "2026-10-23"})
    monkeypatch.setattr(ml_c3_shadow, "record_c3_error", lambda msg: None)

    rc = run_ml_daily.run_live()
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["shadow"]["status"] == "ok"
    assert summary["t4a"]["status"] == "failed"
    assert "t4a boom" in summary["t4a"]["error"]
    # The Stage A reporting step ran against the temporary copy,
    # not the real v1 scorecard.
    assert summary["reporting"]["status"] == "ok"
    assert summary["reporting"]["path"] == str(path)
    # Silence the imported-but-unused checker note: H is used for the
    # panel load inside run_live (real, read-only).
    assert H is not None
