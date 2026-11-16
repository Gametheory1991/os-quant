"""Scorecard persistence + issuance-gate tests (Phase 4 standing operation).

Covers engine/ml_scorecard.py (parse/format round-trip, duplicate
refusal, scoring fills only matured entries, never early) and the
run_ml_daily.py unlock: --live without ML_DAILY_ISSUANCE=1 refuses
with exit 2 and writes nothing; a bare invocation also refuses.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine import ml_scorecard as SC

ROOT = Path(__file__).resolve().parent.parent


def _entry(pid_date, target, model, prediction, seq, target_descr=None):
    feats = {name: 0.5 for name in
             ["log_illiq_agg", "illiq_agg_z252", "illiq_chg5", "adv_chg5",
              "onrun_share_agg", "ats_share_agg", "share_3_5_agg", "oas_hy",
              "oas_hy_chg5d", "oas_hy_z252", "tnx_chg63", "vix",
              "slope_t10y2y", "auction_day"]}
    feats["illiq_t"] = 1.5e-05
    return {
        "prediction_id": f"{pid_date}-{target}-{seq:02d}",
        "issued_at": "2026-10-13T23:00:00-04:00",
        "target": target_descr or f"{target}: test target",
        "horizon": "1 trading day" if target == "T1a1" else "5 trading days",
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


def _synthetic_panel_oas(n=600):
    idx = pd.bdate_range("2023-01-02", periods=n)
    illiq = 1e-5 * (1.0 + (np.arange(n) % 50) / 100.0)
    panel = pd.DataFrame({"illiq_agg": illiq}, index=idx)
    hy = pd.Series(3.0 + np.arange(n) * 0.001, index=idx)
    ig = pd.Series(1.0 + np.arange(n) * 0.0005, index=idx)
    return panel, {"HY": hy, "IG": ig}, idx


def test_format_parse_roundtrip(tmp_path):
    e = _entry("2024-06-03", "T1a1", "ridge:v1-frozen14", {"point": 2.5e-05, "alpha": 10.0}, 3)
    text = "# header\n\n" + SC.format_entry(e) + "\n"
    parsed = SC.parse_entries(text)
    assert len(parsed) == 1
    p = parsed[0]
    assert p["prediction_id"] == e["prediction_id"]
    assert p["target_key"] == "T1a1"
    assert p["asof"] == "2024-06-03"
    assert p["outcome"] == "pending"
    assert p["fields"]["prediction"] == "point=2.5e-05, alpha=10.0"


def test_append_refuses_duplicates(tmp_path):
    path = tmp_path / "scorecard.md"
    path.write_text("# header\n")
    e = _entry("2024-06-03", "T1a1", "baseline:persistence", 1.5e-05, 1)
    assert SC.append_entries([e], path) == 1
    with pytest.raises(ValueError):
        SC.append_entries([e], path)


def test_scoring_fills_only_matured(tmp_path):
    panel, oas, idx = _synthetic_panel_oas()
    asof = idx[500]
    asof_s = str(asof.date())
    path = tmp_path / "scorecard.md"
    path.write_text("# header\n")
    e1 = _entry(asof_s, "T1a1", "baseline:persistence", 1.5e-05, 1,
                target_descr="T1a1: illiq_agg at t+1")
    e2 = _entry(asof_s, "T1b_HY", "baseline:climatology",
                {"probability": 0.3, "threshold_pp": 0.01}, 2,
                target_descr="T1b_HY: HY OAS widens by >= threshold over t+1..t+5")
    e3 = _entry(str(idx[-1].date()), "T1a1", "baseline:persistence", 1.5e-05, 3,
                target_descr="T1a1: illiq_agg at t+1")
    SC.append_entries([e1, e2, e3], path)

    scored = SC.score_pending(panel, oas, path)
    assert {s["prediction_id"] for s in scored} == {e1["prediction_id"], e2["prediction_id"]}

    parsed = {p["prediction_id"]: p for p in SC.parse_entries(path.read_text())}
    realized_t1a1 = float(panel["illiq_agg"].iloc[501])
    assert parsed[e1["prediction_id"]]["outcome"] == repr(realized_t1a1)
    assert parsed[e1["prediction_id"]]["error"] == \
        f"abs_error={abs(1.5e-05 - realized_t1a1)!r}"
    # HY widening over 5 steps of the synthetic series = 0.005 < 0.01 -> event 0
    assert parsed[e2["prediction_id"]]["outcome"] == "widening_pp=0.0050000000000000044, event=0" \
        or parsed[e2["prediction_id"]]["outcome"].endswith(", event=0")
    assert parsed[e2["prediction_id"]]["error"] == f"brier={(0.3 - 0) ** 2!r}"
    # Last-date entry: label not yet realized -> stays pending, never scored early.
    assert parsed[e3["prediction_id"]]["outcome"] == "pending"

    rec = SC.running_record(path)
    by = {(r["target"], r["model_version"]): r for r in rec}
    assert by[("T1a1", "baseline:persistence")]["n_scored"] == 1
    assert by[("T1b_HY", "baseline:climatology")]["value"] == pytest.approx(0.09)


def _run_daily(args, env_extra=None):
    env = {k: v for k, v in os.environ.items() if k != "ML_DAILY_ISSUANCE"}
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "run_ml_daily.py", *args],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=300,
    )


def test_live_without_unlock_refuses():
    r = _run_daily(["--live"])
    assert r.returncode == 2
    assert "REFUSED" in r.stderr and "ML_DAILY_ISSUANCE" in r.stderr


def test_bare_invocation_refuses():
    r = _run_daily([])
    assert r.returncode == 2
    assert "REFUSED" in r.stderr


def test_issuance_unlocked_env(monkeypatch):
    import run_ml_daily as R

    monkeypatch.delenv("ML_DAILY_ISSUANCE", raising=False)
    assert R.issuance_unlocked() is False
    monkeypatch.setenv("ML_DAILY_ISSUANCE", "1")
    assert R.issuance_unlocked() is True
    monkeypatch.setenv("ML_DAILY_ISSUANCE", "yes")
    assert R.issuance_unlocked() is False
