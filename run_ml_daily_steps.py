"""Fallback driver: execute the remaining failure-isolated steps of
run_ml_daily.run_live (shadow, T4a, T2, C3, v1 reporting) after the
full runner was killed twice by VM replacements (2026-11-01) with the
v1 score/issue step already complete and verified in the scorecard.
Calls the identical step functions in the identical order."""
import json, sys
sys.path.insert(0, "/home/hatch/workspace/quant-engine")
from engine import ml_harness as H
from engine import ml_scorecard, ml_reporting

out = {}
panel = H.load_panel()
oas = H.load_oas()
out["panel_through"] = str(panel.index.max().date())

try:
    from engine import ml_shadow
    r = ml_shadow.run_shadow(panel, oas)
    out["shadow"] = {"status": "ok", "scored_count": r["scored_count"], "issued_count": r["issued_count"], "issued_ids": r["issued_ids"], "asof": r["asof"]}
except Exception as exc:
    out["shadow"] = {"status": "failed", "error": str(exc)[-500:]}

try:
    from engine import ml_t4a_daily
    r = ml_t4a_daily.run_t4a()
    out["t4a"] = {"status": "ok", "scored_count": r["scored_count"], "issued_count": r["issued_count"], "issued_ids": r["issued_ids"], "asof": r["asof"]}
except Exception as exc:
    out["t4a"] = {"status": "failed", "error": str(exc)[-500:]}

try:
    from engine import ml_t2_shadow
    r = ml_t2_shadow.run_t2()
    out["t2"] = {"status": "ok", "scored_count": r["scored_count"], "issued_count": r["issued_count"], "issued_ids": r["issued_ids"], "auctions": r["auctions"]}
except Exception as exc:
    out["t2"] = {"status": "failed", "error": str(exc)[-500:]}

try:
    from engine import ml_c3_shadow
    r = ml_c3_shadow.run_c3()
    out["c3"] = {"status": "ok", "scored_count": r["scored_count"], "issued_count": r["issued_count"], "issued_ids": r["issued_ids"], "asof": r["asof"]}
except Exception as exc:
    out["c3"] = {"status": "failed", "error": str(exc)[-500:]}

try:
    out["reporting"] = ml_reporting.refresh_scorecard_reporting(ml_scorecard.SCORECARD_PATH)
except Exception as exc:
    out["reporting"] = {"status": "failed", "error": str(exc)[-500:]}

out["v1_running_record"] = ml_scorecard.running_record()
print(json.dumps(out, indent=2, default=float))
