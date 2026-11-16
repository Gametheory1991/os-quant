"""Runner for ML Stage C-3 (V-a volatility state, V-b move size).

Frozen spec: reports/ml_stage_c3_volatility_move_spec_2026-10-27.md.
Integrity gates run BEFORE any fitting; a gate FAIL aborts with no fit.
Offline evaluation only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine import ml_harness as H
from engine import ml_stage_c3 as C
from engine import ml_t4a as T

ROOT = Path(__file__).resolve().parent


def main() -> None:
    np.random.seed(H.SEED)
    results: dict = {"experiment": "Stage C-3 volatility-state / move-size (V-a, V-b)",
                     "spec": "reports/ml_stage_c3_volatility_move_spec_2026-10-27.md",
                     "seed": H.SEED}

    # ---------------- Leg V-a ----------------
    panel = H.load_panel()
    va_sample = C.build_va_sample(panel)
    va_folds = H.make_folds(va_sample, horizon=C.HORIZON)
    va_gate = C.va_integrity_gate(panel, va_sample, va_folds)
    results["leg_Va"] = {
        "target": "P(VIX at t+5 >= 80th pct of VIX over trailing 252 panel days ending at t)",
        "sample": {"n": int(len(va_sample)),
                   "start": str(va_sample.index[0].date()),
                   "end": str(va_sample.index[-1].date()),
                   "event_rate_full_sample": float(va_sample["y"].mean())},
        "n_folds": len(va_folds), "integrity_gate": va_gate}
    if not va_gate["passed"]:
        results["leg_Va"]["status"] = "ABORTED — integrity gate FAIL, no model fit"
        print("V-a GATE FAIL"); print(json.dumps(va_gate, indent=1, default=str))
    else:
        va_preds, va_fold_meta, va_hyp = C.run_va_walk_forward(va_sample, va_folds)
        va_m = C.va_metrics(va_preds)
        va_m["effective_outcomes"] = va_m["n_test"] / 5.0
        results["leg_Va"].update({"folds": va_fold_meta, "hyperparams": va_hyp,
                                  "metrics": va_m, "status": "RUN"})
        va_preds.assign(date=va_preds["date"].dt.strftime("%Y-%m-%d")).to_parquet(
            ROOT / "data/ml/wf_predictions_stage_c3_va.parquet", index=False)
        print(f"V-a: sample {len(va_sample)} rows, {len(va_folds)} folds, "
              f"{va_m['n_test']} test rows, base rate {va_m['base_rate']:.4f}")
        for m in C.VA_MODELS:
            e = va_m["models"][m]
            print(f"  {m:12s} Brier {e['brier']:.4f}  AUC {e['auc']}")
        for m, v in va_m["verdicts"].items():
            print(f"  VERDICT {m}: {v['verdict']} "
                  f"(alert {v['alert_hits']}/{v['n_alert']} = {v['alert_hit_rate']:.3f}, "
                  f"p={v['alert_binomial_p']:.4f})")

    # ---------------- Leg V-b ----------------
    tpanel = T.load_panel()
    vb_sample = C.build_vb_sample(tpanel)
    vb_folds = H.make_folds(vb_sample, horizon=C.HORIZON)
    vb_gate = C.vb_integrity_gate(tpanel, vb_sample, vb_folds)
    results["leg_Vb"] = {
        "target": "|label_ret5| — absolute 5-trading-day log return of NG=F",
        "sample": {"n": int(len(vb_sample)),
                   "start": str(vb_sample.index[0].date()),
                   "end": str(vb_sample.index[-1].date())},
        "n_folds": len(vb_folds), "integrity_gate": vb_gate}
    if not vb_gate["passed"]:
        results["leg_Vb"]["status"] = "ABORTED — integrity gate FAIL, no model fit"
        print("V-b GATE FAIL"); print(json.dumps(vb_gate, indent=1, default=str))
    else:
        vb_preds, vb_fold_meta, vb_alphas = C.run_vb_walk_forward(vb_sample, vb_folds)
        vb_m = C.vb_metrics(vb_preds)
        vb_m["effective_outcomes"] = int(vb_preds[vb_preds.model == "ridge"].shape[0]) / 5.0
        results["leg_Vb"].update({"folds": vb_fold_meta, "ridge_alphas": vb_alphas,
                                  "metrics": vb_m, "status": "RUN"})
        vb_preds.assign(date=vb_preds["date"].dt.strftime("%Y-%m-%d")).to_parquet(
            ROOT / "data/t4a_weather_futures/wf_predictions_stage_c3_vb.parquet",
            index=False)
        print(f"V-b: sample {len(vb_sample)} rows, {len(vb_folds)} folds")
        for m in C.VB_MODELS:
            e = vb_m["models"][m]
            print(f"  {m:26s} MAE {e['mae_pct']:.4f}%  non-roll {e['slices']['non_roll']['mae_pct']:.4f}%")
        for m, v in vb_m["verdicts"].items():
            print(f"  VERDICT {m}: {v['verdict']}")

    out = ROOT / "reports/ml_stage_c3_results.json"
    out.write_text(json.dumps(results, indent=1, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
