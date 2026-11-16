"""Driver for ML Target 4a (T4a-NG-5d, weather -> natural gas futures).

Pre-registered spec: reports/ml_target4a_experiment_spec.md — written
2026-10-24 BEFORE any fitting. Order of operations is part of the spec:
build samples -> assert the frozen 2,952-row count -> make folds -> run
the integrity gate -> ONLY IF the gate passes, fit the walk-forward ->
write artifacts. Any gate FAIL aborts before any model is fit.

Outputs:
  data/t4a_weather_futures/wf_predictions_t4a.parquet — per test row x
    variant x model: date, variant, model, y_true, y_pred (raw log
    units), fold, roll_window, season
  reports/ml_target4a_results.json — sample, folds, integrity-gate
    output, pooled + per-fold + slice metrics, graduation verdicts
Offline evaluation only. No issuance, no scorecard writes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine import ml_harness as H
from engine import ml_t4a as T

np.random.seed(H.SEED)

ROOT = Path(__file__).resolve().parent
PRED_PATH = T.T4A_DIR / "wf_predictions_t4a.parquet"
RESULTS_PATH = ROOT / "reports" / "ml_target4a_results.json"


def main() -> None:
    panel = T.load_panel()
    samples = {v: T.build_sample(panel, feats) for v, feats in T.VARIANTS.items()}

    # Frozen count assertion (spec §6 / D1): stop BEFORE fitting if off.
    for v, s in samples.items():
        if len(s) != T.EXPECTED_SAMPLE_N:
            raise SystemExit(
                f"STOP: {v} sample has {len(s)} rows, spec froze "
                f"{T.EXPECTED_SAMPLE_N}. No fitting done; report the difference.")
    if not (samples["FULL"].index == samples["REDUCED"].index).all():
        raise SystemExit("STOP: FULL and REDUCED samples differ in dates. No fitting done.")

    # Identical folds for both variants (D1): built once from the FULL sample
    # (sample positions / full_pos identical for REDUCED by the check above).
    folds = H.make_folds(samples["FULL"], horizon=T.HORIZON)

    gate = T.run_integrity_gate(panel, samples, folds)
    if not gate["passed"]:
        RESULTS_PATH.write_text(json.dumps(
            {"experiment": "ML Target 4a — T4a-NG-5d", "integrity_gate": gate,
             "run_aborted": "integrity gate FAILED — no model was fit"},
            indent=2))
        raise SystemExit(f"STOP: integrity gate FAILED — no model was fit. "
                         f"Gate written to {RESULTS_PATH}")

    all_preds, fold_metas, alphas = [], {}, {}
    for vname, feats in T.VARIANTS.items():
        preds, fold_meta, al = T.run_walk_forward(samples[vname], feats, folds)
        preds["variant"] = vname
        all_preds.append(preds)
        fold_metas[vname] = fold_meta
        alphas[vname] = al
    pred_df = pd.concat(all_preds, ignore_index=True)[
        ["date", "variant", "model", "y_true", "y_pred", "fold",
         "roll_window", "season"]].sort_values(
        ["variant", "model", "date"]).reset_index(drop=True)
    pred_df.to_parquet(PRED_PATH, index=False)

    metrics = {v: T.model_metrics(pred_df[pred_df.variant == v])
               for v in T.VARIANTS}
    verdicts = T.graduation_verdicts(metrics)
    n_test = int(len(pred_df[(pred_df.variant == "FULL")
                             & (pred_df.model == "climatology")]))

    results = {
        "experiment": "ML Target 4a — T4a-NG-5d (weather -> NG=F 5-day log return)",
        "spec": "reports/ml_target4a_experiment_spec.md (pre-registered 2026-10-24, before fitting)",
        "target": "label_ret5 — log(close[t+5]/close[t]) of Yahoo NG=F close; "
                  "predictions stored in raw log units; metrics below in % (raw x 100)",
        "panel": {"path": "data/t4a_weather_futures/panel_t4a.parquet",
                  "rows": int(len(panel)),
                  "span": [str(panel['date'].min().date()), str(panel['date'].max().date())]},
        "sample": {v: {"n": int(len(s)), "start": str(s.index[0].date()),
                       "end": str(s.index[-1].date()),
                       "n_features": len(T.VARIANTS[v])}
                   for v, s in samples.items()} | {
            "identical_dates_across_variants": True,
            "coordinator_decision": "D1 — stor_dev5y core in both variants; identical sample/folds"},
        "folds": {"protocol": "expanding train, INITIAL_TRAIN=252, TEST_BLOCK=63, "
                              "EMBARGO_DAYS=5, horizon=5, purge + embargo per engine/ml_harness.py",
                  "n_folds": len(folds), "total_test_rows": n_test,
                  "folds": fold_metas["FULL"]},
        "integrity_gate": gate,
        "ridge_alphas_by_fold": alphas,
        "metrics_pct": metrics,
        "effective_sample": {
            "n_test_rows": n_test,
            "approx_independent_outcomes": n_test // 5,
            "note": "5-day-ahead labels overlap: consecutive test rows share label days, "
                    "so the test set carries roughly 1/5 as many independent outcomes as rows "
                    f"({n_test} rows / 5 = {n_test / 5:.0f})."},
        "graduation": verdicts,
        "trials": {"learned_configurations": 4,
                   "configurations": ["ridge x FULL", "ridge x REDUCED", "gbm x FULL", "gbm x REDUCED"],
                   "baselines_counted": False},
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print(f"gate: {'PASS' if gate['passed'] else 'FAIL'}")
    print(f"sample: FULL n={len(samples['FULL'])}, REDUCED n={len(samples['REDUCED'])}, "
          f"folds={len(folds)}, test rows={n_test}")
    for v in T.VARIANTS:
        for m in T.MODELS:
            e = metrics[v][m]
            print(f"{v:8s} {m:12s} MAE {e['mae_pct']:.4f}%  RMSE {e['rmse_pct']:.4f}%  "
                  f"ratio_vs_clim {e['mae_ratio_vs_climatology']:.4f}  "
                  f"ratio_vs_pers {e['mae_ratio_vs_persistence']:.4f}")
        for m in T.LEARNED:
            print(f"{v:8s} {m:12s} verdict: {verdicts[v][m]['verdict']}")
    print(f"predictions -> {PRED_PATH}")
    print(f"results     -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
