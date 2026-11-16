"""Driver for ML Target 4a Stage B-2 — shrinkage-boundary test (memo M3).

Frozen spec: reports/ml_target4a_b2_boundary_spec_2026-10-27.md — written
2026-10-27 BEFORE any Stage B-2 fitting. Order of operations mirrors the
original experiment: build samples -> assert the frozen 2,952-row count ->
make folds -> run the integrity gate -> ONLY IF the gate passes, fit the
walk-forward under the extended grid -> write artifacts. Any gate FAIL
aborts before any model is fit.

The original Target 4a artifacts are read for comparison only and are
never overwritten:
  data/t4a_weather_futures/wf_predictions_t4a.parquet (read)
  reports/ml_target4a_results.json (read)

Outputs:
  data/t4a_weather_futures/wf_predictions_t4a_b2.parquet
  reports/ml_target4a_b2_results.json
Offline evaluation only. No issuance, no scorecard writes.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from engine import ml_harness as H
from engine import ml_t4a as T
from engine import ml_t4a_b2 as B

np.random.seed(H.SEED)

ROOT = Path(__file__).resolve().parent
PRED_PATH = T.T4A_DIR / "wf_predictions_t4a_b2.parquet"
RESULTS_PATH = ROOT / "reports" / "ml_target4a_b2_results.json"
ORIG_PRED_PATH = T.T4A_DIR / "wf_predictions_t4a.parquet"
ORIG_RESULTS_PATH = ROOT / "reports" / "ml_target4a_results.json"


def mae_pct(df: pd.DataFrame) -> float:
    return float(np.mean(np.abs(df["y_true"].to_numpy()
                                - df["y_pred"].to_numpy())) * 100.0)


def main() -> None:
    panel = T.load_panel()
    orig_json_pre = json.loads(ORIG_RESULTS_PATH.read_text())
    frozen_end = pd.Timestamp(orig_json_pre["sample"]["FULL"]["end"])

    # Panel-extension handling (B-2 spec execution note, decided
    # outcome-blind before any fitting): the panel file was incrementally
    # extended after the original experiment, adding sample rows beyond
    # the frozen sample end. The frozen sample is enforced by its dates.
    raw_samples = {v: T.build_sample(panel, feats) for v, feats in T.VARIANTS.items()}
    panel_extension = {
        "unrestricted_sample_n": int(len(raw_samples["FULL"])),
        "panel_rows_now": int(len(panel)),
        "panel_rows_at_original_run": int(orig_json_pre["panel"]["rows"]),
        "extra_sample_dates_beyond_frozen_end": [
            str(d.date()) for d in raw_samples["FULL"].index
            if d > frozen_end],
        "frozen_sample_end": str(frozen_end.date()),
        "handling": "sample restricted to dates <= frozen end before folding/fitting",
    }
    samples = {v: s[s.index <= frozen_end] for v, s in raw_samples.items()}

    for v, s in samples.items():
        if len(s) != T.EXPECTED_SAMPLE_N:
            raise SystemExit(
                f"STOP: {v} restricted sample has {len(s)} rows, spec froze "
                f"{T.EXPECTED_SAMPLE_N}. No fitting done; report the difference.")
    if not (samples["FULL"].index == samples["REDUCED"].index).all():
        raise SystemExit("STOP: FULL and REDUCED samples differ in dates. No fitting done.")

    folds = H.make_folds(samples["FULL"], horizon=T.HORIZON)
    # Fold identity vs the original run: boundaries, sizes, dates must match.
    for f, o in zip(folds, orig_json_pre["folds"]["folds"]):
        for k in ("train_start", "train_end", "test_start", "test_end",
                  "n_train", "n_test"):
            assert f.meta[k] == o[k], (f.fold_id, k, f.meta[k], o[k])
    assert len(folds) == orig_json_pre["folds"]["n_folds"] == 43

    gate = T.run_integrity_gate(panel, samples, folds)
    if not gate["passed"]:
        RESULTS_PATH.write_text(json.dumps(
            {"experiment": "ML Target 4a Stage B-2 — shrinkage-boundary test",
             "integrity_gate": gate,
             "run_aborted": "integrity gate FAILED — no model was fit"},
            indent=2))
        raise SystemExit(f"STOP: integrity gate FAILED — no model was fit. "
                         f"Gate written to {RESULTS_PATH}")

    sample = samples["REDUCED"]
    feats = T.REDUCED_FEATURES
    pred_df, fold_meta, alphas = B.run_walk_forward_ext(sample, feats, folds)
    pred_df = pred_df[["date", "model", "y_true", "y_pred", "fold",
                       "roll_window", "season"]].sort_values(
        ["model", "date"]).reset_index(drop=True)
    pred_df.to_parquet(PRED_PATH, index=False)

    metrics = B.model_metrics_b2(pred_df)
    verdict = B.verdict_b2(metrics)
    n_test = int(len(pred_df[pred_df.model == "climatology"]))
    assert n_test == 2700 and len(folds) == 43, (n_test, len(folds))

    # --- Comparison record: original graduate, from its saved predictions ---
    orig = pd.read_parquet(ORIG_PRED_PATH)
    orig["date"] = pd.to_datetime(orig["date"])
    o_red = orig[(orig.variant == "REDUCED")]
    o_ridge = o_red[o_red.model == "ridge"].sort_values("date").reset_index(drop=True)
    o_clim = o_red[o_red.model == "climatology"].sort_values("date").reset_index(drop=True)
    o_zc = o_red[o_red.model == "zero_change"].sort_values("date").reset_index(drop=True)
    orig_json = json.loads(ORIG_RESULTS_PATH.read_text())
    oj_ridge = orig_json["metrics_pct"]["REDUCED"]["ridge"]
    oj_clim = orig_json["metrics_pct"]["REDUCED"]["climatology"]
    oj_zc = orig_json["metrics_pct"]["REDUCED"]["zero_change"]
    # Traceability: recomputation from saved predictions must match the JSON.
    assert abs(mae_pct(o_ridge) - oj_ridge["mae_pct"]) < 5e-5
    assert abs(mae_pct(o_clim) - oj_clim["mae_pct"]) < 5e-5
    assert abs(mae_pct(o_zc) - oj_zc["mae_pct"]) < 5e-5

    # Traceability: this run's baselines must reproduce the originals exactly.
    base_check = {}
    for m, o in (("climatology", o_clim), ("zero_change", o_zc)):
        mine = pred_df[pred_df.model == m].sort_values("date").reset_index(drop=True)
        assert (mine["date"].to_numpy() == o["date"].to_numpy()).all()
        base_check[m] = float(np.max(np.abs(
            mine["y_pred"].to_numpy() - o["y_pred"].to_numpy())))
    assert all(v == 0.0 for v in base_check.values()), base_check
    # y_true identity across runs.
    ext = pred_df[pred_df.model == "ridge_ext"].sort_values("date").reset_index(drop=True)
    assert (ext["date"].to_numpy() == o_ridge["date"].to_numpy()).all()
    assert float(np.max(np.abs(ext["y_true"].to_numpy()
                               - o_ridge["y_true"].to_numpy()))) == 0.0

    # --- Frozen reporting set computations ---
    clim_pf = metrics["climatology"]["per_fold"]
    ext_pf = metrics["ridge_ext"]["per_fold"]
    folds_won_ext = int(sum(1 for f in clim_pf
                            if ext_pf[f]["mae_pct"] < clim_pf[f]["mae_pct"]))
    o_ridge_pf = oj_ridge["per_fold"]
    folds_won_orig = int(sum(1 for f in clim_pf
                             if o_ridge_pf[f]["mae_pct"] < clim_pf[f]["mae_pct"]))

    def pooled_excl_fold0(df):
        return mae_pct(df[df.fold != 0])

    ex0 = {
        "n_rows": int(len(ext[ext.fold != 0])),
        "climatology_mae_pct": pooled_excl_fold0(pred_df[pred_df.model == "climatology"]),
        "zero_change_mae_pct": pooled_excl_fold0(pred_df[pred_df.model == "zero_change"]),
        "ridge_ext_mae_pct": pooled_excl_fold0(ext),
        "ridge_orig_mae_pct": pooled_excl_fold0(o_ridge),
    }
    ex0["ridge_ext_ratio_vs_climatology"] = ex0["ridge_ext_mae_pct"] / ex0["climatology_mae_pct"]
    ex0["ridge_orig_ratio_vs_climatology"] = ex0["ridge_orig_mae_pct"] / ex0["climatology_mae_pct"]
    ex0["ridge_ext_margin_over_zero_change_pts"] = ex0["zero_change_mae_pct"] - ex0["ridge_ext_mae_pct"]
    ex0["ridge_orig_margin_over_zero_change_pts"] = ex0["zero_change_mae_pct"] - ex0["ridge_orig_mae_pct"]

    fold0 = {
        "n_rows": int(len(ext[ext.fold == 0])),
        "climatology_mae_pct": clim_pf["0"]["mae_pct"],
        "zero_change_mae_pct": metrics["zero_change"]["per_fold"]["0"]["mae_pct"],
        "ridge_orig_mae_pct": o_ridge_pf["0"]["mae_pct"],
        "ridge_ext_mae_pct": ext_pf["0"]["mae_pct"],
        "ridge_ext_alpha": alphas["fold0"],
        "ridge_orig_alpha": float(orig_json["ridge_alphas_by_fold"]["REDUCED"]["fold0"]),
    }

    diag = {
        "ridge_orig_mean_abs_pred_pct": float(np.mean(np.abs(o_ridge["y_pred"].to_numpy())) * 100.0),
        "ridge_orig_rms_pred_pct": float(np.sqrt(np.mean(o_ridge["y_pred"].to_numpy() ** 2)) * 100.0),
        "ridge_ext_mean_abs_pred_pct": float(np.mean(np.abs(ext["y_pred"].to_numpy())) * 100.0),
        "ridge_ext_rms_pred_pct": float(np.sqrt(np.mean(ext["y_pred"].to_numpy() ** 2)) * 100.0),
        "corr_ext_vs_orig_predictions": float(np.corrcoef(
            ext["y_pred"].to_numpy(), o_ridge["y_pred"].to_numpy())[0, 1]),
        "mean_abs_diff_ext_vs_orig_pred_pct": float(np.mean(np.abs(
            ext["y_pred"].to_numpy() - o_ridge["y_pred"].to_numpy())) * 100.0),
        "max_abs_diff_ext_vs_orig_pred_pct": float(np.max(np.abs(
            ext["y_pred"].to_numpy() - o_ridge["y_pred"].to_numpy())) * 100.0),
        "realized_label_mean_abs_pct": float(np.mean(np.abs(ext["y_true"].to_numpy())) * 100.0),
        "realized_label_rms_pct": float(np.sqrt(np.mean(ext["y_true"].to_numpy() ** 2)) * 100.0),
    }

    ext_m = metrics["ridge_ext"]
    zc_m = metrics["zero_change"]
    comparison = {
        "climatology_pooled_mae_pct": metrics["climatology"]["mae_pct"],
        "zero_change_pooled_mae_pct": zc_m["mae_pct"],
        "ridge_orig_pooled_mae_pct": mae_pct(o_ridge),
        "ridge_orig_ratio_vs_climatology": oj_ridge["mae_ratio_vs_climatology"],
        "ridge_ext_pooled_mae_pct": ext_m["mae_pct"],
        "ridge_ext_ratio_vs_climatology": ext_m["mae_ratio_vs_climatology"],
        "ridge_ext_minus_orig_pooled_mae_pts": ext_m["mae_pct"] - mae_pct(o_ridge),
        "ridge_ext_margin_over_zero_change_pts": zc_m["mae_pct"] - ext_m["mae_pct"],
        "ridge_orig_margin_over_zero_change_pts": oj_zc["mae_pct"] - oj_ridge["mae_pct"],
        "non_roll": {
            "climatology_mae_pct": metrics["climatology"]["slices"]["non_roll"]["mae_pct"],
            "zero_change_mae_pct": zc_m["slices"]["non_roll"]["mae_pct"],
            "ridge_orig_mae_pct": oj_ridge["slices"]["non_roll"]["mae_pct"],
            "ridge_ext_mae_pct": ext_m["slices"]["non_roll"]["mae_pct"],
            "ridge_ext_beats_zero_change": bool(
                ext_m["slices"]["non_roll"]["mae_pct"] < zc_m["slices"]["non_roll"]["mae_pct"]),
            "ridge_ext_beats_climatology": bool(
                ext_m["slices"]["non_roll"]["mae_pct"]
                < metrics["climatology"]["slices"]["non_roll"]["mae_pct"]),
        },
        "roll": {
            "climatology_mae_pct": metrics["climatology"]["slices"]["roll"]["mae_pct"],
            "zero_change_mae_pct": zc_m["slices"]["roll"]["mae_pct"],
            "ridge_orig_mae_pct": oj_ridge["slices"]["roll"]["mae_pct"],
            "ridge_ext_mae_pct": ext_m["slices"]["roll"]["mae_pct"],
        },
        "folds_won_vs_climatology": {"ridge_ext": folds_won_ext,
                                     "ridge_orig": folds_won_orig,
                                     "n_folds": len(folds)},
    }

    # Interpretation under the rule frozen in the B-2 spec section 6.
    within_002_of_zc = abs(ext_m["mae_pct"] - zc_m["mae_pct"]) <= 0.02 or \
        ext_m["mae_pct"] >= zc_m["mae_pct"]
    beats_zc_nonroll = comparison["non_roll"]["ridge_ext_beats_zero_change"]
    genuine = (ext_m["mae_pct"] < comparison["ridge_orig_pooled_mae_pct"]
               and ext_m["mae_pct"] < zc_m["mae_pct"] and beats_zc_nonroll)
    shrinkage = within_002_of_zc or (not beats_zc_nonroll)
    if genuine:
        reading = "GENUINE TILT (frozen rule, B-2 spec section 6)"
    elif shrinkage:
        reading = "DISCIPLINED SHRINKAGE TOWARD ZERO-CHANGE (frozen rule, B-2 spec section 6)"
    else:
        reading = "MIXED (frozen rule, B-2 spec section 6)"

    results = {
        "experiment": "ML Target 4a Stage B-2 — shrinkage-boundary test (memo proposal M3): "
                      "ridge x REDUCED with extended alpha grid",
        "spec": "reports/ml_target4a_b2_boundary_spec_2026-10-27.md (frozen 2026-10-27, before fitting)",
        "parent_spec": "reports/ml_target4a_experiment_spec.md (pre-registered 2026-10-24)",
        "target": "label_ret5 — log(close[t+5]/close[t]) of Yahoo NG=F close; "
                  "predictions stored in raw log units; metrics below in % (raw x 100)",
        "extended_grid": B.EXTENDED_ALPHAS,
        "grid_progression": "frozen grid {0.1, 1, 10, 100, 1000} is geometric with ratio 10; "
                            "the single frozen extension is the next step, 10000",
        "panel": {"path": "data/t4a_weather_futures/panel_t4a.parquet",
                  "rows": int(len(panel)),
                  "span": [str(panel['date'].min().date()), str(panel['date'].max().date())]},
        "panel_extension_handling": panel_extension,
        "sample": {"variant": "REDUCED", "n": int(len(sample)),
                   "start": str(sample.index[0].date()), "end": str(sample.index[-1].date()),
                   "n_features": len(feats), "features": feats},
        "folds": {"protocol": "expanding train, INITIAL_TRAIN=252, TEST_BLOCK=63, "
                              "EMBARGO_DAYS=5, horizon=5, purge + embargo per engine/ml_harness.py",
                  "n_folds": len(folds), "total_test_rows": n_test, "folds": fold_meta},
        "integrity_gate": gate,
        "baseline_reproduction_check": {"max_abs_diff_vs_original_saved_predictions": base_check,
                                        "note": "0.0 = this run's climatology and zero-change "
                                                "predictions are identical to the original run's "
                                                "saved predictions on every test row"},
        "ridge_alphas_by_fold_ext": alphas,
        "alpha_selection_counts_ext": {str(k): int(v) for k, v in
                                       sorted(Counter(alphas.values()).items())},
        "ridge_alphas_by_fold_orig_reduced": orig_json["ridge_alphas_by_fold"]["REDUCED"],
        "metrics_pct": metrics,
        "comparison_vs_original_graduate": comparison,
        "fold0": fold0,
        "excluding_fold0": ex0,
        "shrinkage_diagnostics": diag,
        "graduation_extended_configuration": verdict,
        "interpretation_frozen_rule": {
            "reading": reading,
            "ext_within_0.02_pts_of_zero_change_or_worse_pooled": bool(within_002_of_zc),
            "ext_beats_zero_change_non_roll": bool(beats_zc_nonroll),
            "ext_beats_orig_and_zero_change_pooled_and_zero_change_non_roll": bool(genuine),
        },
        "effective_sample": {
            "n_test_rows": n_test,
            "approx_independent_outcomes": n_test // 5,
            "note": "5-day-ahead labels overlap: roughly 1/5 as many independent "
                    f"outcomes as rows ({n_test} rows / 5 = {n_test / 5:.0f})."},
        "trials": {"learned_configurations": 1,
                   "configuration": "ridge x REDUCED, extended grid {0.1, 1, 10, 100, 1000, 10000}",
                   "ledger_row": 7788,
                   "family": "ML model configurations",
                   "baselines_counted": False},
        "scope": "Offline evaluation only. The graduated model's forward issuance, "
                 "scorecard, and frozen grading window (2027-02-02 to 2027-02-09) are "
                 "unchanged. An extended-grid pass is a finding for a future version "
                 "decision by Harry, not an adoption.",
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print(f"gate: {'PASS' if gate['passed'] else 'FAIL'}")
    print(f"sample: REDUCED n={len(sample)}, folds={len(folds)}, test rows={n_test}")
    print(f"alpha counts (extended): {results['alpha_selection_counts_ext']}")
    print(f"alphas by fold: {alphas}")
    for m in B.MODELS_B2:
        e = metrics[m]
        print(f"{m:12s} MAE {e['mae_pct']:.4f}%  RMSE {e['rmse_pct']:.4f}%  "
              f"ratio_vs_clim {e['mae_ratio_vs_climatology']:.4f}  "
              f"non-roll MAE {e['slices']['non_roll']['mae_pct']:.4f}%")
    print(f"orig ridge   MAE {comparison['ridge_orig_pooled_mae_pct']:.4f}%  "
          f"ratio {comparison['ridge_orig_ratio_vs_climatology']:.4f}  "
          f"non-roll {comparison['non_roll']['ridge_orig_mae_pct']:.4f}%")
    print(f"folds won vs climatology: ext {folds_won_ext}/43, orig {folds_won_orig}/43")
    print(f"fold0: {fold0}")
    print(f"excl fold0: {ex0}")
    print(f"diagnostics: {diag}")
    print(f"verdict (extended, frozen bar): {verdict}")
    print(f"interpretation: {reading}")
    print(f"predictions -> {PRED_PATH}")
    print(f"results     -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
