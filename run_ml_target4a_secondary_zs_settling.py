"""Settling stage for the ML Target 4a secondary-panel soybean (ZS=F)
ridge claim (Stage C-2 settling).

Pre-registered settling specification:
reports/ml_target4a_secondary_zs_settling_spec_2026-10-27.md (frozen
2026-10-27 before any settling computation), applying protocol
amendment v2026-10-27-A section A. Legs: (i) declustered alert test —
not applicable (classification construction, regression claim);
(ii) alert block bootstrap — not applicable (same reason);
(iii) paired score bootstrap on stored predictions (no fitting);
(iv) provenance ablation — provenance-exposed set empty per the
specification's document-only determination (no fitting);
(v) one-day outcome-lag timing re-fit — the only fitted leg
(2 configurations, trials ledger rows 7818-7819, booked before this
run's results were reported).

Offline evaluation only. The claim's panel, stored predictions, and
results are never overwritten; the flagship's forward issuance,
scorecard, and grading window are untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine import ml_fixpack_stats as F
from engine import ml_harness as H
from engine import ml_t4a_secondary as S

ROOT = Path(__file__).resolve().parent
PANEL = "zs"
LAG_FEATURES = ["cb_hdd", "cb_cdd", "cb_hdd_7d", "cb_cdd_7d",
                "cb_hdd_30d", "cb_cdd_30d", "cb_dd_anom7",
                "cb_dsci", "cb_dsci_chg4w", "cb_d2p", "us_d2p"]
BLOCK = 10
N_RESAMPLES = 5000
SEED = 13

# Frozen machinery-check values from the results record (percent units).
EXPECTED_POOLED_DIFF = 2.5852418290368937 - 2.598764644529965   # -0.0135228...
EXPECTED_NONROLL_DIFF = 2.665086362897395 - 2.6784697788013747  # -0.0133834...


def interval_reading(lo: float, hi: float, observed: float) -> str:
    """Section A leg (iii) verdict language for a score in which a
    negative difference is an improvement."""
    if observed >= 0 or lo > 0:
        return "OVERTURNS"
    if hi < 0:
        return "SUSTAINS"
    return "WEAKENS"


def paired_bootstrap(d_pct: np.ndarray) -> dict:
    """Circular block bootstrap of the mean paired difference, frozen
    construction: block 10, 5,000 resamples, default_rng(13)."""
    n = len(d_pct)
    rng = np.random.default_rng(SEED)
    means = np.empty(N_RESAMPLES)
    for r in range(N_RESAMPLES):
        idx = F.circular_block_indices(n, BLOCK, rng)
        means[r] = float(d_pct[idx].mean())
    lo, hi = F.percentile_interval(means)
    observed = float(d_pct.mean())
    return {
        "n": int(n),
        "observed_mean_diff_pct": observed,
        "bootstrap_median_pct": float(np.median(means)),
        "ci95_lo_pct": lo,
        "ci95_hi_pct": hi,
        "share_resamples_ge_0": float(np.mean(means >= 0)),
        "deciles_pct": [float(x) for x in np.percentile(
            means, [10, 25, 50, 75, 90])],
        "reading": interval_reading(lo, hi, observed),
    }


def leg_iii() -> dict:
    preds = pd.read_parquet(
        ROOT / "data/t4a_weather_futures/wf_predictions_t4a_secondary.parquet")
    zs = preds[preds.panel == PANEL]
    ridge = (zs[zs.model == "ridge"].sort_values("date")
             .reset_index(drop=True))
    clim = (zs[zs.model == "climatology"].sort_values("date")
            .reset_index(drop=True))
    assert len(ridge) == 6060 and len(clim) == 6060
    assert (ridge.date.values == clim.date.values).all()
    assert np.array_equal(ridge.y_true.values, clim.y_true.values)
    assert (ridge.roll_window.values == clim.roll_window.values).all()
    d = (np.abs(ridge.y_pred.values - ridge.y_true.values)
         - np.abs(clim.y_pred.values - clim.y_true.values)) * 100.0
    pooled = paired_bootstrap(d)
    # Machinery check (spec section 5): observed means must reproduce
    # the results record before any reading is taken.
    assert abs(pooled["observed_mean_diff_pct"]
               - EXPECTED_POOLED_DIFF) < 1e-9, pooled["observed_mean_diff_pct"]
    nonroll_mask = ~ridge.roll_window.values
    nonroll = paired_bootstrap(d[nonroll_mask])
    assert abs(nonroll["observed_mean_diff_pct"]
               - EXPECTED_NONROLL_DIFF) < 1e-9, nonroll["observed_mean_diff_pct"]
    order = {"SUSTAINS": 0, "WEAKENS": 1, "OVERTURNS": 2}
    verdict = max((pooled["reading"], nonroll["reading"]),
                  key=lambda r: order[r])
    return {"pooled": pooled, "non_roll": nonroll, "verdict": verdict,
            "machinery_check": {
                "pooled_expected_pct": EXPECTED_POOLED_DIFF,
                "pooled_reproduced_pct": pooled["observed_mean_diff_pct"],
                "non_roll_expected_pct": EXPECTED_NONROLL_DIFF,
                "non_roll_reproduced_pct": nonroll["observed_mean_diff_pct"],
                "passed": True}}


def leg_v() -> tuple[dict, pd.DataFrame]:
    df = S.load_panel(PANEL)
    sample = S.build_sample(PANEL, df)
    lagged = sample.copy()
    full_pos = sample["full_pos"].to_numpy()
    assert (full_pos >= 1).all()
    panel_by_pos = df.reset_index(drop=True)
    for col in LAG_FEATURES:
        vals = panel_by_pos[col].to_numpy(dtype=float)
        lagged[col] = vals[full_pos - 1]
    n_nan = int(lagged[S.FEATURES[PANEL] + [S.TARGET]].isna()
                .sum().sum())
    assert n_nan == 0, f"lag shift introduced {n_nan} NaN cells"
    folds = H.make_folds(lagged, horizon=S.HORIZON)
    orig_folds = H.make_folds(sample, horizon=S.HORIZON)
    assert len(folds) == len(orig_folds) == 97
    for a, b in zip(folds, orig_folds):
        assert np.array_equal(a.train_idx, b.train_idx)
        assert np.array_equal(a.test_idx, b.test_idx)
    preds, fold_meta, alphas = S.run_walk_forward(
        lagged, S.FEATURES[PANEL], folds)
    metrics = S.model_metrics(preds, PANEL)
    verdicts = S.graduation_verdicts(metrics, PANEL)
    kept = preds[preds.model.isin(["ridge", "gbm"])].copy()
    kept.insert(1, "panel", PANEL)
    kept.insert(2, "settling_leg", "v_one_day_lag")
    out = {"metrics_pct": metrics, "verdicts": verdicts,
           "ridge_alphas_by_fold": alphas,
           "lag_features": LAG_FEATURES,
           "n_nan_after_lag": n_nan,
           "n_folds": len(folds),
           "total_test_rows": int(sum(len(f.test_idx) for f in folds))}
    return out, kept


def main() -> None:
    results = {
        "stage": "Settling stage for the Stage C-2 soybean (ZS=F) ridge "
                 "claim (protocol amendment v2026-10-27-A section A)",
        "spec": "reports/ml_target4a_secondary_zs_settling_spec_2026-10-27.md "
                "(frozen 2026-10-27, before any settling computation)",
        "claim": {
            "target": "T4a-ZS-5d",
            "champion": "ridge, frozen 16-feature soybean set",
            "pooled_mae_pct": 2.5852418290368937,
            "climatology_pooled_mae_pct": 2.598764644529965,
            "pooled_ratio": 0.9947964447178643,
            "non_roll_mae_pct": 2.665086362897395,
            "climatology_non_roll_mae_pct": 2.6784697788013747,
            "alpha_1000_folds": "97 of 97",
            "vs_zero_change_pooled_pct": 0.015405027482700007,
            "vs_zero_change_non_roll_pct": 0.013546707584841222,
            "ledger_row": 7816,
        },
        "legs": {},
    }
    results["legs"]["i_declustering"] = {
        "applicable": False,
        "verdict": "NOT APPLICABLE",
        "basis": "Classification construction (alert days, hit count, "
                 "binomial p against a null rate). The claim has no alert "
                 "selection, alert budget, binary label, hit count, or "
                 "null rate; its frozen bar is an MAE comparison on every "
                 "test day. Recorded as not applicable per settling "
                 "specification section 3.",
    }
    results["legs"]["ii_alert_block_bootstrap"] = {
        "applicable": False,
        "verdict": "NOT APPLICABLE",
        "basis": "Classification construction (bootstrap of an alert hit "
                 "rate under a frozen alert budget, +1 convention). The "
                 "claim produces point forecasts, not probabilities, and "
                 "has no alert budget or hit rate. Its paired-score "
                 "content is leg (iii) under section A. Recorded as not "
                 "applicable per settling specification section 4.",
    }
    l3 = leg_iii()
    results["legs"]["iii_paired_score_bootstrap"] = {
        "applicable": True,
        "construction": "Per-observation absolute-error difference "
                        "d = |ridge - y| - |climatology - y| on stored "
                        "pooled out-of-sample predictions in date order; "
                        "circular block bootstrap, block length 10, "
                        "5,000 resamples, numpy.random.default_rng(13); "
                        "run on pooled rows (headline) and non-roll rows; "
                        "leg verdict is the weaker of the two readings "
                        "(frozen in the specification).",
        **l3,
    }
    print(f"[leg iii] pooled reading {l3['pooled']['reading']}, "
          f"non-roll reading {l3['non_roll']['reading']}, "
          f"verdict {l3['verdict']}")
    results["legs"]["iv_provenance_ablation"] = {
        "applicable": True,
        "provenance_exposed_features": [],
        "n_design_derived": 16,
        "n_outcome_screened": 0,
        "fitted_configurations": 0,
        "verdict": "SUSTAINS",
        "basis": "Document-only provenance determination (settling "
                 "specification section 6, Table 1): no feature of the "
                 "frozen 16-feature soybean set appears under its own "
                 "column name in the tested set of any named prior "
                 "screening study; the own-price block is design-frozen "
                 "and was baseline in every flagship configuration "
                 "(section A baseline exclusion). The ablation bundle is "
                 "empty, so the ablated champion is the claimed champion "
                 "itself, which passes the full original bar. No distinct "
                 "configuration exists to fit.",
    }
    print("[leg iv] provenance-exposed set empty; verdict SUSTAINS; "
          "0 configurations fitted")
    l5, lag_preds = leg_v()
    rm = l5["metrics_pct"]["ridge"]
    gm = l5["metrics_pct"]["gbm"]
    rv = l5["verdicts"]["ridge"]
    leg_a = bool(rv["leg_a_pooled_mae_beats_climatology"])
    leg_b = bool(rv["leg_b_non_roll_mae_beats_climatology"])
    if rv["verdict"].startswith("GRADUATES"):
        v5 = "SUSTAINS"
    elif leg_a or leg_b:
        v5 = "WEAKENS"
    else:
        v5 = "OVERTURNS"
    gbm_passes = bool(l5["verdicts"]["gbm"]["verdict"]
                      .startswith("GRADUATES"))
    cap_applied = False
    if gbm_passes and v5 == "OVERTURNS":
        v5 = "WEAKENS"
        cap_applied = True
    from collections import Counter
    results["legs"]["v_one_day_lag"] = {
        "applicable": True,
        "construction": "The 11 externally published features lagged one "
                        "additional trading day (shift by one row in the "
                        "full panel, restriction to the frozen sample); "
                        "labels, climatology, folds, other features "
                        "unchanged; no missing values introduced "
                        "(asserted). Ridge (headline) and GBM (supporting) "
                        "re-fitted under identical machinery. Ledger rows "
                        "7818 (ridge) and 7819 (GBM), booked before "
                        "results were reported.",
        "lagged_ridge": {
            "pooled_mae_pct": rm["mae_pct"],
            "ratio_vs_climatology": rm["mae_ratio_vs_climatology"],
            "non_roll_mae_pct": rm["slices"]["non_roll"]["mae_pct"],
            "mae_vs_zero_change_pooled_pct":
                rm["mae_vs_zero_change_pooled"],
            "mae_vs_zero_change_non_roll_pct":
                rm["slices"]["non_roll"]["mae_vs_zero_change"],
            "folds_won_vs_climatology": rm["folds_won_vs_climatology"],
            "alpha_counts": dict(Counter(
                str(v) for v in l5["ridge_alphas_by_fold"].values())),
            "verdict_under_frozen_bar": rv,
        },
        "lagged_gbm_supporting": {
            "pooled_mae_pct": gm["mae_pct"],
            "ratio_vs_climatology": gm["mae_ratio_vs_climatology"],
            "non_roll_mae_pct": gm["slices"]["non_roll"]["mae_pct"],
            "verdict_under_frozen_bar": l5["verdicts"]["gbm"],
        },
        "supporting_model_cap_applied": cap_applied,
        "verdict": v5,
    }
    print(f"[leg v] lagged ridge pooled MAE {rm['mae_pct']:.4f}% vs "
          f"climatology {l5['metrics_pct']['climatology']['mae_pct']:.4f}%, "
          f"bar verdict {rv['verdict']}; leg verdict {v5}")
    lag_preds.to_parquet(
        ROOT / "data/t4a_weather_futures"
               "/wf_predictions_t4a_secondary_zs_settling.parquet",
        index=False)
    verdicts = [results["legs"]["iii_paired_score_bootstrap"]["verdict"],
                results["legs"]["iv_provenance_ablation"]["verdict"],
                results["legs"]["v_one_day_lag"]["verdict"]]
    if "OVERTURNS" in verdicts:
        overall = "DOES NOT STAND"
    elif "WEAKENS" in verdicts:
        overall = "WEAKENED"
    else:
        overall = "STANDS"
    results["overall_disposition"] = overall
    results["applicable_leg_verdicts"] = {
        "iii_paired_score_bootstrap":
            results["legs"]["iii_paired_score_bootstrap"]["verdict"],
        "iv_provenance_ablation":
            results["legs"]["iv_provenance_ablation"]["verdict"],
        "v_one_day_lag": results["legs"]["v_one_day_lag"]["verdict"],
    }
    (ROOT / "reports/ml_target4a_secondary_zs_settling_results.json"
     ).write_text(json.dumps(results, indent=2, default=str))
    print(f"overall disposition: {overall}")
    print("wrote reports/ml_target4a_secondary_zs_settling_results.json")


if __name__ == "__main__":
    main()
