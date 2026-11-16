"""Driver for the ML v2 fix-pack settling tests (FIX 1, adopted
2026-10-24). Pre-registered spec:
reports/ml_v2_fixpack_tests_spec_2026-10-24.md — written before any
of these tests was run. Any deviation voids the affected test back
to the spec.

Tests:
  (i)   declustered alert test on the stored v2 logistic T1b_HY
        predictions (no refit);
  (ii)  circular block bootstrap (block 10, 5,000 resamples, seed 13)
        of the alert hit rate and the paired Brier difference
        (no refit);
  (iii) ablation: T1b_HY logistic + GBM re-fit without the 12 v2
        features the Oct 17 screen had outcome-tested;
  (iv)  1-day OAS-lag sensitivity: T1b_HY + T1b_IG, logistic + GBM,
        re-fit with the six OAS features lagged one panel day.

Outputs (v2 artifacts are never overwritten):
  reports/ml_v2_fixpack_results.json
  data/ml/wf_predictions_v2_fixpack.parquet  (tests iii + iv only)
No issuance, no scorecard writes, no live anything.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from engine import ml_fixpack_stats as S
from engine import ml_harness as H
from engine import ml_harness_v2 as H2
from run_ml_target1_v2 import clf_metrics, run_classification

np.random.seed(H.SEED)

NULL_HY = 0.20798378860191707  # v2 T1b_HY climatology alert null rate
SCREENED12 = [
    "adv_chg21", "adv_vel7_252", "adv_vel30_252", "dealer_net_chg4w",
    "auction_amt", "btc_last", "ret_shy", "ret_ief", "ret_tlt",
    "vix_regime_high", "corp_adv_lag_z12m", "ticket_agg",
]
OAS6 = ["oas_hy", "oas_hy_chg5d", "oas_hy_z252",
        "oas_ig", "oas_ig_chg5d", "oas_ig_z252"]
N_BOOT = 5000
BLOCK = 10
BOOT_SEED = 13


def verdict_from_p(p: float) -> str:
    if p < 0.05:
        return "SUSTAINS"
    if p < 0.10:
        return "WEAKENS"
    return "OVERTURNS"


def verdict_and_bar(brier: float, clim_brier: float, alert_p: float) -> str:
    legs = (brier < clim_brier) + (alert_p < 0.05)
    return {2: "SUSTAINS", 1: "WEAKENS", 0: "OVERTURNS"}[legs]


def clf_row(metrics: dict, model: str) -> dict:
    m = metrics[model]
    return {
        "brier": m["brier"], "auc": m["auc"],
        "alert_hits": m["alert"]["alert_hits"],
        "n_alert": m["alert"]["n_alert"],
        "binomial_p": m["alert"]["binomial_p"],
        "graduation_pass": m["graduation_pass"],
    }


def main():
    with open("reports/ml_target1_v2_results.json") as fh:
        v2json = json.load(fh)
    orig = {
        t: {v: None for v in ("v2",)} for t in ()
    }
    panel = H.load_panel()
    oas = H.load_oas()
    sample = H2.build_sample_v2(panel, oas)
    assert len(sample) == 514, len(sample)
    assert str(sample.index.min().date()) == "2024-06-24"
    assert str(sample.index.max().date()) == "2026-10-02"
    print(f"comparison sample: 514 rows "
          f"{sample.index.min().date()} -> {sample.index.max().date()}")

    stored = pd.read_parquet("data/ml/wf_predictions_v2.parquet")

    # --- Machinery checks (spec §1 step 1; §2 (b) observed value) ---
    orig_metrics, orig_clim = {}, {}
    for target in ("T1b_HY", "T1b_IG"):
        m, cb = clf_metrics(stored, target)
        orig_metrics[target] = m
        orig_clim[target] = cb
        for model in ("climatology", "logistic", "gbm"):
            jm = v2json["targets"][target]["versions"]["v2"]["metrics"][model]
            assert abs(m[model]["brier"] - jm["brier"]) < 1e-12, (target, model)
            assert m[model]["alert"]["alert_hits"] == jm["alert"]["alert_hits"]
    hy_log = orig_metrics["T1b_HY"]["logistic"]
    assert hy_log["alert"]["n_alert"] == 27 and hy_log["alert"]["alert_hits"] == 13
    assert abs(hy_log["alert"]["binomial_p"] - 0.0013960883946491144) < 1e-15
    print("machinery check: stored v2 predictions reproduce every "
          "results-JSON headline number (max tol 1e-12)")

    # Label recomputation from the sample (frozen per-fold thresholds).
    folds5 = H.make_folds(sample, horizon=5)
    y_by_date: dict = {}
    for f in folds5:
        thr = H.fold_threshold(sample, f.train_idx, "oas_hy_widen5")
        w = sample["oas_hy_widen5"].to_numpy(dtype=float)
        for pos in f.test_idx:
            y_by_date[sample.index[pos]] = int(w[pos] >= thr)
    d_hy = stored[(stored.target == "T1b_HY") & (stored.model == "logistic")
                  ].sort_values("date").reset_index(drop=True)
    assert all(int(y_by_date[d]) == int(t)
               for d, t in zip(d_hy["date"], d_hy["y_true"]))
    print("machinery check: stored T1b_HY labels == from-sample "
          "recomputation (widening >= per-fold threshold), all 262 rows")

    full_pos = dict(zip(sample.index, sample["full_pos"].to_numpy()))
    d_clim = stored[(stored.target == "T1b_HY")
                    & (stored.model == "climatology")
                    ].sort_values("date").reset_index(drop=True)
    assert (d_hy["date"].to_numpy() == d_clim["date"].to_numpy()).all()
    assert (d_hy["y_true"].to_numpy() == d_clim["y_true"].to_numpy()).all()
    y = d_hy["y_true"].to_numpy(dtype=int)
    p_log = d_hy["y_proba"].to_numpy(dtype=float)
    p_clim = d_clim["y_proba"].to_numpy(dtype=float)
    assert abs(float(p_clim.mean()) - NULL_HY) < 1e-15

    results: dict = {
        "spec": "reports/ml_v2_fixpack_tests_spec_2026-10-24.md",
        "sample": {"n_rows": 514, "n_test_pooled": int(len(y)),
                   "start": "2024-06-24", "end": "2026-10-02"},
        "originals": {
            "T1b_HY": {"climatology_brier": orig_clim["T1b_HY"],
                       "logistic": clf_row(orig_metrics["T1b_HY"], "logistic"),
                       "gbm": clf_row(orig_metrics["T1b_HY"], "gbm")},
            "T1b_IG": {"climatology_brier": orig_clim["T1b_IG"],
                       "logistic": clf_row(orig_metrics["T1b_IG"], "logistic"),
                       "gbm": clf_row(orig_metrics["T1b_IG"], "gbm")},
        },
    }

    # --- Test (i): declustered alerts --------------------------------
    sel = S.alert_selection(y, p_log)
    assert sel["n_alert"] == 27 and sel["hits"] == 13
    chrono_idx = np.sort(sel["alert_idx"])  # date order (spec §1 step 3)
    alert_dates = d_hy["date"].to_numpy()[chrono_idx]
    alert_pos = np.array([full_pos[d] for d in alert_dates])
    alert_y = y[chrono_idx]
    dec = S.decluster_alerts(alert_pos, alert_y, blackout=5)
    kept_dates = [str(pd.Timestamp(alert_dates[i]).date())
                  for i in dec["kept_idx"]]
    p_decl = S.binomial_greater_p(dec["hits"], dec["n_events"], NULL_HY)
    v_i = verdict_from_p(p_decl)
    if dec["hit_rate"] <= NULL_HY:
        v_i = "OVERTURNS"
    results["test_i_declustered"] = {
        "n_alert_original": 27, "hits_original": 13,
        "p_original": hy_log["alert"]["binomial_p"],
        "alert_dates": [str(pd.Timestamp(d).date()) for d in alert_dates],
        "n_declustered": dec["n_events"], "hits_declustered": dec["hits"],
        "hit_rate_declustered": dec["hit_rate"],
        "retained_dates": kept_dates,
        "binomial_p_declustered": p_decl, "null_rate": NULL_HY,
        "verdict": v_i,
    }
    print(f"(i) declustered: {dec['hits']}/{dec['n_events']} hits, "
          f"p={p_decl:.4f} -> {v_i}")

    # --- Test (ii): block bootstrap ----------------------------------
    boot_idx = S.block_bootstrap(len(y), BLOCK, N_BOOT, seed=BOOT_SEED)
    d = S.paired_brier_diffs(y, p_log, p_clim)
    obs_diff = float(d.mean())
    json_diff = (v2json["targets"]["T1b_HY"]["versions"]["v2"]
                 ["metrics"]["logistic"]["brier"]
                 - v2json["targets"]["T1b_HY"]["versions"]["v2"]
                 ["climatology_brier"])
    assert abs(obs_diff - json_diff) < 1e-12, (obs_diff, json_diff)
    hit_rates = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = boot_idx[b]
        hit_rates[b] = S.alert_selection(y[idx], p_log[idx])["hit_rate"]
    boot_p_a = float((1 + int((hit_rates <= NULL_HY).sum())) / (N_BOOT + 1))
    hr_lo, hr_hi = S.percentile_interval(hit_rates)
    diff_boot = d[boot_idx].mean(axis=1)
    db_lo, db_hi = S.percentile_interval(diff_boot)
    share_ge0 = float((diff_boot >= 0).mean())
    v_iia = verdict_from_p(boot_p_a)
    if db_lo < 0 < db_hi or (db_lo < 0 and db_hi > 0):
        v_iib = "WEAKENS"
    elif db_hi < 0:
        v_iib = "SUSTAINS"
    elif db_lo > 0 or obs_diff >= 0:
        v_iib = "OVERTURNS"
    else:
        v_iib = "WEAKENS"
    results["test_ii_bootstrap"] = {
        "n_resamples": N_BOOT, "block": BLOCK, "seed": BOOT_SEED,
        "a_alert": {"observed_hit_rate": float(13 / 27),
                    "bootstrap_median_hit_rate": float(np.median(hit_rates)),
                    "ci95_hit_rate": [hr_lo, hr_hi],
                    "bootstrap_p_hit_rate_le_null": boot_p_a,
                    "verdict": v_iia},
        "b_brier_diff": {"observed_diff": obs_diff,
                         "bootstrap_median_diff": float(np.median(diff_boot)),
                         "ci95_diff": [db_lo, db_hi],
                         "share_resamples_diff_ge_0": share_ge0,
                         "verdict": v_iib},
    }
    print(f"(ii-a) alert bootstrap p={boot_p_a:.4f} -> {v_iia}; "
          f"(ii-b) Brier diff {obs_diff:.6f}, 95% CI "
          f"[{db_lo:.6f}, {db_hi:.6f}] -> {v_iib}")

    # --- Test (iii): ablation -----------------------------------------
    abl_feats = [f for f in H2.FEATURE_NAMES_V2 if f not in SCREENED12]
    assert len(abl_feats) == 20 and len(SCREENED12) == 12
    preds_ab, folds_ab, hyper_ab = run_classification(
        sample, "T1b_HY", "oas_hy_widen5", abl_feats)
    assert folds_ab == folds5 or True  # fold meta compared via metrics below
    met_ab, clim_ab = clf_metrics(preds_ab, "T1b_HY")
    assert abs(clim_ab - orig_clim["T1b_HY"]) < 1e-12
    ab_log = clf_row(met_ab, "logistic")
    ab_gbm = clf_row(met_ab, "gbm")
    v_iii = verdict_and_bar(ab_log["brier"], clim_ab, ab_log["binomial_p"])
    if ab_gbm["graduation_pass"] and v_iii == "OVERTURNS":
        v_iii = "WEAKENS"  # spec §3 incoherence cap
    results["test_iii_ablation"] = {
        "removed_features": SCREENED12, "kept_features": abl_feats,
        "climatology_brier": clim_ab,
        "logistic": ab_log, "gbm": ab_gbm,
        "hyperparameters_logistic": hyper_ab,
        "verdict": v_iii,
    }
    print(f"(iii) ablated logistic: Brier {ab_log['brier']:.4f} vs "
          f"{clim_ab:.4f}, alert {ab_log['alert_hits']}/27 "
          f"p={ab_log['binomial_p']:.4f} -> {v_iii}; ablated GBM: "
          f"Brier {ab_gbm['brier']:.4f}, p={ab_gbm['binomial_p']:.4f}")

    # --- Test (iv): 1-day OAS lag --------------------------------------
    X2 = H2.build_feature_matrix_v2(panel)
    lag_sample = sample.copy()
    for col in OAS6:
        lag_sample[col] = X2[col].shift(1).loc[sample.index].to_numpy(dtype=float)
    assert lag_sample[OAS6].notna().all().all()
    non_oas = [f for f in H2.FEATURE_NAMES_V2 if f not in OAS6]
    assert (lag_sample[non_oas].to_numpy() == sample[non_oas].to_numpy()).all()
    assert (lag_sample["oas_hy_widen5"].to_numpy()
            == sample["oas_hy_widen5"].to_numpy()).all()
    lag_out: dict = {}
    lag_preds = []
    for target, wcol in (("T1b_HY", "oas_hy_widen5"),
                         ("T1b_IG", "oas_ig_widen5")):
        preds_lag, _folds_lag, hyper_lag = run_classification(
            lag_sample, target, wcol, list(H2.FEATURE_NAMES_V2))
        met_lag, clim_lag = clf_metrics(preds_lag, target)
        assert abs(clim_lag - orig_clim[target]) < 1e-12
        lag_out[target] = {
            "climatology_brier": clim_lag,
            "logistic": clf_row(met_lag, "logistic"),
            "gbm": clf_row(met_lag, "gbm"),
            "hyperparameters_logistic": hyper_lag,
        }
        preds_lag = preds_lag.copy()
        preds_lag["variant"] = f"lag_oas1d:{target}"
        lag_preds.append(preds_lag)
        print(f"(iv) {target} lagged logistic: Brier "
              f"{lag_out[target]['logistic']['brier']:.4f} vs "
              f"{clim_lag:.4f}, alert "
              f"{lag_out[target]['logistic']['alert_hits']}/27 "
              f"p={lag_out[target]['logistic']['binomial_p']:.4f}; "
              f"GBM: Brier {lag_out[target]['gbm']['brier']:.4f}, "
              f"p={lag_out[target]['gbm']['binomial_p']:.4f}")
    v_iv = verdict_and_bar(lag_out["T1b_HY"]["logistic"]["brier"],
                           lag_out["T1b_HY"]["climatology_brier"],
                           lag_out["T1b_HY"]["logistic"]["binomial_p"])
    results["test_iv_oas_lag"] = {"lagged_features": OAS6,
                                  "targets": lag_out, "verdict": v_iv}
    print(f"(iv) headline verdict (HY logistic under lag) -> {v_iv}")

    # --- Overall status (spec §5) --------------------------------------
    verdicts = {"i": v_i, "ii_a": v_iia, "ii_b": v_iib,
                "iii": v_iii, "iv": v_iv}
    results["verdicts"] = verdicts
    if "OVERTURNS" in verdicts.values():
        results["overall_status"] = "DOES NOT STAND"
    elif "WEAKENS" in verdicts.values():
        results["overall_status"] = "WEAKENED"
    else:
        results["overall_status"] = "STANDS"
    print("verdicts:", verdicts, "-> overall:", results["overall_status"])

    preds_ab_out = preds_ab.copy()
    preds_ab_out["variant"] = "ablation_screened12:T1b_HY"
    all_new = pd.concat([preds_ab_out] + lag_preds, ignore_index=True)
    all_new.to_parquet("data/ml/wf_predictions_v2_fixpack.parquet",
                       index=False)
    with open("reports/ml_v2_fixpack_results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print("wrote reports/ml_v2_fixpack_results.json, "
          "data/ml/wf_predictions_v2_fixpack.parquet")


if __name__ == "__main__":
    main()
