"""Settling stage for the ML Target 2 T2c bills logistic claim.

Pre-registered settling specification:
reports/ml_target2_t2c_bills_settling_spec_2026-10-27.md (frozen
2026-10-27 before any settling computation), applying protocol
amendment v2026-10-27-A section A. Legs:
  (i)   declustered alert test on stored predictions (no fitting);
  (ii)  alert block bootstrap (no fitting);
  (iii) paired score (Brier) bootstrap (no fitting);
  (iv)  provenance ablation — bills logistic re-fit with the
        provenance-exposed bundle removed as a single bundle
        (trials ledger row 7828, booked before results reported);
  (v)   one-day outcome-lag timing re-fit — bills logistic re-fit
        with the demand-history block lagged one additional panel
        step (trials ledger row 7829, booked before results
        reported).
Legs (ii) and (iii) share one bootstrap index stack, exactly as the
fix-pack precedent's single bootstrap carried both statistics.

Offline evaluation only. The claim's sample, stored predictions, and
results are never overwritten; no forward issuance is wired.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from engine import ml_fixpack_stats as F
from engine import ml_models as M
from engine import ml_target2_auction as T2
from engine import ml_target2_backtest as B

ROOT = Path(__file__).resolve().parent
BLOCK = 10
N_RESAMPLES = 5000
SEED = 13
DECLUSTER_SEPARATION = 5

# Frozen claim figures (results record, full precision).
NULL_RATE = 0.20349962494744836
BRIER_MODEL = 0.18011723024950568
BRIER_CLIM = 0.18367432283788113
EXPECTED_DIFF = BRIER_MODEL - BRIER_CLIM  # -0.00355709258837545
ALERT_P_ORIGINAL = 0.00011242567441262555

ABLATION_BUNDLE = ["btc_last", "btc_gap_last",
                   "coupon_day_context", "coupon_amt_day"]
LAG_FEATURES = ["bench_med6", "btc_last", "btc_gap_last", "btc_std6"]
KEPT_FEATURES = [f for f in T2.FEATURE_NAMES if f not in ABLATION_BUNDLE]


def verdict_from_p(p: float) -> str:
    if p < 0.05:
        return "SUSTAINS"
    if p < 0.10:
        return "WEAKENS"
    return "OVERTURNS"


def verdict_from_bar(brier: float, clim_brier: float,
                     alert_p: float) -> str:
    legs = int(brier < clim_brier) + int(alert_p < 0.05)
    return {2: "SUSTAINS", 1: "WEAKENS", 0: "OVERTURNS"}[legs]


def logistic_fit_predict_subset(Xtr, ytr, Xte, numeric_features,
                                categories):
    """The claim's frozen logistic construction with a feature
    subset: StandardScaler over the numeric features inside the
    pipeline, group dummies appended unscaled, C by inner
    TimeSeriesSplit(3) on log-loss (engine/ml_models grid, seed 13
    via the estimator defaults of the frozen ladder)."""
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("grp", OneHotEncoder(categories=[categories],
                              handle_unknown="ignore", sparse_output=False),
         ["group"]),
    ])
    pipe = Pipeline([("pre", pre),
                     ("model", LogisticRegression(max_iter=2000))])
    gs = GridSearchCV(pipe, {"model__C": M.LOGISTIC_CS},
                      cv=TimeSeriesSplit(3), scoring="neg_log_loss")
    gs.fit(Xtr, ytr)
    return gs.predict_proba(Xte)[:, 1], {"C": float(gs.best_params_["model__C"])}


def run_logistic_variant(sample, folds, numeric_features):
    """Walk-forward logistic fit of one variant on the bills sample;
    returns a predictions frame in the claim's schema and the
    classification metrics of engine/ml_target2_backtest."""
    categories = T2.BILL_REGULAR_TERMS
    cols = list(numeric_features) + ["group"]
    X = sample[cols]
    y_weak = sample["weak"].to_numpy(dtype=float)
    rows, hypers = [], {}
    for f in folds:
        Xtr, Xte = X.iloc[f.train_idx], X.iloc[f.test_idx]
        test = sample.iloc[f.test_idx]
        proba, h = logistic_fit_predict_subset(
            Xtr, y_weak[f.train_idx], Xte, numeric_features, categories)
        hypers[f"fold{f.fold_id}"] = h
        clim = T2.climatology_t2c(y_weak[f.train_idx], len(f.test_idx))
        for i, pos in enumerate(f.test_idx):
            r = test.iloc[i]
            rows.append(dict(
                panel="bills", fold=int(f.fold_id), sample_pos=int(pos),
                auction_date=r["auction_date"], cusip=r["cusip"],
                security_type=r["security_type"],
                security_term=r["security_term"], group=r["group"],
                weak=float(y_weak[pos]),
                pred_t2c_climatology=float(clim[i]),
                pred_t2c_logistic_proba=float(proba[i]),
            ))
    preds = pd.DataFrame(rows)
    metrics = B.classification_metrics(preds)
    return preds, metrics, hypers


def main() -> None:
    results = {
        "stage": "Settling stage for the ML Target 2 T2c bills "
                 "logistic claim (protocol amendment v2026-10-27-A "
                 "section A)",
        "spec": "reports/ml_target2_t2c_bills_settling_spec_2026-10-27.md "
                "(frozen 2026-10-27, before any settling computation)",
        "claim": {
            "target": "T2c weak-auction classification, bills panel",
            "champion": "logistic regression, frozen 12 numeric "
                        "features + 7 bill-term group dummies",
            "brier_logistic": BRIER_MODEL,
            "brier_climatology": BRIER_CLIM,
            "auc_logistic": 0.5907257047758334,
            "alert_hits": 27, "n_alert": 66,
            "alert_hit_rate": 0.4090909090909091,
            "null_rate": NULL_RATE,
            "alert_binomial_p": ALERT_P_ORIGINAL,
            "folds_won_brier": "6 of 7",
            "ledger_row": 7827,
        },
        "legs": {},
    }

    # --- Stored predictions; machinery check (spec section 2) -----
    stored = pd.read_parquet(
        ROOT / "data/ml/wf_predictions_target2_2026-10-27.parquet")
    bills = (stored[stored.panel == "bills"]
             .sort_values(["auction_date", "cusip"]).reset_index(drop=True))
    assert len(bills) == 660
    assert bills["sample_pos"].is_monotonic_increasing
    y = bills["weak"].to_numpy(dtype=float)
    p_mod = bills["pred_t2c_logistic_proba"].to_numpy(dtype=float)
    p_clim = bills["pred_t2c_climatology"].to_numpy(dtype=float)
    sel = F.alert_selection(y, p_mod)
    assert sel["n_alert"] == 66 and sel["hits"] == 27, sel
    p_orig = F.binomial_greater_p(sel["hits"], sel["n_alert"], NULL_RATE)
    assert abs(p_orig - ALERT_P_ORIGINAL) < 1e-15, p_orig
    assert abs(float(p_clim.mean()) - NULL_RATE) < 1e-15
    d = F.paired_brier_diffs(y, p_mod, p_clim)
    obs_diff = float(d.mean())
    assert abs(obs_diff - EXPECTED_DIFF) < 1e-12, obs_diff
    results["machinery_check"] = {
        "alert_count_reproduced": 66, "alert_hits_reproduced": 27,
        "alert_binomial_p_reproduced": p_orig,
        "paired_brier_diff_reproduced": obs_diff,
        "passed": True}
    print("machinery check passed: alert 27/66, p reproduced, "
          f"paired Brier diff {obs_diff:.17g}")

    # --- Leg (i): declustered alert test ---------------------------
    chrono_idx = np.sort(sel["alert_idx"])
    alert_pos = bills["sample_pos"].to_numpy()[chrono_idx]
    alert_y = y[chrono_idx]
    alert_dates = bills["auction_date"].to_numpy()[chrono_idx]
    dec = F.decluster_alerts(alert_pos, alert_y,
                             blackout=DECLUSTER_SEPARATION)
    p_decl = F.binomial_greater_p(dec["hits"], dec["n_events"], NULL_RATE)
    v_i = verdict_from_p(p_decl)
    if dec["hit_rate"] <= NULL_RATE:
        v_i = "OVERTURNS"
    kept_dates = [str(pd.Timestamp(alert_dates[i]).date())
                  for i in dec["kept_idx"]]
    results["legs"]["i_declustering"] = {
        "applicable": True,
        "construction": "Greedy declustering in date order over the "
                        "66 alert auctions mapped to bills-sample "
                        "positions; retain an alert iff its sample "
                        "position is at least 5 greater than the last "
                        "retained alert's position (specification "
                        "section 3 event-time reading of section A).",
        "n_alert_original": 66, "hits_original": 27,
        "p_original": p_orig,
        "alert_dates": [str(pd.Timestamp(x).date()) for x in alert_dates],
        "n_declustered": dec["n_events"],
        "hits_declustered": dec["hits"],
        "hit_rate_declustered": dec["hit_rate"],
        "retained_positions": [int(x) for x in dec["kept_positions"]],
        "retained_dates": kept_dates,
        "binomial_p_declustered": p_decl,
        "null_rate": NULL_RATE,
        "verdict": v_i,
    }
    print(f"[leg i] declustered {dec['hits']}/{dec['n_events']} hits, "
          f"hit rate {dec['hit_rate']:.4f}, p={p_decl:.6g} -> {v_i}")

    # --- Legs (ii) + (iii): one bootstrap stack --------------------
    boot_idx = F.block_bootstrap(len(y), BLOCK, N_RESAMPLES, seed=SEED)
    hit_rates = np.empty(N_RESAMPLES)
    for b in range(N_RESAMPLES):
        idx = boot_idx[b]
        hit_rates[b] = F.alert_selection(y[idx], p_mod[idx])["hit_rate"]
    boot_p = float((1 + int((hit_rates <= NULL_RATE).sum()))
                   / (N_RESAMPLES + 1))
    hr_lo, hr_hi = F.percentile_interval(hit_rates)
    v_ii = verdict_from_p(boot_p)
    results["legs"]["ii_alert_block_bootstrap"] = {
        "applicable": True,
        "construction": "Circular block bootstrap over the 660 pooled "
                        "rows in date order, block length 10, 5,000 "
                        "resamples, numpy.random.default_rng(13); one "
                        "index stack shared with leg (iii) (fix-pack "
                        "precedent). Per resample: top 66 by champion "
                        "probability (stable sort), hit rate; p under "
                        "the +1 convention.",
        "observed_hit_rate": float(sel["hit_rate"]),
        "bootstrap_median_hit_rate": float(np.median(hit_rates)),
        "ci95_hit_rate": [hr_lo, hr_hi],
        "bootstrap_p_hit_rate_le_null": boot_p,
        "verdict": v_ii,
    }
    print(f"[leg ii] alert bootstrap p={boot_p:.6g} -> {v_ii}")

    diff_boot = d[boot_idx].mean(axis=1)
    db_lo, db_hi = F.percentile_interval(diff_boot)
    share_ge0 = float((diff_boot >= 0).mean())
    if obs_diff >= 0 or db_lo > 0:
        v_iii = "OVERTURNS"
    elif db_hi < 0:
        v_iii = "SUSTAINS"
    else:
        v_iii = "WEAKENS"
    results["legs"]["iii_paired_brier_bootstrap"] = {
        "applicable": True,
        "construction": "Per-observation paired Brier difference "
                        "d = (p_logistic - y)^2 - (p_climatology - y)^2 "
                        "resampled paired on the leg (ii) index stack.",
        "observed_mean_diff": obs_diff,
        "bootstrap_median_diff": float(np.median(diff_boot)),
        "ci95_diff": [db_lo, db_hi],
        "share_resamples_ge_0": share_ge0,
        "verdict": v_iii,
    }
    print(f"[leg iii] Brier diff {obs_diff:.6f}, 95% CI "
          f"[{db_lo:.6f}, {db_hi:.6f}], share>=0 {share_ge0:.4f} "
          f"-> {v_iii}")

    # --- Sample and folds for the re-fit legs ----------------------
    auctions, sample, folds = B.build_panel_sample("bills")
    new_preds = []

    # --- Leg (iv): provenance ablation -----------------------------
    abl_preds, abl_metrics, abl_hypers = run_logistic_variant(
        sample, folds, KEPT_FEATURES)
    v_iv = verdict_from_bar(abl_metrics["brier_logistic"],
                            abl_metrics["brier_climatology"],
                            abl_metrics["alert"]["binomial_p"])
    abl_preds = abl_preds.copy()
    abl_preds.insert(0, "variant", "ablation_provenance_bundle")
    new_preds.append(abl_preds)
    results["legs"]["iv_provenance_ablation"] = {
        "applicable": True,
        "removed_bundle": ABLATION_BUNDLE,
        "kept_numeric_features": KEPT_FEATURES,
        "fitted_configurations": 1,
        "ledger_row": 7828,
        "metrics": abl_metrics,
        "hyperparams_by_fold": abl_hypers,
        "verdict": v_iv,
    }
    print(f"[leg iv] ablated logistic: Brier "
          f"{abl_metrics['brier_logistic']:.6f} vs climatology "
          f"{abl_metrics['brier_climatology']:.6f}, alert "
          f"{abl_metrics['alert']['alert_hits']}/66 "
          f"p={abl_metrics['alert']['binomial_p']:.6g} -> {v_iv}")

    # --- Leg (v): demand-history lag re-fit -------------------------
    # Construction as amended by settling-specification Amendment A1
    # (2026-10-27, written after the original any-term panel shift
    # stopped on the no-missing-values assertion — 37 missing cells —
    # and before any leg (v) fit): lag WITHIN the auction's own term,
    # i.e. recompute the demand-history block from the trailing
    # same-term window excluding the most recent prior reported
    # auction; five-value window on exactly the seven sample rows
    # with exactly six prior reported ratios (A1 section A1.2).
    panel = T2.bills_panel(auctions)
    x_panel = T2.build_features(panel, auctions, "bills")
    panel_pos_by_term: dict[str, list[int]] = {}
    for pos in range(len(panel)):
        panel_pos_by_term.setdefault(panel["group"].iloc[pos], []).append(pos)
    prev_same: dict[int, int] = {}
    for poss in panel_pos_by_term.values():
        for a_, b_ in zip(poss, poss[1:]):
            prev_same[b_] = a_
    lagged = sample.copy()
    five_window_rows: list[int] = []
    n_shift_checked = 0
    for i, (_, r) in enumerate(sample.iterrows()):
        prior = panel[(panel["group"] == r["group"])
                      & (panel["auction_date"] < r["auction_date"])]
        prior_btc = prior["bid_to_cover_ratio"].dropna()
        assert len(prior_btc) >= 6
        window = prior_btc.iloc[:-1].tail(6)  # positions -7..-2
        lag_bench = float(window.median())
        lag_last = float(prior_btc.iloc[-2])
        lagged.iat[i, lagged.columns.get_loc("bench_med6")] = lag_bench
        lagged.iat[i, lagged.columns.get_loc("btc_last")] = lag_last
        lagged.iat[i, lagged.columns.get_loc("btc_gap_last")] = (
            lag_last - lag_bench)
        lagged.iat[i, lagged.columns.get_loc("btc_std6")] = float(
            window.std(ddof=1))
        if len(window) == 5:
            five_window_rows.append(int(r["sample_pos"]))
        # A1 identity check: wherever the within-term panel shift is
        # defined, the recomputation must equal it exactly.
        s = prev_same.get(int(r["event_pos"]))
        if s is not None and not x_panel[LAG_FEATURES].iloc[s].isna().any():
            shifted = x_panel[LAG_FEATURES].iloc[s]
            assert abs(float(shifted["bench_med6"]) - lag_bench) < 1e-12
            assert abs(float(shifted["btc_last"]) - lag_last) < 1e-12
            assert abs(float(shifted["btc_gap_last"])
                       - (lag_last - lag_bench)) < 1e-12
            assert abs(float(shifted["btc_std6"])
                       - float(window.std(ddof=1))) < 1e-12
            n_shift_checked += 1
    assert five_window_rows == [0, 1, 2, 3, 4, 107, 559], five_window_rows
    print(f"[leg v] A1 identity check: recomputation == within-term "
          f"panel shift on {n_shift_checked} rows; five-value window "
          f"rows {five_window_rows}")
    n_nan = int(lagged[T2.FEATURE_NAMES].isna().sum().sum())
    assert n_nan == 0, f"lag shift introduced {n_nan} NaN cells"
    non_lag = [c for c in T2.FEATURE_NAMES if c not in LAG_FEATURES]
    assert (lagged[non_lag].to_numpy()
            == sample[non_lag].to_numpy()).all()
    assert (lagged["weak"].to_numpy()
            == sample["weak"].to_numpy()).all()
    # Folds recomputed on the lagged frame must equal the claim's.
    refunding = T2.refunding_mondays(auctions)
    folds_check = T2.make_folds(sample, refunding=refunding,
                                **B.PANEL_FOLD_PARAMS["bills"])
    assert len(folds_check) == len(folds)
    for a, b_ in zip(folds, folds_check):
        assert np.array_equal(a.train_idx, b_.train_idx)
        assert np.array_equal(a.test_idx, b_.test_idx)
    lag_preds, lag_metrics, lag_hypers = run_logistic_variant(
        lagged, folds, list(T2.FEATURE_NAMES))
    v_v = verdict_from_bar(lag_metrics["brier_logistic"],
                           lag_metrics["brier_climatology"],
                           lag_metrics["alert"]["binomial_p"])
    lag_preds = lag_preds.copy()
    lag_preds.insert(0, "variant", "lag_demand_history_1panel_step")
    new_preds.append(lag_preds)
    results["legs"]["v_one_day_lag"] = {
        "applicable": True,
        "construction": "Demand-history block lagged within its own "
                        "term under Amendment A1 to the settling "
                        "specification (the original any-term panel "
                        "shift stopped on the no-missing-values "
                        "assertion, 37 missing cells, before any leg "
                        "(v) fit): recomputed from the trailing "
                        "same-term window excluding the most recent "
                        "prior reported auction (positions -7..-2), "
                        "asserted identical to the within-term panel "
                        "shift wherever that shift is defined; "
                        "five-value window on exactly the seven "
                        "sample rows with exactly six prior reported "
                        "ratios (sample positions 0, 1, 2, 3, 4, 107, "
                        "559); labels, climatology, folds (verified "
                        "identical), and all other features unchanged; "
                        "no missing values introduced (asserted).",
        "amendment": "Settling specification Amendment A1 "
                     "(2026-10-27)",
        "lagged_features": LAG_FEATURES,
        "n_nan_after_lag": n_nan,
        "fitted_configurations": 1,
        "ledger_row": 7829,
        "metrics": lag_metrics,
        "hyperparams_by_fold": lag_hypers,
        "verdict": v_v,
    }
    print(f"[leg v] lagged logistic: Brier "
          f"{lag_metrics['brier_logistic']:.6f} vs climatology "
          f"{lag_metrics['brier_climatology']:.6f}, alert "
          f"{lag_metrics['alert']['alert_hits']}/66 "
          f"p={lag_metrics['alert']['binomial_p']:.6g} -> {v_v}")

    all_new = pd.concat(new_preds, ignore_index=True)
    all_new.to_parquet(
        ROOT / "data/ml/wf_predictions_target2_bills_settling_2026-10-27.parquet",
        index=False)

    verdicts = {
        "i_declustering": v_i,
        "ii_alert_block_bootstrap": v_ii,
        "iii_paired_brier_bootstrap": v_iii,
        "iv_provenance_ablation": v_iv,
        "v_one_day_lag": v_v,
    }
    results["leg_verdicts"] = verdicts
    if "OVERTURNS" in verdicts.values():
        overall = "DOES NOT STAND"
    elif "WEAKENS" in verdicts.values():
        overall = "WEAKENED"
    else:
        overall = "STANDS"
    results["overall_disposition"] = overall
    results["trials_booked"] = {
        "rows": [7828, 7829],
        "family": "ML model configurations",
        "family_count_after": 58,
        "program_total_after": 7829,
    }
    (ROOT / "reports/ml_target2_t2c_bills_settling_results_2026-10-27.json"
     ).write_text(json.dumps(results, indent=2, default=str))
    print(f"overall disposition: {overall}")
    print("wrote reports/ml_target2_t2c_bills_settling_results_2026-10-27.json "
          "and data/ml/wf_predictions_target2_bills_settling_2026-10-27.parquet")


if __name__ == "__main__":
    main()
