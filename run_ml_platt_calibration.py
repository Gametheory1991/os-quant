"""Driver for Stage B-1 — Platt calibration layer for the T1b widening
probabilities (memo proposal M1).

Frozen spec: reports/ml_target1_platt_calibration_spec_2026-10-27.md
(frozen 2026-10-27 19:05 ET, before any Platt fit).

Method, per the frozen spec: two-parameter Platt scaling (unpenalized
logistic regression of the realized outcome on logit(p)), fitted ONLY
on earlier folds' out-of-sample probabilities, applied fold by fold to
the stored v1 logistic / GBM probabilities for T1b_HY and T1b_IG.
Fold 0 has no earlier fold: no calibrated probability, excluded from
calibrated pooling (pooled calibrated sample = folds 1-4, 197 days).
Isotonic regression is rejected by the spec and is not run in any form.
No base model is re-fitted; no feature is read; no label is recomputed.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from engine import ml_harness as H

CLIP = 1e-6
TARGETS = ["T1b_HY", "T1b_IG"]
MODELS = ["logistic", "gbm"]
CAL_FOLDS = [1, 2, 3, 4]


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), CLIP, 1.0 - CLIP)
    return np.log(p / (1.0 - p))


def fit_platt(y, p):
    """Unpenalized logistic regression of y on logit(p); returns (a, b)."""
    X = logit(p).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    lr.fit(X, np.asarray(y, dtype=int))
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def apply_platt(p, a, b):
    z = a + b * logit(p)
    return 1.0 / (1.0 + np.exp(-z))


def calib_intercept_slope(y, p):
    """Stage A E3 element: logistic regression of outcome on the model's
    own forecast on the log-odds scale. Intercept 0 / slope 1 = perfect."""
    return fit_platt(y, p)


def main():
    run_ts = datetime.now(timezone.utc).isoformat()
    pred = pd.read_parquet("data/ml/wf_predictions.parquet")
    record = json.load(open("reports/ml_target1_results.json"))

    # --- Machinery check: stored v1 pooled (260-day) Brier values must
    # reproduce the parent results record before anything is fitted on
    # subsets of them.
    machinery = {}
    for t in TARGETS:
        for m in MODELS + ["climatology"]:
            sub = pred[(pred.target == t) & (pred.model == m)].sort_values("date")
            b = H.brier(sub.y_true.to_numpy(), sub.y_proba.to_numpy())
            ref = record["targets"][t]["metrics"][m]["brier"]
            machinery[f"{t}:{m}"] = {"recomputed_brier_260": b, "record_brier_260": ref,
                                     "abs_diff": abs(b - ref)}
            assert abs(b - ref) < 1e-12, (t, m, b, ref)

    out = {
        "experiment": "ML Target 1 v1 - Platt calibration layer (Stage B-1, memo proposal M1)",
        "spec": "reports/ml_target1_platt_calibration_spec_2026-10-27.md",
        "spec_frozen": "2026-10-27 19:05 America/New_York (before any Platt fit)",
        "run_timestamp_utc": run_ts,
        "trials_ledger_rows": [7789, 7790, 7791, 7792],
        "trials_family": "ML model configurations",
        "machinery_check_stored_v1_brier": machinery,
        "method": {
            "form": "p_cal = sigmoid(a + b * logit(p)); (a,b) by unpenalized logistic regression of y on logit(p)",
            "fitting_sample": "earlier folds' out-of-sample predictions only (folds 0..k-1 for fold k)",
            "fold0": "no earlier fold - no calibrated probability; excluded from calibrated pooling",
            "clip": CLIP,
            "isotonic_regression": "REJECTED by the frozen spec; not run in any form",
        },
        "targets": {},
    }
    cal_rows = []

    for t in TARGETS:
        clim_all = pred[(pred.target == t) & (pred.model == "climatology")][["date", "y_proba"]]
        clim_map = dict(zip(clim_all.date, clim_all.y_proba))
        out["targets"][t] = {}
        for m in MODELS:
            df = pred[(pred.target == t) & (pred.model == m)].sort_values(["fold", "date"]).reset_index(drop=True)
            assert len(df) == 260
            clip_inactive = bool(((df.y_proba > CLIP) & (df.y_proba < 1 - CLIP)).all())

            platt_by_fold, per_fold = {}, {}
            cal_parts = []
            for k in CAL_FOLDS:
                tr = df[df.fold < k]
                te = df[df.fold == k].copy()
                a, b = fit_platt(tr.y_true.to_numpy(), tr.y_proba.to_numpy())
                te["y_proba_cal"] = apply_platt(te.y_proba.to_numpy(), a, b)
                te["platt_a"], te["platt_b"] = a, b
                te["platt_fit_n"] = len(tr)
                te["platt_fit_folds"] = ",".join(str(int(x)) for x in sorted(tr.fold.unique()))
                platt_by_fold[str(k)] = {"a_intercept": a, "b_slope": b,
                                         "fit_on_folds": sorted(int(x) for x in tr.fold.unique()),
                                         "fit_n": int(len(tr))}
                clim_te = te.date.map(clim_map).to_numpy()
                per_fold[str(k)] = {
                    "n": int(len(te)),
                    "brier_uncalibrated": H.brier(te.y_true, te.y_proba),
                    "brier_calibrated": H.brier(te.y_true, te.y_proba_cal),
                    "brier_climatology": H.brier(te.y_true, clim_te),
                    "base_rate": float(te.y_true.mean()),
                }
                cal_parts.append(te)
            cal = pd.concat(cal_parts).sort_values("date").reset_index(drop=True)
            assert len(cal) == 197, len(cal)
            y = cal.y_true.to_numpy(dtype=float)
            p_u = cal.y_proba.to_numpy(dtype=float)
            p_c = cal.y_proba_cal.to_numpy(dtype=float)
            p_clim = cal.date.map(clim_map).to_numpy(dtype=float)
            null_rate = float(np.mean(p_clim))

            alert_u = H.alert_budget_test(y, p_u, null_rate)
            alert_c = H.alert_budget_test(y, p_c, null_rate)
            # alert-set identity check (same top-20 dates?)
            def alert_dates(p):
                order = np.argsort(-p, kind="stable")[: int(np.ceil(0.10 * len(y)))]
                return sorted(cal.date.iloc[order].astype(str).tolist())
            au, ac = alert_dates(p_u), alert_dates(p_c)
            # within-fold ranking preservation (Spearman of p vs p_cal per fold)
            wf_spear = {}
            for k in CAL_FOLDS:
                s = cal[cal.fold == k]
                wf_spear[str(k)] = float(stats.spearmanr(s.y_proba, s.y_proba_cal).statistic)

            a_bef, b_bef = calib_intercept_slope(y, p_u)
            a_aft, b_aft = calib_intercept_slope(y, p_c)

            brier_u, brier_c, brier_clim = H.brier(y, p_u), H.brier(y, p_c), H.brier(y, p_clim)
            verdict = bool(alert_c["binomial_p"] < 0.05 and brier_c < brier_clim)
            legs_failed = []
            if not alert_c["binomial_p"] < 0.05:
                legs_failed.append("alert")
            if not brier_c < brier_clim:
                legs_failed.append("Brier")

            out["targets"][t][m] = {
                "n_calibrated_pooled": 197,
                "effective_nonoverlapping": 197 / 5,
                "date_start_calibrated": str(cal.date.min().date()),
                "date_end_calibrated": str(cal.date.max().date()),
                "clip_inactive_on_stored_probabilities": clip_inactive,
                "platt_parameters_by_fold": platt_by_fold,
                "per_fold": per_fold,
                "brier_uncalibrated_identical_days": brier_u,
                "brier_calibrated": brier_c,
                "brier_climatology_identical_days": brier_clim,
                "brier_change_cal_minus_uncal": brier_c - brier_u,
                "alert_uncalibrated_identical_days": alert_u,
                "alert_calibrated": alert_c,
                "alert_top20_date_set_identical": au == ac,
                "alert_dates_only_in_uncalibrated": sorted(set(au) - set(ac)),
                "alert_dates_only_in_calibrated": sorted(set(ac) - set(au)),
                "within_fold_spearman_p_vs_pcal": wf_spear,
                "calibration_intercept_slope_before": {"intercept": a_bef, "slope": b_bef},
                "calibration_intercept_slope_after": {"intercept": a_aft, "slope": b_aft,
                                                      "note": "descriptive pooled regression; calibrators differ by fold"},
                "auc_uncalibrated_identical_days": float(roc_auc_score(y, p_u)),
                "auc_calibrated": float(roc_auc_score(y, p_c)),
                "graduation_bar": "alert binomial p < 0.05 AND Brier < climatology (identical days) - UNCHANGED",
                "passes_unchanged_bar_within_experiment": verdict,
                "legs_failed": legs_failed,
                "original_v1_record_260_days": {
                    "brier": record["targets"][t]["metrics"][m]["brier"],
                    "climatology_brier": record["targets"][t]["climatology_brier"],
                    "alert_binomial_p": record["targets"][t]["metrics"][m]["alert"]["binomial_p"],
                    "graduation_pass": record["targets"][t]["metrics"][m]["graduation_pass"],
                },
            }
            for _, r in cal.iterrows():
                cal_rows.append(dict(date=r.date, fold=int(r.fold), target=t, model=m,
                                     y_true=float(r.y_true), y_proba=float(r.y_proba),
                                     y_proba_cal=float(r.y_proba_cal),
                                     platt_a=float(r.platt_a), platt_b=float(r.platt_b),
                                     platt_fit_n=int(r.platt_fit_n),
                                     platt_fit_folds=r.platt_fit_folds))

    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_parquet("data/ml/wf_predictions_platt_calibration_2026-10-27.parquet", index=False)
    with open("reports/ml_target1_platt_calibration_results_2026-10-27.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["targets"], indent=2))


if __name__ == "__main__":
    main()
