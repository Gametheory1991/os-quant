"""Driver for ML Target 1 v2 (offline comparison, pre-registered spec:
reports/ml_target1_v2_experiment_spec.md).

Runs the v1 ladder (14 features) AND the v2 ladder (32 features) on the
IDENTICAL intersection sample and folds, reusing the unmodified v1
harness (engine/ml_harness.py) and models (engine/ml_models.py).

Outputs (v1 artifacts are never overwritten):
  data/ml/wf_predictions_v2.parquet        — v2 walk-forward predictions
  data/ml/wf_predictions_v1_recheck.parquet — v1 re-run on the same sample
  reports/ml_target1_v2_results.json       — sample, folds, pooled +
                                             per-fold metrics, both versions
No issuance, no scorecard writes, no live anything.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from engine import ml_harness as H
from engine import ml_harness_v2 as H2
from engine import ml_models as M

np.random.seed(H.SEED)

REG_MODELS = ["persistence", "climatology", "ridge", "gbm"]
CLF_MODELS = ["climatology", "event_rule", "logistic", "gbm"]
VERSIONS = {"v1": H.FEATURE_NAMES, "v2": H2.FEATURE_NAMES_V2}


def run_regression(sample, target, feature_names):
    folds = H.make_folds(sample, horizon=H.HORIZON[target])
    rows, fold_meta, hyper = [], [], {}
    X = sample[feature_names]
    y_all = sample[target].to_numpy(dtype=float)
    for f in folds:
        Xtr, Xte = X.iloc[f.train_idx], X.iloc[f.test_idx]
        ytr, yte = y_all[f.train_idx], y_all[f.test_idx]
        preds = {}
        preds["persistence"] = M.persistence_predict(sample.iloc[f.test_idx], target)
        preds["climatology"], _ = M.climatology_reg_predict(ytr, len(f.test_idx))
        preds["ridge"], hp = M.ridge_fit_predict(Xtr, ytr, Xte)
        hyper[f"fold{f.fold_id}"] = hp
        preds["gbm"], _ = M.gbm_reg_fit_predict(Xtr, ytr, Xte)
        for model, p in preds.items():
            for i, pos in enumerate(f.test_idx):
                rows.append(
                    dict(
                        date=sample.index[pos], fold=f.fold_id, target=target,
                        model=model, y_true=float(yte[i]), y_pred=float(p[i]),
                        y_proba=np.nan, threshold=np.nan,
                    )
                )
        fold_meta.append(dict(fold=f.fold_id, **f.meta))
    return pd.DataFrame(rows), fold_meta, hyper


def run_classification(sample, target, widen_col, feature_names):
    folds = H.make_folds(sample, horizon=H.HORIZON[target])
    rows, fold_meta, hyper = [], [], {}
    X = sample[feature_names]
    widen = sample[widen_col].to_numpy(dtype=float)
    event = sample["event_topdecile"].to_numpy(dtype=float)
    for f in folds:
        thr = H.fold_threshold(sample, f.train_idx, widen_col)
        ytr = (widen[f.train_idx] >= thr).astype(int)
        yte = (widen[f.test_idx] >= thr).astype(int)
        Xtr, Xte = X.iloc[f.train_idx], X.iloc[f.test_idx]
        preds = {}
        preds["climatology"], _ = M.climatology_clf_predict(ytr, len(f.test_idx))
        preds["event_rule"], _ = M.event_rule_predict(
            event[f.train_idx], ytr, event[f.test_idx])
        preds["logistic"], hp = M.logistic_fit_predict(Xtr, ytr, Xte)
        hyper[f"fold{f.fold_id}"] = hp
        preds["gbm"], _ = M.gbm_clf_fit_predict(Xtr, ytr, Xte)
        for model, p in preds.items():
            for i, pos in enumerate(f.test_idx):
                rows.append(
                    dict(
                        date=sample.index[pos], fold=f.fold_id, target=target,
                        model=model, y_true=int(yte[i]), y_pred=np.nan,
                        y_proba=float(p[i]), threshold=float(thr),
                    )
                )
        fold_meta.append(dict(fold=f.fold_id, **f.meta, threshold=thr,
                              train_base_rate=float(np.mean(ytr))))
    return pd.DataFrame(rows), fold_meta, hyper


def reg_metrics(sample, preds, target):
    out = {}
    ref_col = "illiq_t" if target == "T1a1" else "illiq_trail5_mean"
    pers = preds[(preds.target == target) & (preds.model == "persistence")]
    pers_mae = H.mae(pers.y_true, pers.y_pred)
    for model in REG_MODELS:
        d = preds[(preds.target == target) & (preds.model == model)].sort_values("date")
        ref = sample.loc[d.date, ref_col].to_numpy(dtype=float)
        per_fold = {}
        for fid, g in d.groupby("fold"):
            per_fold[str(fid)] = dict(mae=H.mae(g.y_true, g.y_pred),
                                      rmse=H.rmse(g.y_true, g.y_pred), n=int(len(g)))
        m = dict(
            mae=H.mae(d.y_true, d.y_pred), rmse=H.rmse(d.y_true, d.y_pred),
            mae_ratio_vs_persistence=H.mae(d.y_true, d.y_pred) / pers_mae,
            directional_accuracy=H.directional_accuracy(d.y_pred, d.y_true, ref),
            n_test=int(len(d)), per_fold=per_fold,
        )
        m["beats_persistence"] = bool(m["mae"] < pers_mae)
        out[model] = m
    return out


def clf_metrics(preds, target):
    out = {}
    clim = preds[(preds.target == target) & (preds.model == "climatology")].sort_values("date")
    clim_brier = H.brier(clim.y_true, clim.y_proba)
    null_rate = float(clim.y_proba.mean())
    for model in CLF_MODELS:
        d = preds[(preds.target == target) & (preds.model == model)].sort_values("date")
        alert = H.alert_budget_test(d.y_true.to_numpy(), d.y_proba.to_numpy(), null_rate)
        per_fold = {}
        for fid, g in d.groupby("fold"):
            per_fold[str(fid)] = dict(
                brier=H.brier(g.y_true, g.y_proba), auc=H.auc(g.y_true, g.y_proba),
                n=int(len(g)), base_rate=float(g.y_true.mean()),
            )
        m = dict(
            brier=H.brier(d.y_true, d.y_proba), auc=H.auc(d.y_true, d.y_proba),
            n_test=int(len(d)), alert=alert, per_fold=per_fold,
        )
        m["graduation_pass"] = bool(
            alert["binomial_p"] < 0.05 and m["brier"] < clim_brier
        )
        out[model] = m
    return out, clim_brier


def main():
    panel = H.load_panel()
    oas = H.load_oas()
    v1_sample = H.build_sample(panel, oas)
    sample = H2.build_sample_v2(panel, oas)
    sample_info = dict(
        comparison_sample_n_rows=int(len(sample)),
        comparison_sample_date_start=str(sample.index.min().date()),
        comparison_sample_date_end=str(sample.index.max().date()),
        v1_sample_on_current_panel_n_rows=int(len(v1_sample)),
        v1_sample_on_current_panel_date_start=str(v1_sample.index.min().date()),
        v1_sample_on_current_panel_date_end=str(v1_sample.index.max().date()),
        dropped_from_v1_sample=[str(d.date()) for d in v1_sample.index.difference(sample.index)],
        panel_rows=int(len(panel)),
        panel_start=str(panel.index.min().date()),
        panel_end=str(panel.index.max().date()),
        features_v1=list(H.FEATURE_NAMES),
        features_v2=list(H2.FEATURE_NAMES_V2),
        model_version_suffix_v2=H2.MODEL_VERSION_SUFFIX_V2,
        protocol=dict(initial_train=H.INITIAL_TRAIN, test_block=H.TEST_BLOCK,
                      embargo_days=H.EMBARGO_DAYS, seed=H.SEED),
    )
    print(f"comparison sample: {sample_info['comparison_sample_n_rows']} rows "
          f"{sample_info['comparison_sample_date_start']} -> "
          f"{sample_info['comparison_sample_date_end']} "
          f"(v1 sample on current panel: {sample_info['v1_sample_on_current_panel_n_rows']})")

    results = {"spec": "reports/ml_target1_v2_experiment_spec.md",
               "sample": sample_info, "targets": {}}
    preds_by_version = {v: [] for v in VERSIONS}

    for target in ("T1a1", "T1a5"):
        results["targets"][target] = {"kind": "regression", "versions": {}}
        fold_ref = None
        for version, feats in VERSIONS.items():
            preds, folds, hyper = run_regression(sample, target, feats)
            if fold_ref is None:
                fold_ref = folds
            else:
                assert folds == fold_ref, f"{target}: folds differ across versions"
            preds_by_version[version].append(preds)
            metrics = reg_metrics(sample, preds, target)
            results["targets"][target]["versions"][version] = dict(
                folds=folds, hyperparameters=hyper, metrics=metrics,
                n_test=int(len(preds[preds.model == "persistence"])),
            )
            print(target, version,
                  {m: round(metrics[m]["mae"], 10) for m in REG_MODELS})

    for target, widen_col in (("T1b_HY", "oas_hy_widen5"), ("T1b_IG", "oas_ig_widen5")):
        results["targets"][target] = {"kind": "classification", "versions": {}}
        fold_ref = None
        for version, feats in VERSIONS.items():
            preds, folds, hyper = run_classification(sample, target, widen_col, feats)
            if fold_ref is None:
                fold_ref = folds
            else:
                # thresholds/base rates are part of fold meta and must match too
                assert folds == fold_ref, f"{target}: folds differ across versions"
            preds_by_version[version].append(preds)
            metrics, clim_brier = clf_metrics(preds, target)
            results["targets"][target]["versions"][version] = dict(
                folds=folds, hyperparameters=hyper, metrics=metrics,
                climatology_brier=clim_brier,
                n_test=int(len(preds[preds.model == "climatology"])),
            )
            print(target, version,
                  {m: round(metrics[m]["brier"], 4) for m in CLF_MODELS},
                  "grad:", {m: metrics[m]["graduation_pass"] for m in CLF_MODELS})

    v2_preds = pd.concat(preds_by_version["v2"], ignore_index=True).sort_values(
        ["target", "model", "date"])
    v1_preds = pd.concat(preds_by_version["v1"], ignore_index=True).sort_values(
        ["target", "model", "date"])
    v2_preds.to_parquet("data/ml/wf_predictions_v2.parquet", index=False)
    v1_preds.to_parquet("data/ml/wf_predictions_v1_recheck.parquet", index=False)
    with open("reports/ml_target1_v2_results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print("wrote data/ml/wf_predictions_v2.parquet, "
          "data/ml/wf_predictions_v1_recheck.parquet, "
          "reports/ml_target1_v2_results.json")


if __name__ == "__main__":
    main()
