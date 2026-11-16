"""Driver for ML Target 1, Phases 2–3 (frozen spec:
reports/ml_target1_experiment_spec.md). Walk-forward harness + full
model ladder; writes wf_predictions.parquet, ml_target1_results.json
and ml_phase2_3_results.md. No tuning beyond the frozen spec.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from engine import ml_harness as H
from engine import ml_models as M

np.random.seed(H.SEED)

REG_MODELS = ["persistence", "climatology", "ridge", "gbm"]
CLF_MODELS = ["climatology", "event_rule", "logistic", "gbm"]


def run_regression(sample, target):
    folds = H.make_folds(sample, horizon=H.HORIZON[target])
    rows, fold_meta, hyper = [], [], {}
    X = sample[H.FEATURE_NAMES]
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


def run_classification(sample, target, widen_col):
    folds = H.make_folds(sample, horizon=H.HORIZON[target])
    rows, fold_meta, hyper = [], [], {}
    X = sample[H.FEATURE_NAMES]
    widen = sample[widen_col].to_numpy(dtype=float)
    event = sample["event_topdecile"].to_numpy(dtype=float)
    for f in folds:
        thr = H.fold_threshold(sample, f.train_idx, widen_col)
        ytr = (widen[f.train_idx] >= thr).astype(int)
        yte = (widen[f.test_idx] >= thr).astype(int)
        Xtr, Xte = X.iloc[f.train_idx], X.iloc[f.test_idx]
        preds = {}
        preds["climatology"], _ = M.climatology_clf_predict(ytr, len(f.test_idx))
        preds["event_rule"], _ = M.event_rule_predict(event[f.train_idx], ytr, event[f.test_idx])
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
    sample = H.build_sample(panel, oas)
    sample_info = dict(
        n_rows=int(len(sample)),
        date_start=str(sample.index.min().date()),
        date_end=str(sample.index.max().date()),
        panel_rows=int(len(panel)),
        panel_start=str(panel.index.min().date()),
        panel_end=str(panel.index.max().date()),
        features=H.FEATURE_NAMES,
    )
    print(f"sample: {sample_info['n_rows']} rows {sample_info['date_start']} -> {sample_info['date_end']}")

    all_preds, results = [], {"sample": sample_info, "targets": {}}
    fold_tables = {}

    for target in ("T1a1", "T1a5"):
        preds, folds, hyper = run_regression(sample, target)
        all_preds.append(preds)
        fold_tables[target] = folds
        metrics = reg_metrics(sample, preds, target)
        n_test = int(len(preds[preds.model == "persistence"]))
        results["targets"][target] = dict(
            kind="regression", folds=folds, hyperparameters=hyper, metrics=metrics,
            n_test=n_test,
            effective_nonoverlapping=n_test if target == "T1a1" else n_test / 5,
        )
        print(target, {m: round(metrics[m]["mae"], 10) for m in REG_MODELS})

    for target, widen_col in (("T1b_HY", "oas_hy_widen5"), ("T1b_IG", "oas_ig_widen5")):
        preds, folds, hyper = run_classification(sample, target, widen_col)
        all_preds.append(preds)
        fold_tables[target] = folds
        metrics, clim_brier = clf_metrics(preds, target)
        n_test = int(len(preds[preds.model == "climatology"]))
        results["targets"][target] = dict(
            kind="classification", folds=folds, hyperparameters=hyper, metrics=metrics,
            climatology_brier=clim_brier, n_test=n_test,
            effective_nonoverlapping=n_test / 5,
        )
        print(target, {m: round(metrics[m]["brier"], 4) for m in CLF_MODELS},
              "grad:", {m: metrics[m]["graduation_pass"] for m in CLF_MODELS})

    preds = pd.concat(all_preds, ignore_index=True).sort_values(
        ["target", "model", "date"]
    )
    preds.to_parquet("data/ml/wf_predictions.parquet", index=False)
    with open("reports/ml_target1_results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    write_report(results)
    print("wrote data/ml/wf_predictions.parquet, reports/ml_target1_results.json, reports/ml_phase2_3_results.md")


def fmt(x, nd=4):
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def write_report(results):
    s = results["sample"]
    L = []
    L.append("# ML Target 1 — Phases 2–3 Results (walk-forward, frozen spec)")
    L.append("")
    L.append("Spec: `reports/ml_target1_experiment_spec.md` (frozen 2026-10-13 before any fit; "
             "14 features, expanding walk-forward, purge + ≥5-day embargo, no random k-fold, no neural nets).")
    L.append("")
    L.append(f"**Sample:** {s['n_rows']} rows, {s['date_start']} → {s['date_end']} "
             f"(common sample: all 14 features non-NaN and all four labels computable; "
             f"panel {s['panel_rows']} rows {s['panel_start']} → {s['panel_end']}).")
    L.append("")
    L.append("Features: " + ", ".join(f"`{f}`" for f in s["features"]) + ".")
    L.append("")
    for target, t in results["targets"].items():
        folds = t["folds"]
        L.append(f"## {target} ({t['kind']}) — {len(folds)} folds, n_test={t['n_test']}, "
                 f"effective non-overlapping n={t['effective_nonoverlapping']}")
        L.append("")
        if t["kind"] == "classification":
            thr = ", ".join(f"fold {f['fold']}: thr={f['threshold']:.4f} pp (train base rate {f['train_base_rate']:.3f})"
                             for f in folds)
            L.append(f"Per-fold widening thresholds (80th pct of train-fold 5d widening): {thr}.")
            L.append("")
            L.append("| Model | Brier | AUC | Alert hit rate (top 10%) | Alert binomial p | Graduation |")
            L.append("|---|---|---|---|---|---|")
            for m, mm in t["metrics"].items():
                L.append(f"| {m} | {fmt(mm['brier'])} | {fmt(mm['auc'])} | "
                         f"{fmt(mm['alert']['alert_hit_rate'], 3)} ({mm['alert']['alert_hits']}/{mm['alert']['n_alert']}) | "
                         f"{fmt(mm['alert']['binomial_p'], 4)} | "
                         f"{'PASS' if mm['graduation_pass'] else 'FAIL'} |")
            L.append("")
            L.append(f"Climatology Brier: {fmt(t['climatology_brier'])}. Graduation bar: alert binomial p < 0.05 AND Brier < climatology.")
        else:
            L.append("| Model | MAE | RMSE | MAE / persistence MAE | Directional acc. | Beats persistence |")
            L.append("|---|---|---|---|---|---|")
            for m, mm in t["metrics"].items():
                L.append(f"| {m} | {mm['mae']:.4e} | {mm['rmse']:.4e} | {fmt(mm['mae_ratio_vs_persistence'], 3)} | "
                         f"{fmt(mm['directional_accuracy'], 3)} | {'YES' if mm['beats_persistence'] else 'NO'} |")
        L.append("")
    L.append("## Verdicts")
    L.append("")
    for target, t in results["targets"].items():
        if t["kind"] == "regression":
            for m, mm in t["metrics"].items():
                if m in ("ridge", "gbm"):
                    L.append(f"- {target} {m}: pooled MAE {mm['mae']:.4e} vs persistence — "
                             f"{'BEATS persistence' if mm['beats_persistence'] else 'LOSES to persistence (negative result)'}.")
        else:
            for m, mm in t["metrics"].items():
                if m in ("logistic", "gbm", "event_rule"):
                    L.append(f"- {target} {m}: Brier {fmt(mm['brier'])} vs climatology {fmt(t['climatology_brier'])}, "
                             f"alert p={fmt(mm['alert']['binomial_p'], 4)} — graduation {'PASS' if mm['graduation_pass'] else 'FAIL'}.")
    L.append("")
    L.append("Full per-fold metrics, fold date ranges, thresholds and chosen hyperparameters: "
             "`reports/ml_target1_results.json`. Every out-of-sample prediction: `data/ml/wf_predictions.parquet`.")
    L.append("")
    L.append("## Notes")
    L.append("")
    L.append("- Persistence directional accuracy is 0.000 by construction: its prediction minus the "
             "reference is identically zero (sign 0), which only matches a exactly-zero actual move. "
             "The metric is meaningful for climatology / ridge / gbm, whose predictions vary.")
    L.append("- The common sample starts 2024-06-24, not at the feature warmup (2024-04-16): the "
             "event-rule flag needs a full trailing-252-row window and the panel's early index is "
             "sparse. 512 sample rows → 5 folds, 260 pooled test rows per target.")
    L.append("- Effective non-overlapping counts: T1a1 = 260 (1-day target); T1a5 / T1b_HY / T1b_IG = 260/5 = 52. "
             "All T1b claims rest on ~52 independent outcomes — small-sample, per the design's own warning.")
    L.append("")
    with open("reports/ml_phase2_3_results.md", "w") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    main()
