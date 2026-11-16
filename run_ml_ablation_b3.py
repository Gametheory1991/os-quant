"""Runner — ML feature-block ablations, first cycle (Stage B-3, M2).

Frozen spec: reports/ml_feature_block_ablation_spec_2026-10-27.md
(frozen 2026-10-27 before any fitting in this cycle). Trials booked
in the ledger (rows 7793-7802) before this runner was executed.

Uses the standing component engine/ml_ablation.py for block
validation, cell enumeration, walk-forward predictions, and the
verdict mapping. Model constructions, folds, thresholds, and metrics
are the program's existing machinery (engine/ml_models.py,
engine/ml_harness.py, engine/ml_t4a.py), unmodified.

DIAGNOSTIC ONLY: these ablations can demote a claim or block a
promotion; they can never graduate a model. Offline only; no live
issuance changes; no parent predictions file is overwritten.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine import ml_ablation as A
from engine import ml_harness as H
from engine import ml_harness_v2 as H2
from engine import ml_models as M
from engine import ml_t4a as T4

np.random.seed(H.SEED)

ROOT = Path(__file__).resolve().parent
RESULTS_JSON = ROOT / "reports" / "ml_feature_block_ablation_results_2026-10-27.json"
PRED_T1 = ROOT / "data" / "ml" / "wf_predictions_ablation_b3_2026-10-27.parquet"
PRED_T4A = T4.T4A_DIR / "wf_predictions_ablation_b3_2026-10-27.parquet"
V2_RESULTS = ROOT / "reports" / "ml_target1_v2_results.json"
# Spec Amendment 1 (2026-10-27): the machinery check is a divergence
# guard at 2e-02 relative against the record (the feature store is a
# revised vintage; verdicts use the champion re-run on the current
# vintage, on identical days). See the spec's Amendment 1 for the
# diagnosis and the discarded first partial run.
TOL = 2e-2

# --- Blocks (spec §3) ------------------------------------------------------------
CREDIT_V2 = A.FeatureBlock(
    "T1-CREDIT",
    ("oas_hy", "oas_hy_chg5d", "oas_hy_z252", "oas_ig", "oas_ig_chg5d", "oas_ig_z252"),
    "credit/spread block (v2-frozen32 membership)",
)
LAGTGT_V2 = A.FeatureBlock(
    "T1-LAGTGT",
    ("log_illiq_agg", "illiq_agg_z252", "illiq_chg5",
     "illiq_vel7_252", "illiq_vel30_252", "illiq_chg21"),
    "lagged target-transform block (v2-frozen32 membership)",
)
CREDIT_V1 = A.FeatureBlock(
    "T1-CREDIT", ("oas_hy", "oas_hy_chg5d", "oas_hy_z252"),
    "credit/spread block (v1-frozen14 membership)",
)
LAGTGT_V1 = A.FeatureBlock(
    "T1-LAGTGT", ("log_illiq_agg", "illiq_agg_z252", "illiq_chg5"),
    "lagged target-transform block (v1-frozen14 membership)",
)
STORAGE = A.FeatureBlock(
    "T4A-STORAGE", ("stor_level", "stor_dev5y", "stor_wow", "stor_yoy"),
    "Target 4a storage block",
)
EXWX = A.FeatureBlock(
    "T4A-EXWX",
    ("hdd", "cdd", "hdd_7d", "cdd_7d", "hdd_30d", "cdd_30d", "dd_anom7"),
    "Target 4a experienced-weather block",
)

DIAGNOSTIC_NOTE = (
    "DIAGNOSTIC ONLY: ablations can demote a claim or block a promotion; "
    "they can never graduate a model."
)


def rel_diff(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b != 0 else abs(a - b)


def check(name: str, got: float, record: float) -> dict:
    ok = rel_diff(got, record) <= TOL
    print(f"  machinery check [{name}]: got {got:.10g}, record {record:.10g}, "
          f"rel diff {rel_diff(got, record):.2e} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(f"STOP: machinery check failed for {name}; "
                         "no ablated result is reported from this harness.")
    return {"check": name, "got": got, "record": record,
            "rel_diff": rel_diff(got, record), "pass": True}


# --- Regression cycles (T1a) -------------------------------------------------------
def regression_cycle(sample, features, blocks, target, fit_predict,
                     record_champion_mae, cell_id_start):
    """Champion + ablated runs via the standing component; baselines
    (persistence, climatology) recomputed per fold on identical rows."""
    folds = H.make_folds(sample, horizon=H.HORIZON[target])
    y = sample[target].to_numpy(dtype=float)
    res = A.run_ablation_cycle(sample, features, y, folds, fit_predict,
                               blocks, target=target, champion=target,
                               cell_id_prefix="TMP")
    # Baselines on the champion's identical test rows.
    pers_rows, clim_rows = [], []
    for f in folds:
        te = sample.iloc[f.test_idx]
        pers_rows.append(M.persistence_predict(te, target))
        clim_rows.append(np.full(len(f.test_idx),
                                 float(np.mean(y[f.train_idx]))))
    pers_pred = np.concatenate(pers_rows)
    clim_pred = np.concatenate(clim_rows)
    y_true = res.champion_preds["y_true"].to_numpy()
    pers_mae = H.mae(y_true, pers_pred)
    clim_mae = H.mae(y_true, clim_pred)

    def cell_metrics(preds):
        m = H.mae(preds["y_true"], preds["y_pred"])
        legs = {"leg_i_mae_below_persistence": bool(m < pers_mae),
                "leg_ii_mae_below_climatology": bool(m < clim_mae)}
        return {"mae": m, "mae_ratio_vs_persistence": m / pers_mae,
                "mae_ratio_vs_climatology": m / clim_mae, "legs": legs}

    champ_m = cell_metrics(res.champion_preds)
    machinery = check(f"{target} champion MAE", champ_m["mae"],
                      record_champion_mae)
    cells = []
    pred_frames = [res.champion_preds.assign(cell="CHAMPION", target=target)]
    for i, cell in enumerate(res.cells):
        cm = cell_metrics(res.ablated_preds[cell.cell_id])
        gid = f"B3-{cell_id_start + i:02d}"
        cells.append({
            "cell_id": gid, "target": target, "block": cell.block_name,
            "removed_features": list(cell.removed),
            "kept_features": list(cell.kept),
            "n_test": int(len(res.ablated_preds[cell.cell_id])),
            "champion": champ_m, "ablated": cm,
            "ablated_to_champion_mae_ratio": cm["mae"] / champ_m["mae"],
            "verdict": A.verdict_from_legs(cm["legs"]),
        })
        pred_frames.append(res.ablated_preds[cell.cell_id].assign(
            cell=gid, target=target))
    return {"folds": [dict(fold=f.fold_id, **f.meta) for f in folds],
            "baselines": {"persistence_mae": pers_mae,
                          "climatology_mae": clim_mae},
            "machinery_check": machinery, "cells": cells}, pred_frames


# --- Classification cycles (T1b) ---------------------------------------------------
def classification_cycle(sample, features, blocks, target, widen_col,
                         record, cell_id_start):
    """Per-fold labels are fold-specific (training-only thresholds), so
    predictions are produced fold by fold through the component's
    walk-forward machinery with that fold's binarized labels; days and
    labels are asserted identical across champion and ablated runs."""
    folds = H.make_folds(sample, horizon=H.HORIZON[target])
    widen = sample[widen_col].to_numpy(dtype=float)

    def run_for(features_):
        frames, clim_frames, hyper = [], [], {}
        for f in folds:
            thr = H.fold_threshold(sample, f.train_idx, widen_col)
            y_bin = (widen >= thr).astype(float)
            # Only this fold's train/test entries of y_bin are read.
            def fit_predict(Xtr, ytr, Xte, _f=f):
                p, hp = M.logistic_fit_predict(Xtr, ytr, Xte)
                hyper[f"fold{_f.fold_id}"] = hp
                return p
            frames.append(A.walk_forward_predict(
                sample, features_, y_bin, [f], fit_predict))
            clim_p, _ = M.climatology_clf_predict(
                y_bin[f.train_idx].astype(int), len(f.test_idx))
            clim_frames.append(pd.DataFrame({
                "date": sample.index[f.test_idx], "fold": int(f.fold_id),
                "y_true": y_bin[f.test_idx], "y_pred": clim_p}))
        return pd.concat(frames, ignore_index=True), pd.concat(
            clim_frames, ignore_index=True), hyper

    champ_preds, clim_preds, champ_hyper = run_for(features)
    clim_brier = H.brier(clim_preds["y_true"], clim_preds["y_pred"])
    null_rate = float(clim_preds["y_pred"].mean())

    def cell_metrics(preds):
        b = H.brier(preds["y_true"], preds["y_pred"])
        alert = H.alert_budget_test(preds["y_true"].to_numpy(),
                                    preds["y_pred"].to_numpy(), null_rate)
        legs = {"leg_i_brier_below_climatology": bool(b < clim_brier),
                "leg_ii_alert_p_below_0.05": bool(alert["binomial_p"] < 0.05)}
        return {"brier": b, "auc": H.auc(preds["y_true"], preds["y_pred"]),
                "alert": alert, "legs": legs}

    champ_m = cell_metrics(champ_preds)
    machinery = [
        check(f"{target} champion Brier", champ_m["brier"], record["brier"]),
        check(f"{target} champion alert p", champ_m["alert"]["binomial_p"],
              record["alert_p"]),
        check(f"{target} climatology Brier", clim_brier, record["clim_brier"]),
    ]
    A.validate_blocks(features, blocks)
    cells, pred_frames = [], [champ_preds.assign(cell="CHAMPION", target=target)]
    for i, block in enumerate(blocks):
        kept = A.kept_features(features, block)
        abl_preds, _, _ = run_for(kept)
        assert (abl_preds["date"].to_numpy()
                == champ_preds["date"].to_numpy()).all()
        assert (abl_preds["y_true"].to_numpy()
                == champ_preds["y_true"].to_numpy()).all()
        cm = cell_metrics(abl_preds)
        gid = f"B3-{cell_id_start + i:02d}"
        cells.append({
            "cell_id": gid, "target": target, "block": block.name,
            "removed_features": list(block.features), "kept_features": kept,
            "n_test": int(len(abl_preds)), "champion": champ_m, "ablated": cm,
            "verdict": A.verdict_from_legs(cm["legs"]),
        })
        pred_frames.append(abl_preds.assign(cell=gid, target=target))
    return {"folds": [dict(fold=f.fold_id, **f.meta) for f in folds],
            "climatology_brier": clim_brier, "null_rate": null_rate,
            "champion_hyperparameters": champ_hyper,
            "machinery_checks": machinery, "cells": cells}, pred_frames


# --- Target 4a cycle -----------------------------------------------------------------
def t4a_cycle(cell_id_start=9):
    panel = T4.load_panel()
    sample_full = T4.build_sample(panel, T4.REDUCED_FEATURES)
    # Spec Amendment 2 (2026-10-27): restrict to the original
    # experiment's sample dates (the panel has gained one row,
    # 2026-10-19, since the original run); assert the frozen count.
    sample = sample_full.loc[sample_full.index <= "2026-10-16"]
    assert len(sample) == T4.EXPECTED_SAMPLE_N, len(sample)
    assert str(sample.index.min().date()) == "2015-01-08"
    assert str(sample.index.max().date()) == "2026-10-16"
    folds = H.make_folds(sample, horizon=T4.HORIZON)
    y = sample[T4.TARGET].to_numpy(dtype=float)
    alphas = {}

    def ridge_fp(Xtr, ytr, Xte):
        p, hp = M.ridge_fit_predict(Xtr, ytr, Xte)
        alphas.setdefault("seen", []).append(hp["alpha"])
        return p

    res = A.run_ablation_cycle(sample, list(T4.REDUCED_FEATURES), y, folds,
                               ridge_fp, [STORAGE, EXWX],
                               target="T4a-NG-5d", champion="REDUCED ridge",
                               cell_id_prefix="TMP")
    roll = sample["roll_window"].to_numpy()[np.concatenate(
        [f.test_idx for f in folds])].astype(bool)
    # Baselines on identical rows (frozen Target 4a definitions).
    clim = np.concatenate([T4.climatology_predict(sample, f.test_idx)
                           for f in folds])
    y_true = res.champion_preds["y_true"].to_numpy()
    assert (res.champion_preds["date"].to_numpy()
            == sample.index[np.concatenate([f.test_idx for f in folds])]).all()

    def metrics(pred):
        mae_pct = float(np.mean(np.abs(y_true - pred)) * 100.0)
        nr = float(np.mean(np.abs(y_true[~roll] - pred[~roll])) * 100.0)
        return mae_pct, nr

    clim_mae, clim_nr = metrics(clim)
    champ_pred = res.champion_preds["y_pred"].to_numpy()
    champ_mae, champ_nr = metrics(champ_pred)
    machinery = [
        check("T4a champion pooled MAE %", champ_mae, 6.404230538927461),
        check("T4a climatology pooled MAE %", clim_mae, 6.517473483305825),
    ]

    def legs(mae_pct, nr):
        return {"leg_a_pooled_mae_below_climatology": bool(mae_pct < clim_mae),
                "leg_b_non_roll_mae_below_climatology": bool(nr < clim_nr)}

    champ_block = {"mae_pct": champ_mae, "non_roll_mae_pct": champ_nr,
                   "mae_ratio_vs_climatology": champ_mae / clim_mae,
                   "legs": legs(champ_mae, champ_nr)}
    cells, pred_frames = [], [res.champion_preds.assign(
        cell="CHAMPION", target="T4a-NG-5d")]
    for i, cell in enumerate(res.cells):
        ap = res.ablated_preds[cell.cell_id]["y_pred"].to_numpy()
        a_mae, a_nr = metrics(ap)
        cm = {"mae_pct": a_mae, "non_roll_mae_pct": a_nr,
              "mae_ratio_vs_climatology": a_mae / clim_mae,
              "legs": legs(a_mae, a_nr)}
        gid = f"B3-{cell_id_start + i:02d}"
        cells.append({
            "cell_id": gid, "target": "T4a-NG-5d", "block": cell.block_name,
            "removed_features": list(cell.removed),
            "kept_features": list(cell.kept),
            "n_test": int(len(ap)), "champion": champ_block, "ablated": cm,
            "ablated_to_champion_mae_ratio": a_mae / champ_mae,
            "verdict": A.verdict_from_legs(cm["legs"]),
        })
        pred_frames.append(res.ablated_preds[cell.cell_id].assign(
            cell=gid, target="T4a-NG-5d"))
    out = {"sample": {"n_rows": int(len(sample)),
                      "start": str(sample.index.min().date()),
                      "end": str(sample.index.max().date()),
                      "n_folds": len(folds),
                      "n_test": int(len(y_true))},
           "baselines": {"climatology_mae_pct": clim_mae,
                         "climatology_non_roll_mae_pct": clim_nr},
           "machinery_checks": machinery, "cells": cells}
    return out, pred_frames


def main():
    results = {"spec": "reports/ml_feature_block_ablation_spec_2026-10-27.md",
               "diagnostic_only": DIAGNOSTIC_NOTE, "cycles": {}}
    pred_frames_t1 = []

    panel = H.load_panel()
    oas = H.load_oas()
    v2 = json.loads(V2_RESULTS.read_text())

    # C1 — T1a1 ridge, v1 sample (panel as of the original v1 run).
    print("C1 T1a1 ridge (v1-frozen14), v1 sample...")
    panel_v1 = panel.loc[panel.index <= "2026-10-02"]
    sample_v1 = H.build_sample(panel_v1, oas)
    assert len(sample_v1) == 512, len(sample_v1)
    cyc, frames = regression_cycle(
        sample_v1, H.FEATURE_NAMES, [CREDIT_V1, LAGTGT_V1], "T1a1",
        lambda Xtr, ytr, Xte: M.ridge_fit_predict(Xtr, ytr, Xte)[0],
        record_champion_mae=1.1936506923576809e-05, cell_id_start=1)
    cyc["sample"] = {"n_rows": 512, "basis": "v1 common sample, panel truncated at 2026-10-02 (original v1 run)"}
    results["cycles"]["C1_T1a1_ridge_v1"] = cyc
    pred_frames_t1 += frames

    # C2 — T1a5 GBM, v2 intersection sample.
    print("C2 T1a5 GBM (v2-frozen32), v2 intersection sample...")
    sample_v2 = H2.build_sample_v2(panel, oas)
    assert len(sample_v2) == 514, len(sample_v2)
    rec_t1a5_gbm = v2["targets"]["T1a5"]["versions"]["v2"]["metrics"]["gbm"]["mae"]
    cyc, frames = regression_cycle(
        sample_v2, H2.FEATURE_NAMES_V2, [CREDIT_V2, LAGTGT_V2], "T1a5",
        lambda Xtr, ytr, Xte: M.gbm_reg_fit_predict(Xtr, ytr, Xte)[0],
        record_champion_mae=rec_t1a5_gbm, cell_id_start=3)
    cyc["sample"] = {"n_rows": 514, "basis": "v2 intersection sample"}
    results["cycles"]["C2_T1a5_gbm_v2"] = cyc
    pred_frames_t1 += frames

    # C3 / C4 — T1b logistic champions, v2 intersection sample.
    for target, widen_col, start in (("T1b_HY", "oas_hy_widen5", 5),
                                     ("T1b_IG", "oas_ig_widen5", 7)):
        print(f"{target} logistic (v2-frozen32)...")
        tv = v2["targets"][target]["versions"]["v2"]["metrics"]
        record = {"brier": tv["logistic"]["brier"],
                  "alert_p": tv["logistic"]["alert"]["binomial_p"],
                  "clim_brier": v2["targets"][target]["versions"]["v2"]
                  ["metrics"]["climatology"]["brier"]}
        cyc, frames = classification_cycle(
            sample_v2, H2.FEATURE_NAMES_V2, [CREDIT_V2, LAGTGT_V2],
            target, widen_col, record, cell_id_start=start)
        cyc["sample"] = {"n_rows": 514, "basis": "v2 intersection sample"}
        results["cycles"][f"{target}_logistic_v2"] = cyc
        pred_frames_t1 += frames

    # C5 — Target 4a REDUCED ridge.
    print("C5 Target 4a REDUCED ridge...")
    cyc4, frames4 = t4a_cycle(cell_id_start=9)
    results["cycles"]["C5_T4a_reduced_ridge"] = cyc4

    all_cells = [c for cyc in results["cycles"].values() for c in cyc["cells"]]
    all_cells.sort(key=lambda c: c["cell_id"])
    assert [c["cell_id"] for c in all_cells] == [f"B3-{i:02d}" for i in range(1, 11)], \
        [c["cell_id"] for c in all_cells]
    results["cells"] = all_cells
    results["n_cells"] = len(all_cells)

    RESULTS_JSON.write_text(json.dumps(results, indent=2, default=float))
    pd.concat(pred_frames_t1, ignore_index=True).to_parquet(PRED_T1, index=False)
    pd.concat(frames4, ignore_index=True).to_parquet(PRED_T4A, index=False)
    print(f"wrote {RESULTS_JSON.name}, {PRED_T1.name}, {PRED_T4A.name}")
    for c in all_cells:
        print(c["cell_id"], c["target"], c["block"], "->", c["verdict"])


if __name__ == "__main__":
    main()
