"""ML Target 4a — Stage B-2 shrinkage-boundary test (memo proposal M3).

Frozen spec: reports/ml_target4a_b2_boundary_spec_2026-10-27.md (written
2026-10-27 before any Stage B-2 fitting). This module implements ONLY the
single change that spec freezes: the ridge penalty grid extended by one
step in its own x10 progression, {0.1, 1, 10, 100, 1000} -> {0.1, 1, 10,
100, 1000, 10000}, for the REDUCED (18-feature) variant only. Everything
else — panel, sample, folds, baselines, selection rule, seed, metrics,
graduation bar — is reused unmodified from engine/ml_t4a.py and
engine/ml_harness.py. engine/ml_models.py and its RIDGE_ALPHAS default are
NOT modified; the extended grid is supplied here.

Offline evaluation only — nothing here is issued or scheduled, and the
graduated model's forward issuance, scorecard, and grading window are not
touched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from engine import ml_t4a as T

EXTENDED_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
MODELS_B2 = ["climatology", "zero_change", "persistence", "ridge_ext"]


def ridge_fit_predict_ext(Xtr, ytr, Xte):
    """Identical construction to engine/ml_models.ridge_fit_predict
    (Pipeline: StandardScaler inside the pipeline + Ridge; inner
    TimeSeriesSplit(3); scoring negative MSE) with the frozen extended
    grid of the Stage B-2 spec substituted for RIDGE_ALPHAS."""
    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    gs = GridSearchCV(
        pipe,
        {"model__alpha": EXTENDED_ALPHAS},
        cv=TimeSeriesSplit(3),
        scoring="neg_mean_squared_error",
    )
    gs.fit(Xtr, ytr)
    return gs.predict(Xte), {"alpha": float(gs.best_params_["model__alpha"])}


def run_walk_forward_ext(sample: pd.DataFrame, features: list[str], folds: list):
    """Walk-forward for the REDUCED variant under the extended grid.
    Mirrors engine/ml_t4a.run_walk_forward, with ridge replaced by
    ridge-extended and no GBM (not in this test's scope)."""
    X = sample[features]
    y_all = sample[T.TARGET].to_numpy(dtype=float)
    rows, fold_meta, alphas = [], [], {}
    for f in folds:
        Xtr, Xte = X.iloc[f.train_idx], X.iloc[f.test_idx]
        ytr, yte = y_all[f.train_idx], y_all[f.test_idx]
        test_rows = sample.iloc[f.test_idx]
        preds = {
            "climatology": T.climatology_predict(sample, f.test_idx),
            "zero_change": np.zeros(len(f.test_idx)),
            "persistence": test_rows["ret5"].to_numpy(dtype=float),
        }
        preds["ridge_ext"], hp = ridge_fit_predict_ext(Xtr, ytr, Xte)
        alphas[f"fold{f.fold_id}"] = hp["alpha"]
        for model in MODELS_B2:
            p = preds[model]
            for i, pos in enumerate(f.test_idx):
                r = test_rows.iloc[i]
                rows.append(dict(
                    date=sample.index[pos], model=model,
                    y_true=float(yte[i]), y_pred=float(p[i]), fold=int(f.fold_id),
                    roll_window=bool(r["roll_window"]), season=str(r["season"]),
                ))
        fold_meta.append(dict(fold=int(f.fold_id), **f.meta,
                              n_purged=int(f.test_idx[0] - len(f.train_idx))))
    return pd.DataFrame(rows), fold_meta, alphas


def block(df: pd.DataFrame) -> dict:
    yt, yp = df["y_true"].to_numpy() * 100.0, df["y_pred"].to_numpy() * 100.0
    return {"n": int(len(df)), "mae_pct": float(np.mean(np.abs(yt - yp))),
            "rmse_pct": float(np.sqrt(np.mean((yt - yp) ** 2)))}


def model_metrics_b2(preds: pd.DataFrame) -> dict:
    """Same metric definitions as engine/ml_t4a.model_metrics, restricted
    to this test's model set."""
    clim_mae = block(preds[preds.model == "climatology"])["mae_pct"]
    pers_mae = block(preds[preds.model == "persistence"])["mae_pct"]
    out = {}
    for model in MODELS_B2:
        d = preds[preds.model == model].sort_values("date")
        b = block(d)
        entry = {**b,
                 "mae_ratio_vs_climatology": b["mae_pct"] / clim_mae,
                 "mae_ratio_vs_persistence": b["mae_pct"] / pers_mae}
        if model == "zero_change":
            entry["directional_accuracy"] = None
            entry["realized_positive_share_of_test_labels"] = float(
                np.mean(d["y_true"].to_numpy() > 0))
        else:
            entry["directional_accuracy"] = float(np.mean(
                np.sign(d["y_pred"].to_numpy()) == np.sign(d["y_true"].to_numpy())))
        entry["slices"] = {
            "roll": block(d[d.roll_window]),
            "non_roll": block(d[~d.roll_window]),
            "seasons": {s: block(d[d.season == s])
                        for s in ("heating", "shoulder", "cooling")},
        }
        entry["per_fold"] = {str(fid): block(g) for fid, g in d.groupby("fold")}
        out[model] = entry
    return out


def verdict_b2(metrics: dict) -> dict:
    """The frozen graduation bar of the original spec section 8, applied
    mechanically to the extended configuration (ridge_ext)."""
    clim = metrics["climatology"]
    mm = metrics["ridge_ext"]
    leg_a = bool(mm["mae_pct"] < clim["mae_pct"])
    leg_b = bool(mm["slices"]["non_roll"]["mae_pct"]
                 < clim["slices"]["non_roll"]["mae_pct"])
    season_losses = [s for s in ("heating", "shoulder", "cooling")
                     if mm["slices"]["seasons"][s]["mae_pct"]
                     > clim["slices"]["seasons"][s]["mae_pct"]]
    if leg_a and leg_b and len(season_losses) < 2:
        verdict = "GRADUATES"
    elif leg_a and leg_b:
        verdict = "SEASON-SCOPED (not graduation)"
    else:
        verdict = "FAIL (does not graduate)"
    return {"verdict": verdict,
            "leg_a_pooled_mae_beats_climatology": leg_a,
            "leg_b_non_roll_mae_beats_climatology": leg_b,
            "season_slices_lost_to_climatology": season_losses,
            "n_season_slices_lost": len(season_losses)}
