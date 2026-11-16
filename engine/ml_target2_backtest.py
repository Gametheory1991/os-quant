"""ML Stage C — Target 2 backtest phase (frozen protocol, no tuning).

Frozen spec: reports/ml_target2_experiment_spec_2026-10-27.md INCLUDING
Amendment A1 (amended fold construction and restated counts govern).
Phase 1 implementation (loading, labels, features, folds, baselines) is
reused unmodified from engine/ml_target2_auction.py. The model ladder is
the program's frozen ladder from engine/ml_models.py (same estimators,
same grids, same inner TimeSeriesSplit(3), same seed), with the single
adaptation the specification itself prescribes in §5-§6: the twelve
numeric features are standardised inside the pipeline and the group
(tenor / term) dummies are appended unscaled, one-hot encoded inside
the pipeline against the panel's frozen category list.

Exactly the eight learned configurations of spec §10 are evaluated
(ledger rows 7820-7827). Baselines are deterministic comparators.
Panels are never pooled. Offline evaluation only — no forward
issuance is wired anywhere in this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from engine import ml_harness as H
from engine import ml_models as M
from engine import ml_reporting as R
from engine import ml_target2_auction as T2

COUPON_FOLD_PARAMS = dict(initial_train=150, test_block=40, embargo_days=7)
BILLS_FOLD_PARAMS = dict(initial_train=400, test_block=100, embargo_days=7)
PANEL_CATEGORIES = {"coupon": T2.COUPON_TENORS, "bills": T2.BILL_REGULAR_TERMS}
PANEL_FOLD_PARAMS = {"coupon": COUPON_FOLD_PARAMS, "bills": BILLS_FOLD_PARAMS}
EXPECTED = {
    "coupon": {"sample": 357, "pooled_test": 207, "pooled_weak": 48, "n_folds": 6},
    "bills": {"sample": 1060, "pooled_test": 660, "pooled_weak": 158, "n_folds": 7},
}


def _preprocessor(categories: list[str], scale_numeric: bool) -> ColumnTransformer:
    num = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer([
        ("num", num, T2.FEATURE_NAMES),
        ("grp", OneHotEncoder(categories=[categories],
                              handle_unknown="ignore", sparse_output=False),
         ["group"]),
    ])


def ridge_fit_predict_df(Xtr: pd.DataFrame, ytr, Xte: pd.DataFrame, categories):
    pipe = Pipeline([("pre", _preprocessor(categories, True)),
                     ("model", Ridge())])
    gs = GridSearchCV(pipe, {"model__alpha": M.RIDGE_ALPHAS},
                      cv=TimeSeriesSplit(3), scoring="neg_mean_squared_error")
    gs.fit(Xtr, ytr)
    return gs.predict(Xte), {"alpha": float(gs.best_params_["model__alpha"])}


def logistic_fit_predict_df(Xtr: pd.DataFrame, ytr, Xte: pd.DataFrame, categories):
    pipe = Pipeline([("pre", _preprocessor(categories, True)),
                     ("model", LogisticRegression(max_iter=2000))])
    gs = GridSearchCV(pipe, {"model__C": M.LOGISTIC_CS},
                      cv=TimeSeriesSplit(3), scoring="neg_log_loss")
    gs.fit(Xtr, ytr)
    return gs.predict_proba(Xte)[:, 1], {"C": float(gs.best_params_["model__C"])}


def gbm_reg_fit_predict_df(Xtr: pd.DataFrame, ytr, Xte: pd.DataFrame, categories):
    pre = _preprocessor(categories, False)
    Ztr = pre.fit_transform(Xtr)
    Zte = pre.transform(Xte)
    m = HistGradientBoostingRegressor(**M.GBM_PARAMS)
    m.fit(Ztr, ytr)
    return m.predict(Zte), {}


def build_panel_sample(kind: str):
    auctions = T2.load_auctions()
    refunding = T2.refunding_mondays(auctions)
    panel = T2.coupon_panel(auctions) if kind == "coupon" else T2.bills_panel(auctions)
    sample = T2.build_sample(panel, auctions, kind)
    folds = T2.make_folds(sample, refunding=refunding, **PANEL_FOLD_PARAMS[kind])
    exp = EXPECTED[kind]
    assert len(sample) == exp["sample"], (kind, len(sample))
    distinct = T2.assert_cross_fold_test_uniqueness(folds)
    assert distinct == exp["pooled_test"], (kind, distinct)
    assert len(folds) == exp["n_folds"], (kind, len(folds))
    pooled_weak = int(sum(int(sample["weak"].iloc[f.test_idx].sum()) for f in folds))
    assert pooled_weak == exp["pooled_weak"], (kind, pooled_weak)
    return auctions, sample, folds


def run_panel(kind: str):
    """Walk-forward fit of every frozen configuration for one panel."""
    _, sample, folds = build_panel_sample(kind)
    categories = PANEL_CATEGORIES[kind]
    is_coupon = kind == "coupon"
    cols = T2.FEATURE_NAMES + ["group"]
    X = sample[cols]
    y_btc = sample["btc"].to_numpy(dtype=float)
    y_sur = sample["surprise"].to_numpy(dtype=float)
    y_weak = sample["weak"].to_numpy(dtype=float)
    rows, fold_meta, hypers = [], [], {}
    for f in folds:
        Xtr, Xte = X.iloc[f.train_idx], X.iloc[f.test_idx]
        test = sample.iloc[f.test_idx]
        train_weak = y_weak[f.train_idx]
        p = {}
        p["t2a_ridge"], h = ridge_fit_predict_df(Xtr, y_btc[f.train_idx], Xte, categories)
        hp = {"t2a_ridge_alpha": h["alpha"]}
        p["t2b_ridge"], h = ridge_fit_predict_df(Xtr, y_sur[f.train_idx], Xte, categories)
        hp["t2b_ridge_alpha"] = h["alpha"]
        if is_coupon:
            p["t2a_gbm"], _ = gbm_reg_fit_predict_df(Xtr, y_btc[f.train_idx], Xte, categories)
            p["t2b_gbm"], _ = gbm_reg_fit_predict_df(Xtr, y_sur[f.train_idx], Xte, categories)
        p["t2c_logistic"], h = logistic_fit_predict_df(Xtr, train_weak, Xte, categories)
        hp["t2c_logistic_C"] = h["C"]
        clim = T2.climatology_t2c(train_weak, len(f.test_idx))
        base_a = T2.baseline_t2a(test)
        hypers[f"fold{f.fold_id}"] = hp
        for i, pos in enumerate(f.test_idx):
            r = test.iloc[i]
            rows.append(dict(
                panel=kind, fold=int(f.fold_id), sample_pos=int(pos),
                auction_date=r["auction_date"], cusip=r["cusip"],
                security_type=r["security_type"], security_term=r["security_term"],
                group=r["group"],
                btc=float(y_btc[pos]), bench_med6=float(r["bench_med6_label"]),
                surprise=float(y_sur[pos]), weak=float(y_weak[pos]),
                pred_t2a_baseline=float(base_a[i]),
                pred_t2a_ridge=float(p["t2a_ridge"][i]),
                pred_t2a_gbm=(float(p["t2a_gbm"][i]) if is_coupon else np.nan),
                pred_t2b_baseline=0.0,
                pred_t2b_ridge=float(p["t2b_ridge"][i]),
                pred_t2b_gbm=(float(p["t2b_gbm"][i]) if is_coupon else np.nan),
                pred_t2c_climatology=float(clim[i]),
                pred_t2c_logistic_proba=float(p["t2c_logistic"][i]),
            ))
        fold_meta.append(dict(fold=int(f.fold_id), **f.meta))
    return sample, folds, pd.DataFrame(rows), fold_meta, hypers


# ---------------------------------------------------------------------------
# Metrics (spec §8)
# ---------------------------------------------------------------------------
def _reg_block(df: pd.DataFrame, pred_col: str, y_col: str) -> dict:
    yt = df[y_col].to_numpy(dtype=float)
    yp = df[pred_col].to_numpy(dtype=float)
    return {"n": int(len(df)), "mae": float(np.mean(np.abs(yt - yp)))}


def regression_metrics(preds: pd.DataFrame, subtarget: str) -> dict:
    """subtarget: 't2a' (y=btc, baseline=bench) or 't2b' (y=surprise, base=0)."""
    y_col = "btc" if subtarget == "t2a" else "surprise"
    base_col = f"pred_{subtarget}_baseline"
    models = ["ridge"] + (["gbm"] if (preds["panel"].iloc[0] == "coupon") else [])
    base = _reg_block(preds, base_col, y_col)
    out = {"baseline": {**base, "per_fold": {
        str(fid): _reg_block(g, base_col, y_col) for fid, g in preds.groupby("fold")}},
        "models": {}}
    for m in models:
        col = f"pred_{subtarget}_{m}"
        pooled = _reg_block(preds, col, y_col)
        per_fold = {str(fid): _reg_block(g, col, y_col)
                    for fid, g in preds.groupby("fold")}
        wins = sum(1 for fid, g in preds.groupby("fold")
                   if _reg_block(g, col, y_col)["mae"] < _reg_block(g, base_col, y_col)["mae"])
        out["models"][m] = {
            **pooled,
            "mae_ratio_vs_baseline": pooled["mae"] / base["mae"],
            "mae_diff_vs_baseline": pooled["mae"] - base["mae"],
            "per_fold": per_fold,
            "folds_won": int(wins), "n_folds": int(preds["fold"].nunique()),
            "win": bool(pooled["mae"] < base["mae"]),
            "verdict": ("WIN (within-experiment only)"
                        if pooled["mae"] < base["mae"] else "FAIL"),
        }
    return out


def _calibration_equal_count(probs, outcomes, n_bins: int = 5) -> dict:
    """Spec §8: five-bin EQUAL-COUNT reliability table + ECE."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    order = np.argsort(p, kind="stable")
    bins = []
    for idx in np.array_split(order, n_bins):
        bins.append({
            "n": int(len(idx)),
            "mean_predicted": float(p[idx].mean()),
            "realised_frequency": float(y[idx].mean()),
        })
    ece = float(sum((b["n"] / len(p)) * abs(b["mean_predicted"] - b["realised_frequency"])
                    for b in bins))
    fit = R.calibration_intercept_slope(p, y)
    return {"bins": bins, "ece": ece,
            "recalibration_intercept": fit["intercept"],
            "recalibration_slope": fit["slope"],
            "fit_reason": fit["reason"],
            "note": ("Five equal-count bins (pooled test probabilities sorted "
                     "ascending, stable order). Recalibration intercept/slope: "
                     "unpenalised logistic regression of outcome on logit(p), "
                     "p clipped to [0.001, 0.999] (program ml_reporting "
                     "convention). Reported descriptively; not a bar element.")}


def classification_metrics(preds: pd.DataFrame) -> dict:
    d = preds.reset_index(drop=True)  # fold order = chronological order
    y = d["weak"].to_numpy(dtype=float)
    p_mod = d["pred_t2c_logistic_proba"].to_numpy(dtype=float)
    p_clim = d["pred_t2c_climatology"].to_numpy(dtype=float)
    null_rate = float(p_clim.mean())
    alert = H.alert_budget_test(y, p_mod, null_rate)
    brier_mod = H.brier(y, p_mod)
    brier_clim = H.brier(y, p_clim)
    per_fold = {}
    for fid, g in d.groupby("fold"):
        per_fold[str(fid)] = {
            "n": int(len(g)),
            "brier_logistic": H.brier(g["weak"], g["pred_t2c_logistic_proba"]),
            "brier_climatology": H.brier(g["weak"], g["pred_t2c_climatology"]),
        }
    folds_won = sum(1 for v in per_fold.values()
                    if v["brier_logistic"] < v["brier_climatology"])
    leg_alert = bool(alert["binomial_p"] < 0.05)
    leg_brier = bool(brier_mod < brier_clim)
    passed = leg_alert and leg_brier
    failed = [name for name, ok in (("alert binomial p < 0.05", leg_alert),
                                    ("pooled Brier < climatology", leg_brier)) if not ok]
    return {
        "n_test": int(len(d)), "n_weak": int(y.sum()),
        "weak_rate_test": float(y.mean()),
        "climatology_null_rate": null_rate,
        "brier_logistic": brier_mod, "brier_climatology": brier_clim,
        "auc_logistic": H.auc(y, p_mod),
        "alert": alert,
        "per_fold": per_fold, "folds_won_brier": int(folds_won),
        "n_folds": int(d["fold"].nunique()),
        "calibration": _calibration_equal_count(p_mod, y),
        "leg_alert_binomial_p_lt_0.05": leg_alert,
        "leg_brier_beats_climatology": leg_brier,
        "verdict": "PASS" if passed else "FAIL",
        "failed_legs": failed,
    }
