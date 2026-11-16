"""Model ladder for ML Target 1 — exactly as frozen in
reports/ml_target1_experiment_spec.md. Baselines first, then ridge /
logistic (inner TimeSeriesSplit(3) selection, training fold only), then
GBM with frozen hyperparameters. No neural nets (overfitting stop list).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 13
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
LOGISTIC_CS = [0.01, 0.1, 1.0, 10.0]
GBM_PARAMS = dict(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=15,
    min_samples_leaf=20,
    l2_regularization=1.0,
    random_state=SEED,
)


def _inner_cv():
    return TimeSeriesSplit(3)


# --- T1a regression ------------------------------------------------------------
def ridge_fit_predict(Xtr, ytr, Xte):
    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    gs = GridSearchCV(
        pipe,
        {"model__alpha": RIDGE_ALPHAS},
        cv=_inner_cv(),
        scoring="neg_mean_squared_error",
    )
    gs.fit(Xtr, ytr)
    return gs.predict(Xte), {"alpha": float(gs.best_params_["model__alpha"])}


def gbm_reg_fit_predict(Xtr, ytr, Xte):
    m = HistGradientBoostingRegressor(**GBM_PARAMS)
    m.fit(Xtr, ytr)
    return m.predict(Xte), {}


def persistence_predict(sample_rows, target: str) -> np.ndarray:
    if target == "T1a1":
        return sample_rows["illiq_t"].to_numpy(dtype=float)
    if target == "T1a5":
        return sample_rows["illiq_trail5_mean"].to_numpy(dtype=float)
    raise ValueError(target)


def climatology_reg_predict(y_train: np.ndarray, n_test: int):
    return np.full(n_test, float(np.mean(y_train))), {"train_mean": float(np.mean(y_train))}


# --- T1b classification ----------------------------------------------------------
def climatology_clf_predict(y_train: np.ndarray, n_test: int):
    rate = float(np.mean(y_train))
    return np.full(n_test, rate), {"train_rate": rate}


def event_rule_predict(event_train: np.ndarray, y_train: np.ndarray, event_test: np.ndarray):
    """Probability = train-fold hit rate among event days / non-event days.
    Fallback to the overall train hit rate when a cell is empty."""
    overall = float(np.mean(y_train))
    ev = event_train.astype(bool)
    rate_ev = float(np.mean(y_train[ev])) if ev.any() else overall
    rate_non = float(np.mean(y_train[~ev])) if (~ev).any() else overall
    p = np.where(event_test.astype(bool), rate_ev, rate_non)
    return p, {"rate_event": rate_ev, "rate_non_event": rate_non, "train_rate": overall}


def logistic_fit_predict(Xtr, ytr, Xte):
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]
    )
    gs = GridSearchCV(
        pipe,
        {"model__C": LOGISTIC_CS},
        cv=_inner_cv(),
        scoring="neg_log_loss",
    )
    gs.fit(Xtr, ytr)
    proba = gs.predict_proba(Xte)[:, 1]
    return proba, {"C": float(gs.best_params_["model__C"])}


def gbm_clf_fit_predict(Xtr, ytr, Xte):
    m = HistGradientBoostingClassifier(**GBM_PARAMS)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1], {}
