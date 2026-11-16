"""Walk-forward harness for ML Target 1 (Treasury liquidity).

Frozen spec: reports/ml_target1_experiment_spec.md. Strict walk-forward
only — expanding train, blocked tests, purge + embargo enforced with
assertions that raise. No random k-fold anywhere in this program.

All label windows / purge / embargo distances are measured in
full-panel calendar positions (trading-day steps in the feature-store
index, or in an OAS series' own observation index), never in sample-row
positions, because the usable sample has interior gaps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "data" / "ml"

# --- Frozen feature list (spec, 14 features) -------------------------------
# (model feature name, store column, transform)
FEATURE_DEFS = [
    ("log_illiq_agg", "illiq_agg", "log"),
    ("illiq_agg_z252", "illiq_agg_z252", None),
    ("illiq_chg5", "illiq_chg5", None),
    ("adv_chg5", "adv_chg5", None),
    ("onrun_share_agg", "onrun_share_agg", None),
    ("ats_share_agg", "ats_share_agg", None),
    ("share_3_5_agg", "share_3_5_agg", None),
    ("oas_hy", "oas_hy", None),
    ("oas_hy_chg5d", "oas_hy_chg5d", None),
    ("oas_hy_z252", "oas_hy_z252", None),
    ("tnx_chg63", "tnx_chg63", None),
    ("vix", "vix", None),
    ("slope_t10y2y", "slope_t10y2y", None),
    ("auction_day", "auction_day", None),
]
FEATURE_NAMES = [d[0] for d in FEATURE_DEFS]

TARGETS = ["T1a1", "T1a5", "T1b_HY", "T1b_IG"]
HORIZON = {"T1a1": 1, "T1a5": 5, "T1b_HY": 5, "T1b_IG": 5}

INITIAL_TRAIN = 252
TEST_BLOCK = 63
EMBARGO_DAYS = 5  # trading days strictly between last train date and first test date
SEED = 13


def load_panel(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path or (ML_DIR / "feature_store.parquet"))
    df = df.sort_index()
    assert df.index.is_monotonic_increasing and df.index.is_unique
    return df


def load_oas(ml_dir: Path | None = None) -> dict[str, pd.Series]:
    d = ml_dir or ML_DIR
    hy = pd.read_csv(d / "oas_hy.csv", parse_dates=["observation_date"])
    ig = pd.read_csv(d / "oas_ig.csv", parse_dates=["observation_date"])
    out = {
        "HY": hy.set_index("observation_date")["BAMLH0A0HYM2"].sort_index(),
        "IG": ig.set_index("observation_date")["BAMLC0A0CM"].sort_index(),
    }
    for s in out.values():
        assert s.index.is_monotonic_increasing and s.index.is_unique
        assert s.notna().all()
    return out


def build_feature_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """The 14 frozen features, indexed like the panel. Features only —
    every column is a store column (or its log) at the same date."""
    X = pd.DataFrame(index=panel.index)
    for name, col, tf in FEATURE_DEFS:
        v = panel[col].astype(float)
        X[name] = np.log(v) if tf == "log" else v
    return X


def oas_widening(panel_index: pd.DatetimeIndex, oas: pd.Series, h: int = 5) -> pd.Series:
    """OAS(t+h) - OAS(t) with steps in the OAS series' own index.

    Uses future OAS rows for the label only. Returns NaN where t is not
    an OAS observation date or t+h steps do not exist yet."""
    pos = oas.index.get_indexer(panel_index)  # -1 where panel date not in OAS index
    vals = oas.to_numpy()
    out = np.full(len(panel_index), np.nan)
    ok = (pos >= 0) & (pos + h < len(vals))
    idx = np.where(ok)[0]
    out[idx] = vals[pos[idx] + h] - vals[pos[idx]]
    return pd.Series(out, index=panel_index, name="oas_widen")


def build_labels(panel: pd.DataFrame, oas: dict[str, pd.Series]) -> pd.DataFrame:
    """Label-side frame indexed like the panel. Labels use ONLY future
    rows of the label series; nothing here is ever used as a feature."""
    illiq = panel["illiq_agg"].astype(float)
    lab = pd.DataFrame(index=panel.index)
    lab["illiq_t"] = illiq
    lab["T1a1"] = illiq.shift(-1)
    fwd = pd.concat([illiq.shift(-k) for k in range(1, 6)], axis=1)
    lab["T1a5"] = fwd.mean(axis=1).where(fwd.notna().all(axis=1))
    # Trailing-5d mean incl. t (persistence comparator for T1a5 / dir. acc.)
    back = pd.concat([illiq.shift(k) for k in range(0, 5)], axis=1)
    lab["illiq_trail5_mean"] = back.mean(axis=1).where(back.notna().all(axis=1))
    lab["oas_hy_widen5"] = oas_widening(panel.index, oas["HY"], 5)
    lab["oas_ig_widen5"] = oas_widening(panel.index, oas["IG"], 5)
    # Event flag: illiq_t in top decile of trailing 252 panel days ending
    # at t (window t-251..t, NaN stub days dropped from the reference
    # window). Computed with an explicit loop: pandas Rolling.apply
    # NaN-poisons any window containing a stub NaN, which would erase
    # the flag for a full year after each stub day.
    vals = illiq.to_numpy(dtype=float)
    ev = np.full(len(vals), np.nan)
    for i in range(251, len(vals)):
        if np.isnan(vals[i]):
            continue
        w = vals[i - 251 : i + 1]
        w = w[~np.isnan(w)]
        if len(w) < 126:
            continue
        ev[i] = float(vals[i] >= np.quantile(w, 0.90))
    lab["event_topdecile"] = ev
    return lab


def build_sample(panel: pd.DataFrame, oas: dict[str, pd.Series]) -> pd.DataFrame:
    """Common sample: all 14 features non-NaN AND all four labels
    computable (T1b label computable := widening non-NaN for both series;
    the 0/1 label itself is fold-specific via the per-fold threshold)."""
    X = build_feature_matrix(panel)
    lab = build_labels(panel, oas)
    df = X.join(lab)
    df["full_pos"] = np.arange(len(panel))
    keep = (
        df[FEATURE_NAMES].notna().all(axis=1)
        & df["T1a1"].notna()
        & df["T1a5"].notna()
        & df["oas_hy_widen5"].notna()
        & df["oas_ig_widen5"].notna()
        & df["illiq_trail5_mean"].notna()
        & df["event_topdecile"].notna()
    )
    sample = df.loc[keep].copy()
    sample["sample_pos"] = np.arange(len(sample))
    return sample


# --- Folds -------------------------------------------------------------------
@dataclass
class Fold:
    fold_id: int
    train_idx: np.ndarray  # sample positions
    test_idx: np.ndarray   # sample positions
    horizon: int
    meta: dict = field(default_factory=dict)


def make_folds(
    sample: pd.DataFrame,
    horizon: int,
    initial_train: int = INITIAL_TRAIN,
    test_block: int = TEST_BLOCK,
    embargo_days: int = EMBARGO_DAYS,
) -> list[Fold]:
    """Expanding-train walk-forward folds with purge + embargo.

    Purge: drop train rows whose label window (t, t+h] reaches the first
    test date (full-calendar positions). Embargo: >= embargo_days trading
    days strictly between the last train feature date and the first test
    feature date. Both are asserted per fold (hard errors)."""
    n = len(sample)
    full_pos = sample["full_pos"].to_numpy()
    folds: list[Fold] = []
    start = initial_train
    fid = 0
    while start < n:
        test_idx = np.arange(start, min(start + test_block, n))
        test_start_full = full_pos[test_idx[0]]
        cand = np.arange(0, start)
        cand_full = full_pos[cand]
        # Purge: label window (t, t+h] must end strictly before test start.
        keep_purge = cand_full + horizon < test_start_full
        # Embargo: >= embargo_days full trading days strictly between.
        keep_embargo = test_start_full - cand_full > embargo_days
        train_idx = cand[keep_purge & keep_embargo]
        assert len(train_idx) > 0, f"fold {fid}: empty training set"
        _assert_fold_integrity(sample, train_idx, test_idx, horizon, embargo_days)
        folds.append(
            Fold(
                fold_id=fid,
                train_idx=train_idx,
                test_idx=test_idx,
                horizon=horizon,
                meta={
                    "train_start": str(sample.index[train_idx[0]].date()),
                    "train_end": str(sample.index[train_idx[-1]].date()),
                    "test_start": str(sample.index[test_idx[0]].date()),
                    "test_end": str(sample.index[test_idx[-1]].date()),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                },
            )
        )
        fid += 1
        start += test_block
    return folds


def _assert_fold_integrity(sample, train_idx, test_idx, horizon, embargo_days):
    full_pos = sample["full_pos"].to_numpy()
    dates = sample.index
    test_start_full = full_pos[test_idx[0]]
    train_full = full_pos[train_idx]
    # No train label window touches the test block.
    assert (train_full + horizon < test_start_full).all(), (
        "purge violated: a training label window overlaps the test block"
    )
    # Embargo gap holds.
    gap = test_start_full - train_full.max()
    assert gap > embargo_days, f"embargo violated: gap={gap} <= {embargo_days}"
    # Train strictly before test in calendar time.
    assert dates[train_idx].max() < dates[test_idx].min(), "train/test order violated"
    assert len(np.intersect1d(train_idx, test_idx)) == 0, "train/test overlap"


def fold_threshold(sample: pd.DataFrame, train_idx: np.ndarray, widen_col: str) -> float:
    """80th percentile of the 5-day OAS widening over TRAINING rows only."""
    w = sample[widen_col].to_numpy()[train_idx]
    assert np.isfinite(w).all()
    return float(np.quantile(w, 0.80))


# --- Metrics -------------------------------------------------------------------
def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def directional_accuracy(pred, actual, reference) -> float:
    pred, actual, reference = map(np.asarray, (pred, actual, reference))
    return float(np.mean(np.sign(pred - reference) == np.sign(actual - reference)))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def auc(y, p) -> float | None:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, np.asarray(p)))


def alert_budget_test(y, p, null_rate: float, budget: float = 0.10) -> dict:
    """Top-decile (by predicted prob) alert test, pooled test rows."""
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    n_alert = int(np.ceil(budget * n))
    order = np.argsort(-p, kind="stable")  # ties: earlier rows first (date order upstream)
    alert_idx = order[:n_alert]
    k = int(y[alert_idx].sum())
    pval = float(stats.binomtest(k, n_alert, null_rate, alternative="greater").pvalue)
    return {
        "n_test": int(n),
        "n_alert": n_alert,
        "alert_hits": k,
        "alert_hit_rate": float(k / n_alert) if n_alert else float("nan"),
        "null_rate": float(null_rate),
        "binomial_p": pval,
    }
