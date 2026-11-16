"""ML Target 5a — per-fund ETF premium prediction (T5a-L1 / T5a-C5).

Pre-registered spec: reports/ml_target5a_experiment_spec_2026-10-28.md
(frozen 2026-10-28, before any Target 5a fit). This module implements
ONLY what that spec freezes: the section 3 panel reconstruction, the
section 6 feature list (13 numeric features + fund indicators +
day-of-week indicators), the section 7 pooled walk-forward protocol
(252 initial training dates / 63-date test blocks on the stratum date
axis, expanding window, purge + 5-date embargo, seed 13), the section 8
model ladder (persistence and trailing-median baselines, ridge with the
Target 1 / Target 4a alpha policy, gradient-boosting regressor with the
program's frozen hyperparameters), and the section 10 metrics and
graduation bar.

Offline evaluation only — nothing here is issued or scheduled, and no
existing module's defaults are modified.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from engine import ml_models as M

ROOT = Path(__file__).resolve().parent.parent
SCREENER_DIR = (Path.home() / "workspace" / "goals" /
                "ultimate-arbitrage-screener-mispricing-dislocation-flagging-machine")
NAV_CACHE = SCREENER_DIR / "hidden_files" / "nav_cache"
PRICE_CACHE = ROOT / "data" / "cache"
T5A_DIR = ROOT / "data" / "t5a_premium"
PANEL_PATH = T5A_DIR / "panel_t5a.parquet"
PRED_PATH = T5A_DIR / "wf_predictions_t5a.parquet"
ELIGIBILITY_PATH = ROOT / "reports" / "ml_target5a_eligibility_2026-10-28.csv"
SPEC_PATH = ROOT / "reports" / "ml_target5a_experiment_spec_2026-10-28.md"
PREMIUM_TABLE_PATH = SCREENER_DIR / "hidden_files" / "phase2_premium_table.md"

PANEL_END = pd.Timestamp("2026-10-27")
SEED = 13
INITIAL_TRAIN = 252
TEST_BLOCK = 63
EMBARGO_DATES = 5

STD_FUNDS = [
    # SSGA navhist workbooks (19)
    "SPY", "GLD", "SRLN", "PRIV", "XSD", "XLB", "XLC", "XLE", "XLF",
    "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "KBE", "KRE", "KIE",
    # Invesco dng-api NAV history (29)
    "BKLN", "ICLO", "PSI", "SOXQ", "KBWB", "KBWP", "FDIQ", "KBWD", "PFI",
    "RSPC", "RSPD", "RSPS", "RSPG", "RSPF", "RSPH", "RSPN", "RSPM", "RSPR",
    "RSPT", "RSPU", "PSCD", "PSCC", "PSCE", "PSCF", "PSCH", "PSCI", "PSCM",
    "PSCT", "PSCU",
    # VanEck fundhistoprices workbooks (5)
    "CLOI", "CLOB", "BIZD", "SMH", "SMHX",
    # ARK nav-historical-change API (2)
    "ARKQ", "ARKF",
    # WisdomTree fund-history API (1); DWS performanceChart API (1)
    "WTAI", "XAIX",
]
PRO_FUNDS = ["USD", "SSG", "VIXY", "VIXM", "UVXY", "SVXY"]
ELIGIBLE = STD_FUNDS + PRO_FUNDS
STRATUM_OF = {**{t: "STD" for t in STD_FUNDS}, **{t: "PRO" for t in PRO_FUNDS}}
CREDIT_FUNDS = ["CLOI", "CLOB", "ICLO", "PRIV", "BKLN", "SRLN", "BIZD"]
LEVERAGED_PRO = ["USD", "SSG", "UVXY", "SVXY"]

NUMERIC_FEATURES = [
    "prem_level_bp", "prem_chg1_bp", "prem_chg5_bp", "prem_chg21_bp",
    "prem_med_abs_252_bp", "prem_med_252_bp",
    "ret1_bp", "ret5_bp", "ret21_bp",
    "logvol", "logvol_chg5", "vix_level", "vix_stress",
]
TARGETS = {"T5a-L1": 1, "T5a-C5": 5}
MODELS = ["persistence", "trailing_median", "ridge", "gbm"]
LEARNED = ["ridge", "gbm"]
GRAD_MIN_FUNDS = {"STD": 29, "PRO": 4}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_hashes() -> dict[str, str]:
    """Spec section 9.1 item 7: SHA-256 of every input file used."""
    out: dict[str, str] = {}
    for t in ELIGIBLE:
        out[f"nav_cache/{t}_nav.csv"] = sha256_file(NAV_CACHE / f"{t}_nav.csv")
        out[f"data/cache/{t}.parquet"] = sha256_file(PRICE_CACHE / f"{t}.parquet")
    out["data/cache/^VIX.parquet"] = sha256_file(PRICE_CACHE / "^VIX.parquet")
    out["reports/ml_target5a_eligibility_2026-10-28.csv"] = sha256_file(ELIGIBILITY_PATH)
    out["reports/ml_target5a_experiment_spec_2026-10-28.md"] = sha256_file(SPEC_PATH)
    return out


# --- Section 3 reconstruction ---------------------------------------------------

def load_closes_own(ticker: str) -> pd.DataFrame:
    """Own loader replicating screener class2.load_closes semantics
    (reset the date index, normalise dates, drop null closes, sort by
    date), extended with the volume column of spec section 3 step 7."""
    df = pd.read_parquet(PRICE_CACHE / f"{ticker}.parquet")
    df = df.reset_index().rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return (df[["date", "close", "volume"]].dropna(subset=["close"])
            .sort_values("date").reset_index(drop=True))


def reconstruct_fund(ticker: str, end: pd.Timestamp | None = PANEL_END,
                     ) -> pd.DataFrame:
    """Spec section 3 reconstruction for one eligible fund.

    Steps: drop missing / non-positive NAV rows before joining (step 2
    guard); price leg is the Yahoo raw close for stratum STD and the
    NAV cache's market_price column for stratum PRO (steps 1 and 3);
    inner join on date (step 4); premium = price / NAV - 1, in basis
    points (step 5); panel frozen at aligned dates on or before
    2026-10-27 (step 6)."""
    stratum = STRATUM_OF[ticker]
    nav = pd.read_csv(NAV_CACHE / f"{ticker}_nav.csv", parse_dates=["date"])
    nav["date"] = pd.to_datetime(nav["date"]).dt.normalize()
    nav = nav.loc[nav["nav"].notna() & (nav["nav"] > 0)].copy()
    closes = load_closes_own(ticker)
    if stratum == "PRO":
        merged = closes.merge(nav[["date", "nav", "market_price"]],
                              on="date", how="inner")
        merged["price"] = merged["market_price"]
    else:
        merged = closes.merge(nav[["date", "nav"]], on="date", how="inner")
        merged["price"] = merged["close"]
    merged = merged.sort_values("date").reset_index(drop=True)
    merged["premium"] = merged["price"] / merged["nav"] - 1.0
    merged = merged.dropna(subset=["premium"]).reset_index(drop=True)
    if end is not None:
        merged = merged.loc[merged["date"] <= end].reset_index(drop=True)
    out = pd.DataFrame({
        "ticker": ticker,
        "stratum": stratum,
        "date": merged["date"],
        "aligned_pos": np.arange(len(merged)),
        "price": merged["price"].astype(float),
        "nav": merged["nav"].astype(float),
        "premium_bp": merged["premium"].astype(float) * 10_000.0,
        "volume": merged["volume"].astype(float),
        "yahoo_close": merged["close"].astype(float),
    })
    assert (out["nav"] > 0).all() and (out["price"] > 0).all()
    return out


def build_panel() -> pd.DataFrame:
    parts = [reconstruct_fund(t) for t in ELIGIBLE]
    panel = pd.concat(parts, ignore_index=True)
    return panel.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_vix() -> pd.DataFrame:
    df = pd.read_parquet(PRICE_CACHE / "^VIX.parquet")
    df = df.reset_index().rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return (df[["date", "close"]].dropna().sort_values("date")
            .reset_index(drop=True).rename(columns={"close": "vix_close"}))


# --- Section 6 features -----------------------------------------------------------

def build_fund_features(fund: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """The 13 frozen numeric features for one fund's aligned panel,
    computed within fund from data dated on or before each row's date,
    plus both labels, label end dates, and the baseline inputs."""
    f = fund.sort_values("date").reset_index(drop=True).copy()
    prem = f["premium_bp"]
    price = f["price"]
    f["prem_level_bp"] = prem
    f["prem_chg1_bp"] = prem - prem.shift(1)
    f["prem_chg5_bp"] = prem - prem.shift(5)
    f["prem_chg21_bp"] = prem - prem.shift(21)
    f["prem_med_abs_252_bp"] = prem.abs().rolling(252, min_periods=20).median()
    f["prem_med_252_bp"] = prem.rolling(252, min_periods=20).median()
    f["ret1_bp"] = (price / price.shift(1) - 1.0) * 10_000.0
    f["ret5_bp"] = (price / price.shift(5) - 1.0) * 10_000.0
    f["ret21_bp"] = (price / price.shift(21) - 1.0) * 10_000.0
    lv = np.log1p(f["volume"].astype(float))
    f["logvol"] = lv
    f["logvol_chg5"] = lv.rolling(5).mean() - lv.rolling(10).mean().shift(5)
    f = pd.merge_asof(f.sort_values("date"), vix, on="date",
                      direction="backward")
    f["vix_level"] = f["vix_close"]
    f["vix_stress"] = (f["vix_level"] >= 30.0).astype(float)
    f = f.drop(columns=["vix_close"])
    wd = f["date"].dt.weekday  # Monday = 0 ... Friday = 4
    assert set(wd.unique()) <= {0, 1, 2, 3, 4}, "weekend aligned date"
    for k, name in [(1, "dow_tue"), (2, "dow_wed"), (3, "dow_thu"), (4, "dow_fri")]:
        f[name] = (wd == k).astype(float)
    f["y_L1"] = prem.shift(-1)
    f["y_C5"] = prem.shift(-5) - prem
    f["label_end_date_L1"] = f["date"].shift(-1)
    f["label_end_date_C5"] = f["date"].shift(-5)
    return f


def build_all_features(panel: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for t in ELIGIBLE:
        parts.append(build_fund_features(panel[panel["ticker"] == t], vix))
    return pd.concat(parts, ignore_index=True)


def design_columns(stratum: str) -> list[str]:
    funds = STD_FUNDS if stratum == "STD" else PRO_FUNDS
    return (NUMERIC_FEATURES + [f"fund_{t}" for t in funds]
            + ["dow_tue", "dow_wed", "dow_thu", "dow_fri"])


def build_sample(features: pd.DataFrame, target: str, stratum: str) -> pd.DataFrame:
    """Spec section 7.1 sample for one target x stratum: aligned
    position >= 21, all 13 numeric features non-NaN, label computable.
    Sorted by (date, ticker) — the pooled clock of section 7.2 and the
    ordering the inner TimeSeriesSplit sees."""
    funds = STD_FUNDS if stratum == "STD" else PRO_FUNDS
    ycol = "y_L1" if target == "T5a-L1" else "y_C5"
    endcol = "label_end_date_L1" if target == "T5a-L1" else "label_end_date_C5"
    d = features[(features["stratum"] == stratum)
                 & (features["aligned_pos"] >= 21)].copy()
    d = d[d[NUMERIC_FEATURES].notna().all(axis=1) & d[ycol].notna()].copy()
    d["y"] = d[ycol].astype(float)
    d["label_end_date"] = d[endcol]
    for t in funds:
        d[f"fund_{t}"] = (d["ticker"] == t).astype(float)
    d = d.sort_values(["date", "ticker"]).reset_index(drop=True)
    assert d[NUMERIC_FEATURES + ["y"]].notna().all().all()
    return d


# --- Section 7.2 pooled folds ------------------------------------------------------

def make_pooled_folds(sample: pd.DataFrame, horizon: int):
    """Expanding walk-forward folds counted in pooled sample dates on
    the stratum's date axis D (spec section 7.2): test dates
    D[252 + 63k ... 252 + 63(k+1) - 1]; training candidates dated
    strictly before the first test date; purge drops a training row
    whose label end date is on or after the first test date; embargo
    keeps a training row only if its feature date's position in D is
    at most (first test date position) - 6. Assertions raise."""
    dates_sorted = np.sort(sample["date"].unique())
    D = pd.DatetimeIndex(dates_sorted)
    pos_in_D = {d: i for i, d in enumerate(D)}
    sample_pos_in_D = sample["date"].map(pos_in_D).to_numpy()
    sample_dates = sample["date"].to_numpy()
    label_end = sample["label_end_date"].to_numpy()
    folds = []
    k = 0
    while 252 + 63 * k < len(D):
        lo = 252 + 63 * k
        hi = min(lo + 63, len(D))
        test_dates = D[lo:hi]
        first_test = test_dates[0]
        test_mask = sample["date"].isin(test_dates).to_numpy()
        test_idx = np.where(test_mask)[0]
        cand_mask = sample_dates < np.datetime64(first_test)
        cand = np.where(cand_mask)[0]
        keep_embargo = sample_pos_in_D[cand] <= lo - 6
        keep_purge = label_end[cand] < np.datetime64(first_test)
        train_idx = cand[keep_embargo & keep_purge]
        # Counts stated exactly: candidates failing the embargo are
        # counted first; of the embargo survivors, those removed by the
        # purge are counted second. The two counts plus n_train sum to
        # the candidate count.
        n_cand = int(cand_mask.sum())
        n_after_embargo = int(keep_embargo.sum())
        n_train = int(len(train_idx))
        assert n_train > 0, f"fold {k}: empty training set"
        assert sample_dates[train_idx].max() < sample_dates[test_idx].min()
        gap = lo - sample_pos_in_D[train_idx].max()
        assert gap > EMBARGO_DATES, f"fold {k}: embargo gap {gap} <= 5"
        assert (label_end[train_idx] < np.datetime64(first_test)).all(), \
            f"fold {k}: purge violated"
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        folds.append({
            "fold_id": k,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "meta": {
                "fold": int(k),
                "test_start": str(pd.Timestamp(test_dates[0]).date()),
                "test_end": str(pd.Timestamp(test_dates[-1]).date()),
                "train_start": str(pd.Timestamp(sample_dates[train_idx[0]]).date()),
                "train_end": str(pd.Timestamp(sample_dates[train_idx[-1]]).date()),
                "n_train": n_train,
                "n_test": int(len(test_idx)),
                "n_candidates": n_cand,
                "n_dropped_embargo": int(n_cand - n_after_embargo),
                "n_dropped_purge_only": int(n_after_embargo - n_train),
                "n_funds_in_test": int(sample.iloc[test_idx]["ticker"].nunique()),
            },
        })
        k += 1
    return folds


# --- Section 8 model ladder ---------------------------------------------------------

def ridge_fit_predict_t5a(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray):
    """Ridge per spec section 8 item 3: the 13 numeric features are
    standardised inside the pipeline (statistics from the training
    fold only); fund and day-of-week indicators are appended unscaled;
    alpha selected from the frozen grid by inner TimeSeriesSplit(3)
    on the training fold only (lowest mean cross-validation mean
    squared error), then refit on the full training fold — the
    Target 1 / Target 4a policy and grid of engine/ml_models.py,
    assembled for a panel design matrix as in Target 2."""
    n_num = len(NUMERIC_FEATURES)
    pre = ColumnTransformer([
        ("num", StandardScaler(), list(range(n_num))),
        ("ind", "passthrough", list(range(n_num, Xtr.shape[1]))),
    ])
    pipe = Pipeline([("pre", pre), ("model", Ridge())])
    gs = GridSearchCV(pipe, {"model__alpha": M.RIDGE_ALPHAS},
                      cv=TimeSeriesSplit(3), scoring="neg_mean_squared_error")
    gs.fit(Xtr, ytr)
    return gs.predict(Xte), {"alpha": float(gs.best_params_["model__alpha"])}


def run_walk_forward(sample: pd.DataFrame, target: str, stratum: str,
                     folds: list) -> tuple[pd.DataFrame, dict]:
    """One target x stratum: both baselines and both learned models on
    identical folds and identical test rows (spec section 7.3)."""
    cols = design_columns(stratum)
    assert len(cols) == (74 if stratum == "STD" else 23)
    X = sample[cols].to_numpy(dtype=float)
    y_all = sample["y"].to_numpy(dtype=float)
    rows: list[dict] = []
    alphas: dict[str, float] = {}
    for f in folds:
        tr, te = f["train_idx"], f["test_idx"]
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y_all[tr], y_all[te]
        test_rows = sample.iloc[te]
        if target == "T5a-L1":
            pred_pers = test_rows["prem_level_bp"].to_numpy(dtype=float)
            pred_med = test_rows["prem_med_252_bp"].to_numpy(dtype=float)
        else:
            pred_pers = np.zeros(len(te))
            pred_med = (test_rows["prem_med_252_bp"]
                        - test_rows["prem_level_bp"]).to_numpy(dtype=float)
        pred_ridge, hp = ridge_fit_predict_t5a(Xtr, ytr, Xte)
        alphas[f"fold{f['fold_id']}"] = hp["alpha"]
        pred_gbm, _ = M.gbm_reg_fit_predict(Xtr, ytr, Xte)
        preds = {"persistence": pred_pers, "trailing_median": pred_med,
                 "ridge": pred_ridge, "gbm": pred_gbm}
        for model, p in preds.items():
            for i, pos in enumerate(te):
                r = test_rows.iloc[i]
                rows.append({
                    "ticker": str(r["ticker"]), "stratum": stratum,
                    "date": pd.Timestamp(r["date"]), "fold": int(f["fold_id"]),
                    "target": target, "model": model,
                    "y_true": float(yte[i]), "y_pred": float(p[i]),
                    "vix_stress": int(r["vix_stress"]),
                    "prem_level_bp": float(r["prem_level_bp"]),
                    "alpha": hp["alpha"] if model == "ridge" else np.nan,
                })
    return pd.DataFrame(rows), alphas


# --- Section 10 metrics ----------------------------------------------------------------

def _mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, dtype=float)
                                - np.asarray(p, dtype=float))))


def _rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, dtype=float)
                                  - np.asarray(p, dtype=float)) ** 2)))


def _dir_acc(d: pd.DataFrame, target: str) -> float:
    if target == "T5a-L1":
        return float(np.mean(np.sign(d["y_pred"] - d["prem_level_bp"])
                             == np.sign(d["y_true"] - d["prem_level_bp"])))
    return float(np.mean(np.sign(d["y_pred"]) == np.sign(d["y_true"])))


def block_metrics(d: pd.DataFrame, target: str) -> dict:
    return {"n": int(len(d)), "mae_bp": _mae(d["y_true"], d["y_pred"]),
            "rmse_bp": _rmse(d["y_true"], d["y_pred"]),
            "directional_accuracy": _dir_acc(d, target)}


def compute_metrics(preds: pd.DataFrame, target: str, stratum: str) -> dict:
    sub = preds[(preds["target"] == target) & (preds["stratum"] == stratum)]
    pers = sub[sub["model"] == "persistence"]
    med = sub[sub["model"] == "trailing_median"]
    pers_mae, med_mae = _mae(pers["y_true"], pers["y_pred"]), _mae(med["y_true"], med["y_pred"])
    n_test = int(len(pers))
    eff = n_test if target == "T5a-L1" else n_test / 5.0
    out: dict = {"n_test": n_test, "effective_independent_outcomes": eff,
                 "models": {}, "per_fund": {}, "verdicts": {}}
    for model in MODELS:
        d = sub[sub["model"] == model].sort_values(["date", "ticker"])
        b = block_metrics(d, target)
        b["mae_ratio_vs_persistence"] = b["mae_bp"] / pers_mae
        b["mae_ratio_vs_trailing_median"] = b["mae_bp"] / med_mae
        b["per_fold"] = {str(fid): block_metrics(g, target)
                         for fid, g in d.groupby("fold")}
        b["regime_slice"] = {
            "stress_vix_ge_30": block_metrics(d[d["vix_stress"] == 1], target)
            if (d["vix_stress"] == 1).any() else {"n": 0, "mae_bp": None,
                                                  "rmse_bp": None,
                                                  "directional_accuracy": None},
            "calm_vix_lt_30": block_metrics(d[d["vix_stress"] == 0], target),
        }
        if stratum == "STD":
            b["credit_slice"] = {
                "credit_7_funds": block_metrics(d[d["ticker"].isin(CREDIT_FUNDS)], target),
                "remaining_50_funds": block_metrics(d[~d["ticker"].isin(CREDIT_FUNDS)], target),
            }
        out["models"][model] = b
    # Per-fund table (descriptive slices of the pooled configurations).
    funds = STD_FUNDS if stratum == "STD" else PRO_FUNDS
    for t in funds:
        entry: dict = {}
        base = sub[(sub["model"] == "persistence") & (sub["ticker"] == t)]
        entry["n_test"] = int(len(base))
        entry["effective_independent_outcomes"] = (
            int(len(base)) if target == "T5a-L1" else len(base) / 5.0)
        entry["n_test_blocks"] = int(base["fold"].nunique())
        for model in MODELS:
            d = sub[(sub["model"] == model) & (sub["ticker"] == t)]
            entry[f"mae_{model}"] = _mae(d["y_true"], d["y_pred"])
        for model in LEARNED:
            entry[f"{model}_beats_persistence"] = bool(
                entry[f"mae_{model}"] < entry["mae_persistence"])
        out["per_fund"][t] = entry
    # Graduation bar (spec section 10, applied mechanically).
    denom = len(funds)
    for model in LEARNED:
        b = out["models"][model]
        leg_a = bool(b["mae_bp"] < pers_mae and b["mae_bp"] < med_mae)
        n_beat = int(sum(1 for t in funds
                         if out["per_fund"][t][f"{model}_beats_persistence"]))
        leg_b = bool(n_beat >= GRAD_MIN_FUNDS[stratum])
        out["verdicts"][model] = {
            "verdict": "GRADUATES" if (leg_a and leg_b) else "DOES NOT GRADUATE",
            "leg_a_pooled_mae_below_both_baselines": leg_a,
            "model_mae_bp": b["mae_bp"],
            "persistence_mae_bp": pers_mae,
            "trailing_median_mae_bp": med_mae,
            "leg_b_funds_beating_persistence": n_beat,
            "leg_b_denominator_funds": denom,
            "leg_b_required": GRAD_MIN_FUNDS[stratum],
            "leg_b_pass": leg_b,
        }
        # Concentration disclosure (reporting only, not a bar leg).
        imp_total, per_fund_imp = 0.0, {}
        for t in funds:
            pe = np.abs(base_err(sub, "persistence", t))
            me = np.abs(base_err(sub, model, t))
            imp = float(pe.sum() - me.sum())
            per_fund_imp[t] = imp
            imp_total += imp
        if imp_total > 0:
            top = max(per_fund_imp, key=per_fund_imp.get)
            out["verdicts"][model]["concentration"] = {
                "largest_fund": top,
                "largest_fund_share_of_improvement": per_fund_imp[top] / imp_total,
                "total_improvement_bp_sum": imp_total,
            }
        else:
            out["verdicts"][model]["concentration"] = {
                "largest_fund": None,
                "largest_fund_share_of_improvement": None,
                "total_improvement_bp_sum": imp_total,
                "note": "total improvement against persistence is not positive; "
                        "share not stated per spec section 10",
            }
    return out


def base_err(sub: pd.DataFrame, model: str, ticker: str) -> np.ndarray:
    d = sub[(sub["model"] == model) & (sub["ticker"] == ticker)]
    return (d["y_true"] - d["y_pred"]).to_numpy(dtype=float)
