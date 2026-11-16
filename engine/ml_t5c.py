"""ML Target 5c — per-fund ETF return prediction.

Frozen specification:
``~/workspace/goals/quant-research-backtest-engine/files/ml_target5c_experiment_spec_frozen_2026-11-04.md``
(frozen 2026-11-04, before any Target 5c fit; trials booked in the
programme trials ledger, rows 8451-8454). This module implements
only what that specification freezes:

- section 3 panel construction from the Yahoo price cache
  (``data/cache/<ticker>.parquet`` only), panel end frozen at
  2026-11-03, the ``adjclose`` column supplying the return series and
  the served ``close`` / ``open`` / ``high`` / ``low`` / ``volume``
  columns supplying the volume and market-state features; the
  ``dividends`` column is never used;
- section 4 universe: the 137 funds of the freeze audit appendix
  whose raw cache row count at the frozen panel end is at least 399
  (next-day target) / 404 (5-day target) — the same 137 funds at
  both thresholds at the freeze;
- section 6 feature list: 14 numeric features, plus 137 fund
  indicators and 4 day-of-week indicators (design width 155);
- section 7 sample rule: every numeric feature non-NaN and the
  label computable (the 5-day trailing-median baseline input is
  additionally required non-NaN for the 5-day target, because that
  baseline is otherwise undefined on the row and the comparison
  sample of section 8 must be identical across models and
  baselines);
- section 8 walk-forward protocol: expanding folds on the pooled
  sample-date axis, INITIAL_TRAIN = 252 pooled sample dates,
  TEST_BLOCK = 63, seed 13, purge on the label end date, and a
  5-pooled-sample-date embargo for both horizons;
- section 9 model ladder: ridge (numeric features standardised on
  the training fold, fund and day-of-week indicators unscaled;
  alpha selected from {0.1, 1, 10, 100, 1000} by inner
  TimeSeriesSplit(3) mean squared error) and gradient boosting with
  the frozen hyperparameters, both constructions following
  ``engine/ml_models.py`` (the alpha grid and the gradient-boosting
  parameters are imported from that module, and the
  gradient-boosting fit is that module's own function); the three
  deterministic comparators (persistence, trailing median, zero
  forecast) are computed in the driver;
- section 10 memory bound: the panel is built one fund at a time
  and appended incrementally; predictions are written per fold and
  per target to separate part files and are never concatenated in
  memory by any step of this pipeline.

Market-state features (features 12-14) follow the Target 5b
section 7.2 construction transferred by the frozen specification:
each market series is computed on its own positions and matched to
a fund row by the latest market-series date on or before the row's
date (an as-of join). Feature 14 (``spy_ret5_bp``) is computed from
the SPY served ``close`` column, as in Target 5b; feature 12 is the
``^VIX`` close and feature 13 is the change in the ``^TNX`` close
over that series' own positions ``i-5`` to ``i``. Feature 10
(``logvol``) is the natural logarithm of dollar volume (served
close times as-reported volume); it is undefined (NaN) where dollar
volume is zero, and such rows leave the sample under the section 7
rule.

Offline evaluation only. No forward issuance, no live prediction,
no trading implication.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from engine import ml_models as M

ROOT = Path(__file__).resolve().parent.parent
PRICE_CACHE = ROOT / "data" / "cache"
T5C_DIR = ROOT / "data" / "t5c_ret"
PANEL_PATH = T5C_DIR / "panel_t5c.parquet"
PARTS_DIR = T5C_DIR / "pred_parts"
STATE_PATH = T5C_DIR / "run_state.json"

GOAL_FILES = (Path.home() / "workspace" / "goals" /
              "quant-research-backtest-engine" / "files")
APPENDIX_PATH = GOAL_FILES / "ml_target5c_audit_appendix_2026-11-04.csv"
SPEC_PATH = GOAL_FILES / "ml_target5c_experiment_spec_frozen_2026-11-04.md"
SPEC_RECEIPT_SHA256 = "e9b56df7761dab27b4d53016397fbac3d6c90faee1394b520c964b90a235cb1d"
SPEC_RECEIPT_BYTES = 26948

PANEL_END = pd.Timestamp("2026-11-03")
TRUNC_END = pd.Timestamp("2022-12-31")
RUN_DATE = "2026-11-04"
SEED = 13
INITIAL_TRAIN = 252
TEST_BLOCK = 63
EMBARGO_DATES = 5

MIN_ROWS_R1 = 399
MIN_ROWS_R5 = 404
EXCLUDED_HISTORY = ["DRAM", "JA", "RAAR", "RAAY", "CHPX", "PCR", "AIPO", "RAAA"]
EXPECTED_UNIVERSE = 137
GRAD_MIN_FUNDS = EXPECTED_UNIVERSE // 2 + 1  # 69 of 137 (spec section 11)

TARGETS = {"T5c-R1": 1, "T5c-R5": 5}
HORIZON_OF = dict(TARGETS)

NUMERIC_FEATURES = [
    "ret1_bp", "ret_chg5_bp", "ret_chg21_bp", "ret_med21_bp",
    "ret_med252_bp", "ret_mad252_bp", "ret5cum_bp", "absret1_bp",
    "absret21_bp", "logvol", "logvol_chg5", "vix_level", "tnx_chg5",
    "spy_ret5_bp",
]
MODELS = ["persistence", "trailing_median", "zero", "ridge", "gbm"]
LEARNED = ["ridge", "gbm"]
COMPARATORS = ["persistence", "trailing_median", "zero"]

# Columns compared by the entry gate's hand checks and truncation
# check (labels are excluded from the truncation check only, because
# a label's forward window crosses the truncation date by
# construction; labels are included in the hand checks).
RETURN_AND_FEATURE_COLS = ["ret_bp"] + NUMERIC_FEATURES + ["ret5cum_med252_bp"]


# --- Robust writes (host write fault: bounded retries with verification) -------

def _retry(fn, attempts: int = 25, what: str = "write"):
    last = None
    for k in range(attempts):
        try:
            return fn()
        except (PermissionError, OSError) as exc:  # host EPERM fault
            last = exc
            time.sleep(random.uniform(0.5, 3.0) + 0.25 * k)
    raise RuntimeError(f"{what} failed after {attempts} attempts: {last}")


def robust_write_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")

    def _once():
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        got = path.read_bytes()
        if got != data:
            raise OSError(f"read-back mismatch for {path}")
    _retry(_once, what=f"write {path}")


def robust_write_text(path: Path, text: str):
    robust_write_bytes(path, text.encode("utf-8"))


def robust_write_json(path: Path, obj) -> str:
    text = json.dumps(obj, indent=1, default=str)
    robust_write_text(path, text)
    return text


def robust_write_parquet(path: Path, frame: pd.DataFrame, expected_rows: int):
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")

    def _once():
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        meta = pq.ParquetFile(path).metadata
        if meta.num_rows != expected_rows:
            raise OSError(f"read-back row mismatch for {path}: "
                          f"{meta.num_rows} != {expected_rows}")
    _retry(_once, what=f"write {path}")


# --- Freeze records --------------------------------------------------------------

def load_appendix() -> pd.DataFrame:
    return pd.read_csv(APPENDIX_PATH)


def appendix_funds(app: pd.DataFrame | None = None) -> list[str]:
    if app is None:
        app = load_appendix()
    return list(app.loc[app["series_kind"] == "FUND", "ticker"])


def eligible_funds(app: pd.DataFrame | None = None) -> list[str]:
    """Universe by the section 4 rule applied to the freeze audit
    appendix counts at the frozen panel end (399 rows, next-day)."""
    if app is None:
        app = load_appendix()
    funds = app[app["series_kind"] == "FUND"]
    return list(funds.loc[funds["n_rows_at_panel_end_2026-11-03"] >= MIN_ROWS_R1,
                          "ticker"])


def eligible_funds_r5(app: pd.DataFrame | None = None) -> list[str]:
    if app is None:
        app = load_appendix()
    funds = app[app["series_kind"] == "FUND"]
    return list(funds.loc[funds["n_rows_at_panel_end_2026-11-03"] >= MIN_ROWS_R5,
                          "ticker"])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_spec_receipt() -> dict:
    """Section 17 receipt procedure: replace the recorded SHA-256
    field by 64 zero characters and the byte-count field by 000000,
    compute the SHA-256 of the result, and compare with the recorded
    hash; confirm the total byte length."""
    raw = SPEC_PATH.read_bytes()
    text = raw.decode("utf-8")
    n_hash = text.count(SPEC_RECEIPT_SHA256)
    n_bytes_field = text.count("`026948`")
    replaced = text.replace(SPEC_RECEIPT_SHA256, "0" * 64)
    replaced = replaced.replace("`026948`", "`000000`")
    digest = hashlib.sha256(replaced.encode("utf-8")).hexdigest()
    return {
        "file_bytes": len(raw),
        "expected_bytes": SPEC_RECEIPT_BYTES,
        "bytes_match": len(raw) == SPEC_RECEIPT_BYTES,
        "hash_field_occurrences": n_hash,
        "byte_field_occurrences": n_bytes_field,
        "recomputed_sha256": digest,
        "expected_sha256": SPEC_RECEIPT_SHA256,
        "hash_match": digest == SPEC_RECEIPT_SHA256,
    }


def input_hashes(app: pd.DataFrame | None = None) -> dict[str, str]:
    """Section 10.1 item 4: SHA-256 of every input file — the 145
    fund cache files (SPY among them, in its dual role as a universe
    fund and as the market-state SPY series), the ^VIX and ^TNX
    market-state cache files, and the frozen specification file."""
    if app is None:
        app = load_appendix()
    out: dict[str, str] = {}
    for t in appendix_funds(app):
        out[f"data/cache/{t}.parquet"] = sha256_file(PRICE_CACHE / f"{t}.parquet")
    out["data/cache/^VIX.parquet"] = sha256_file(PRICE_CACHE / "^VIX.parquet")
    out["data/cache/^TNX.parquet"] = sha256_file(PRICE_CACHE / "^TNX.parquet")
    out[str(SPEC_PATH)] = sha256_file(SPEC_PATH)
    return out


# --- Raw inputs --------------------------------------------------------------------

def load_fund_raw(ticker: str, end: pd.Timestamp | None = PANEL_END) -> pd.DataFrame:
    """One fund's cached rows (section 3): served columns only (the
    dividends column is never carried), sorted by date, restricted
    to dates on or before ``end`` (no restriction when ``end`` is
    None)."""
    df = pd.read_parquet(PRICE_CACHE / f"{ticker}.parquet")
    df = df.reset_index().rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)
    if end is not None:
        df = df.loc[df["date"] <= end].reset_index(drop=True)
    return df[["date", "open", "high", "low", "close", "adjclose",
               "volume"]].copy()


def load_market_series(end: pd.Timestamp | None = PANEL_END) -> dict[str, dict]:
    """Market-state series of section 6 features 12-14, each computed
    on its own positions and restricted to dates on or before
    ``end``: ^VIX close, ^TNX close and its own-position 5-change,
    SPY served close and its own-position 5-trading-day return in
    basis points."""
    out: dict[str, dict] = {}
    for name in ["^VIX", "^TNX", "SPY"]:
        df = pd.read_parquet(PRICE_CACHE / f"{name}.parquet")
        df = df.reset_index().rename(columns={"index": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df[["date", "close"]].dropna().sort_values("date").reset_index(drop=True)
        if end is not None:
            df = df.loc[df["date"] <= end].reset_index(drop=True)
        out[name] = {"dates": df["date"].to_numpy(dtype="datetime64[ns]"),
                     "close": df["close"].to_numpy(dtype=float)}
    tnx = out["^TNX"]["close"]
    chg5 = np.full(len(tnx), np.nan)
    if len(tnx) > 5:
        chg5[5:] = tnx[5:] - tnx[:-5]
    out["^TNX"]["chg5"] = chg5
    spy = out["SPY"]["close"]
    ret5 = np.full(len(spy), np.nan)
    if len(spy) > 5:
        ret5[5:] = (spy[5:] / spy[:-5] - 1.0) * 10_000.0
    out["SPY"]["ret5_bp"] = ret5
    return out


def _asof_idx(market_dates: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Index of the latest market date on or before each row date
    (-1 where none exists)."""
    return np.searchsorted(market_dates, dates, side="right") - 1


# --- Section 6 constructions (per fund) ---------------------------------------------

def _shift(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    if len(a) > k:
        out[k:] = a[:-k]
    return out


def _rolling_mad(s: np.ndarray, center: np.ndarray, window: int,
                 min_defined: int) -> np.ndarray:
    """Feature 6: median of |s_t - center_i| over the defined s_t in
    positions i-window+1..i (center_i = feature 5's value at i),
    requiring at least min_defined defined values. The Target 5b
    construction, transferred."""
    n = len(s)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    for i in range(min(window - 1, n)):
        if not np.isfinite(center[i]):
            continue
        w = s[:i + 1]
        w = w[np.isfinite(w)]
        if len(w) < min_defined:
            continue
        out[i] = float(np.median(np.abs(w - center[i])))
    if n >= window:
        from numpy.lib.stride_tricks import sliding_window_view
        W = sliding_window_view(s, window)
        ctr = center[window - 1:]
        finite = np.isfinite(W)
        counts = finite.sum(axis=1)
        D = np.abs(W - ctr[:, None])
        D = np.where(finite, D, np.inf)
        D.sort(axis=1)
        k = counts
        mid_hi = k // 2
        mid_lo = np.maximum((k - 1) // 2, 0)
        rows = np.arange(len(D))
        med = 0.5 * (D[rows, mid_lo] + D[rows, mid_hi])
        ok = (k >= min_defined) & np.isfinite(ctr)
        out[window - 1:][ok] = med[ok]
    return out


def build_fund_panel(raw: pd.DataFrame, market: dict[str, dict]) -> pd.DataFrame:
    """The section 6 features, both section 2 labels, and the
    section 9 baseline inputs for one fund. ``raw`` carries
    date/open/high/low/close/adjclose/volume in panel order."""
    f = raw.reset_index(drop=True)
    n = len(f)
    dates = f["date"].to_numpy(dtype="datetime64[ns]")
    adj = f["adjclose"].to_numpy(dtype=float)
    close = f["close"].to_numpy(dtype=float)
    vol = f["volume"].to_numpy(dtype=float)

    # Own-history return series r, in basis points (section 2 basis).
    r = np.full(n, np.nan)
    if n > 1:
        r[1:] = (adj[1:] / adj[:-1] - 1.0) * 10_000.0
    rs = pd.Series(r)
    out = pd.DataFrame({
        "ticker": f["ticker"] if "ticker" in f else None,
        "date": f["date"],
        "pos": np.arange(n),
        "open": f["open"].to_numpy(dtype=float),
        "high": f["high"].to_numpy(dtype=float),
        "low": f["low"].to_numpy(dtype=float),
        "close": close,
        "adjclose": adj,
        "volume": vol,
        "ret_bp": r,
    })
    out["ret1_bp"] = r
    out["ret_chg5_bp"] = r - _shift(r, 5)
    out["ret_chg21_bp"] = r - _shift(r, 21)
    out["ret_med21_bp"] = rs.rolling(21, min_periods=10).median().to_numpy()
    med252 = rs.rolling(252, min_periods=20).median().to_numpy()
    out["ret_med252_bp"] = med252
    out["ret_mad252_bp"] = _rolling_mad(r, med252, 252, 20)
    ret5cum = np.full(n, np.nan)
    if n > 5:
        ret5cum[5:] = (adj[5:] / adj[:-5] - 1.0) * 10_000.0
    out["ret5cum_bp"] = ret5cum
    out["absret1_bp"] = np.abs(r)
    out["absret21_bp"] = (pd.Series(np.abs(r)).rolling(21, min_periods=10)
                          .mean().to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        dv = close * vol
        logvol = np.where(dv > 0, np.log(np.where(dv > 0, dv, 1.0)), np.nan)
    out["logvol"] = logvol
    out["logvol_chg5"] = logvol - _shift(logvol, 5)
    # Market-state features by as-of join (latest market date on or
    # before the row's date; the series' own precomputed values).
    vix_idx = _asof_idx(market["^VIX"]["dates"], dates)
    out["vix_level"] = np.where(vix_idx >= 0,
                                market["^VIX"]["close"][np.maximum(vix_idx, 0)],
                                np.nan)
    tnx_idx = _asof_idx(market["^TNX"]["dates"], dates)
    out["tnx_chg5"] = np.where(tnx_idx >= 0,
                               market["^TNX"]["chg5"][np.maximum(tnx_idx, 0)],
                               np.nan)
    spy_idx = _asof_idx(market["SPY"]["dates"], dates)
    out["spy_ret5_bp"] = np.where(spy_idx >= 0,
                                  market["SPY"]["ret5_bp"][np.maximum(spy_idx, 0)],
                                  np.nan)
    # Baseline input for the 5-day trailing median (section 9):
    # median of the fund's completed 5-trading-day cumulative
    # returns ending at positions i-251..i, minimum 20 values.
    out["ret5cum_med252_bp"] = (pd.Series(ret5cum)
                                .rolling(252, min_periods=20).median().to_numpy())
    # Labels and label end dates (section 2).
    for h in (1, 5):
        lab = np.full(n, np.nan)
        if n > h:
            lab[:-h] = (adj[h:] / adj[:-h] - 1.0) * 10_000.0
        out[f"y_{h}"] = lab
        end_dates = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
        if n > h:
            end_dates[:-h] = dates[h:]
        out[f"label_end_date_{h}"] = pd.to_datetime(end_dates)
        end_pos = np.full(n, -1, dtype=int)
        if n > h:
            end_pos[:-h] = np.arange(h, n)
        out[f"label_end_pos_{h}"] = end_pos
    wd = f["date"].dt.weekday
    assert set(wd.unique()) <= {0, 1, 2, 3, 4}, "weekend panel date"
    for k, name in [(1, "dow_tue"), (2, "dow_wed"), (3, "dow_thu"), (4, "dow_fri")]:
        out[name] = (wd == k).astype(float)
    return out


# --- Section 7 sample -----------------------------------------------------------------

def build_sample(panel: pd.DataFrame, target: str,
                 funds: list[str]) -> pd.DataFrame:
    """Section 7 sample for one target: every numeric feature
    non-NaN and the label computable (plus, for the 5-day target,
    the 5-day trailing-median baseline input non-NaN, so the
    comparison sample is identical across models and baselines).
    Sorted by (date, ticker), with fund and day-of-week indices."""
    horizon = HORIZON_OF[target]
    fund_index = {t: i for i, t in enumerate(funds)}
    d = panel[panel["ticker"].isin(funds)].copy()
    keep = d[NUMERIC_FEATURES].notna().all(axis=1) & d[f"y_{horizon}"].notna()
    if horizon == 5:
        keep = keep & d["ret5cum_med252_bp"].notna()
    d = d[keep].copy()
    d["y"] = d[f"y_{horizon}"].astype(float)
    d["label_end_date"] = d[f"label_end_date_{horizon}"]
    d["label_end_pos"] = d[f"label_end_pos_{horizon}"]
    d["fund_idx"] = d["ticker"].map(fund_index).astype(int)
    d["dow_idx"] = d["date"].dt.weekday.astype(int)
    d = d.sort_values(["date", "ticker"]).reset_index(drop=True)
    assert d[NUMERIC_FEATURES + ["y"]].notna().all().all()
    return d


# --- Section 8 pooled folds -------------------------------------------------------------

def make_pooled_folds(sample: pd.DataFrame, horizon: int):
    """Expanding walk-forward folds counted in pooled sample dates
    (section 8, the Target 5b section 9 construction transferred):
    test dates D[252 + 63k ... 252 + 63(k+1) - 1]; training
    candidates dated strictly before the first test date; purge
    drops a training row whose label end date is on or after the
    first test date; embargo keeps a training row only if its
    feature date's position in D is at most (first test date
    position) - 6. Assertions raise (section 10.2)."""
    dates_sorted = np.sort(sample["date"].unique())
    D = pd.DatetimeIndex(dates_sorted)
    pos_in_D = {d: i for i, d in enumerate(D)}
    sample_pos_in_D = sample["date"].map(pos_in_D).to_numpy()
    sample_dates = sample["date"].to_numpy()
    label_end = sample["label_end_date"].to_numpy()
    folds = []
    k = 0
    while INITIAL_TRAIN + TEST_BLOCK * k < len(D):
        lo = INITIAL_TRAIN + TEST_BLOCK * k
        hi = min(lo + TEST_BLOCK, len(D))
        test_dates = D[lo:hi]
        first_test = test_dates[0]
        test_idx = np.where(sample["date"].isin(test_dates).to_numpy())[0]
        cand = np.where(sample_dates < np.datetime64(first_test))[0]
        keep_embargo = sample_pos_in_D[cand] <= lo - 6
        keep_purge = label_end[cand] < np.datetime64(first_test)
        train_idx = cand[keep_embargo & keep_purge]
        n_cand = int(len(cand))
        n_after_embargo = int(keep_embargo.sum())
        n_train = int(len(train_idx))
        assert n_train > 0, f"fold {k}: empty training set"
        assert sample_dates[train_idx].max() < sample_dates[test_idx].min()
        gap = lo - sample_pos_in_D[train_idx].max()
        assert gap > EMBARGO_DATES, f"fold {k}: embargo gap {gap} <= 5"
        assert (label_end[train_idx] < np.datetime64(first_test)).all(), \
            f"fold {k}: purge violated"
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        # Knowledge-date assertion (section 10.2): every test row's
        # label window ends strictly after its feature date.
        assert (label_end[test_idx] > sample_dates[test_idx]).all(), \
            f"fold {k}: label end date not after feature date"
        folds.append({
            "fold_id": k,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "meta": {
                "fold": int(k),
                "test_start": str(pd.Timestamp(test_dates[0]).date()),
                "test_end": str(pd.Timestamp(test_dates[-1]).date()),
                "n_train": n_train,
                "n_test": int(len(test_idx)),
                "n_candidates": n_cand,
                "n_dropped_embargo": int(n_cand - n_after_embargo),
                "n_dropped_purge_only": int(n_after_embargo - n_train),
                "n_funds_in_test": int(sample.iloc[test_idx]["ticker"].nunique()),
            },
        })
        k += 1
    # Fold-disjointness across the whole axis (section 10.2).
    all_test = np.concatenate([f["test_idx"] for f in folds]) if folds else \
        np.array([], dtype=int)
    assert len(all_test) == len(np.unique(all_test)), "test folds not disjoint"
    return folds


# --- Section 9 model ladder ---------------------------------------------------------------

def ridge_fit_predict_pipeline(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray):
    """Ridge per section 9: the 14 numeric features are standardised
    inside the pipeline (statistics from the training fold only);
    fund and day-of-week indicators are appended unscaled; alpha is
    selected from the frozen grid of ``engine/ml_models.py`` by
    inner TimeSeriesSplit(3) on the training fold only, then refit
    on the full training fold. This is the reference evaluation of
    the frozen procedure (the Target 5b ``ridge_fit_predict_t5b``
    construction, transferred)."""
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


def gbm_fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray):
    """Gradient boosting per section 9: the frozen construction of
    ``engine/ml_models.py``, imported unmodified."""
    return M.gbm_reg_fit_predict(Xtr, ytr, Xte)


# --- Ridge via Gram assembly (identical procedure, efficient evaluation) ------------------
#
# The frozen ridge procedure (section 9) is: standardise the 14
# numeric features with statistics from the training subset in
# question, append fund and day-of-week indicators unscaled, fit
# sklearn Ridge (intercept by centering, penalty alpha * ||w||^2 on
# all coefficients), select alpha from the frozen grid by inner
# TimeSeriesSplit(3) cross-validation (lowest mean CV mean squared
# error), refit on the full training fold. Because the design has
# 155 columns but the indicator block is one-hot, the normal
# equations of every candidate fit are assembled exactly from group
# aggregates (Gram matrix and X'y), which evaluates the identical
# procedure without repeatedly copying the pooled design matrix.
# This is the Target 5b evaluation, transferred unchanged.

def _ridge_gram(Xn, fund_idx, dow_idx, y, rows, n_funds):
    Xn_s = Xn[rows]
    mu = Xn_s.mean(axis=0)
    sd = Xn_s.std(axis=0)  # population std, as sklearn StandardScaler
    sd = np.where(sd == 0.0, 1.0, sd)
    Z = (Xn_s - mu) / sd
    fi = fund_idx[rows]
    di = dow_idx[rows]
    yy = y[rows]
    n = len(rows)
    n_num = Xn.shape[1]
    p = n_num + n_funds + 4
    G = np.zeros((p, p))
    xty = np.zeros(p)
    G[:n_num, :n_num] = Z.T @ Z
    xty[:n_num] = Z.T @ yy
    ZF = np.zeros((n_funds, n_num))
    np.add.at(ZF, fi, Z)
    G[:n_num, n_num:n_num + n_funds] = ZF.T
    G[n_num:n_num + n_funds, :n_num] = ZF
    xty[n_num:n_num + n_funds] = np.bincount(fi, weights=yy, minlength=n_funds)
    ZD = np.zeros((4, n_num))
    m_dow = di > 0
    np.add.at(ZD, di[m_dow] - 1, Z[m_dow])
    G[:n_num, n_num + n_funds:] = ZD.T
    G[n_num + n_funds:, :n_num] = ZD
    xty[n_num + n_funds:] = np.bincount(di[m_dow] - 1, weights=yy[m_dow],
                                        minlength=4)
    fc = np.bincount(fi, minlength=n_funds).astype(float)
    G[n_num:n_num + n_funds, n_num:n_num + n_funds] = np.diag(fc)
    FD = np.zeros((n_funds, 4))
    np.add.at(FD, (fi[m_dow], di[m_dow] - 1), 1.0)
    G[n_num:n_num + n_funds, n_num + n_funds:] = FD
    G[n_num + n_funds:, n_num:n_num + n_funds] = FD.T
    dc = np.bincount(di[m_dow] - 1, minlength=4).astype(float)
    G[n_num + n_funds:, n_num + n_funds:] = np.diag(dc)
    mvec = np.zeros(p)
    mvec[:n_num] = Z.mean(axis=0)
    mvec[n_num:n_num + n_funds] = fc / n
    mvec[n_num + n_funds:] = dc / n
    return {"G": G, "xty": xty, "mvec": mvec, "ybar": float(yy.mean()),
            "mu": mu, "sd": sd, "n": n}


def _ridge_solve(gram, alpha):
    n = gram["n"]
    Gc = gram["G"] - n * np.outer(gram["mvec"], gram["mvec"])
    rhs = gram["xty"] - n * gram["mvec"] * gram["ybar"]
    Gc[np.diag_indices_from(Gc)] += alpha
    return np.linalg.solve(Gc, rhs)


def _ridge_predict(gram, w, Xn, fund_idx, dow_idx, rows, n_funds):
    Z = (Xn[rows] - gram["mu"]) / gram["sd"]
    n_num = Xn.shape[1]
    pred = Z @ w[:n_num] + w[n_num + fund_idx[rows]]
    di = dow_idx[rows]
    m_dow = di > 0
    pred = pred + gram["ybar"] - float(gram["mvec"] @ w)
    dow_w = np.zeros(len(rows))
    dow_w[m_dow] = w[n_num + n_funds + di[m_dow] - 1]
    return pred + dow_w


def ridge_fit_predict_gram(Xn, fund_idx, dow_idx, y, train_rows, test_rows,
                           n_funds):
    """The frozen ridge procedure evaluated by Gram assembly (see
    the note above). train_rows / test_rows are positional indices
    into the sample arrays, in sample order."""
    Xtr_n = Xn[train_rows]
    splits = list(TimeSeriesSplit(3).split(Xtr_n))
    mses = {a: [] for a in M.RIDGE_ALPHAS}
    for sub_tr_pos, sub_val_pos in splits:
        sub_tr = train_rows[sub_tr_pos]
        sub_val = train_rows[sub_val_pos]
        gram = _ridge_gram(Xn, fund_idx, dow_idx, y, sub_tr, n_funds)
        for a in M.RIDGE_ALPHAS:
            w = _ridge_solve(gram, a)
            pv = _ridge_predict(gram, w, Xn, fund_idx, dow_idx, sub_val,
                                n_funds)
            mses[a].append(float(np.mean((y[sub_val] - pv) ** 2)))
    best_alpha = min(M.RIDGE_ALPHAS, key=lambda a: float(np.mean(mses[a])))
    gram_full = _ridge_gram(Xn, fund_idx, dow_idx, y, train_rows, n_funds)
    w = _ridge_solve(gram_full, best_alpha)
    pred = _ridge_predict(gram_full, w, Xn, fund_idx, dow_idx, test_rows,
                          n_funds)
    return pred, {"alpha": float(best_alpha)}


def design_matrix(Xn, fund_idx, dow_cols, rows, n_funds) -> np.ndarray:
    """The full 155-column design for the given sample rows
    (numeric block, fund indicators, day-of-week indicators), for
    the gradient-boosting fit, which consumes the design
    unstandardised."""
    Xn_r = Xn[rows]
    X = np.zeros((len(rows), Xn.shape[1] + n_funds + 4), dtype=float)
    X[:, :Xn.shape[1]] = Xn_r
    X[np.arange(len(rows)), Xn.shape[1] + fund_idx[rows]] = 1.0
    X[:, Xn.shape[1] + n_funds:] = dow_cols[rows]
    return X
