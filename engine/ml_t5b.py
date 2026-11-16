"""ML Target 5b — per-fund ETF trading-cost prediction (multi-measure).

Pre-registered spec: reports/ml_target5b_experiment_spec_2026-10-28.md
(frozen 2026-10-28, before any Target 5b fit). This module implements
ONLY what that spec freezes: section 3 panel construction from the
Yahoo cache (served OHLCV only, panel end 2026-10-28), the section 6
measure constructions M1-M12 and M16-M21 (M13 frozen NOT BUILT in spec
section 6.13; M14 and M15 conditionally built under spec sections
6.14/6.15 — see the driver for their admission records), the section 7
feature list (14 numeric features + fund indicators + 4 day-of-week
indicators), the section 9 pooled walk-forward protocol (252 initial
training dates / 63-date test blocks on the pooled sample-date axis,
expanding window, purge by label end date + 5 pooled-sample-date
embargo for both horizons, seed 13), the section 10 model ladder
(persistence and trailing-median baselines, ridge with the program's
frozen alpha policy, gradient-boosting regressor with the program's
frozen hyperparameters), and the section 12 metrics and graduation
bar.

Offline evaluation only — nothing here is issued or scheduled, and no
existing module's defaults are modified.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from engine import ml_models as M

ROOT = Path(__file__).resolve().parent.parent
PRICE_CACHE = ROOT / "data" / "cache"
T5B_DIR = ROOT / "data" / "t5b_cost"
PANEL_PATH = T5B_DIR / "panel_t5b.parquet"
PRED_PATH = T5B_DIR / "wf_predictions_t5b.parquet"
ELIGIBILITY_PATH = ROOT / "reports" / "ml_target5a_eligibility_2026-10-28.csv"
SPEC_PATH = ROOT / "reports" / "ml_target5b_experiment_spec_2026-10-28.md"
AUDIT_PATH = ROOT / "reports" / "ml_target5b_measure_audit_2026-10-28.md"
DATASET_CARD_PATH = (Path.home() / "workspace" / "goals" /
                     "market-knowledge-agent-ml-prediction-engine" / "files" /
                     "ml_target5b_dataset_card_2026-10-28.md")

PANEL_END = pd.Timestamp("2026-10-28")
SEED = 13
INITIAL_TRAIN = 252
TEST_BLOCK = 63
EMBARGO_DATES = 5

# Spec section 4.2: the 145 Class 2 panel tickers are those listed in
# the Target 5a eligibility file; the eight funds below are excluded
# on history alone.
EXCLUDED_HISTORY = ["DRAM", "JA", "RAAR", "RAAY", "CHPX", "PCR", "AIPO", "RAAA"]
# Spec section 4.2: M16 universe excludes these 12 further funds.
M16_EXCLUDED = ["AIBD", "AIBU", "AIPI", "AGIX", "AIS", "BAI", "BCLO", "CLOB",
                "PCMM", "PRIV", "SMHX", "XAIX"]
# Spec section 4.2 (record only; M13 is NOT BUILT, spec section 6.13).
M13_EXCLUDED = ["BCLO", "PRIV"]

MEASURES_BUILT = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
                  "M10", "M11", "M12", "M14", "M16", "M17", "M18", "M19",
                  "M20", "M21"]
MEASURE_NAMES = {
    "M1": "Amihud ILLIQ",
    "M2": "log(1 + ILLIQ)",
    "M3": "Amihud share-volume form",
    "M4": "Amihud intraday component",
    "M5": "Amihud overnight component (proxy attribution)",
    "M6": "Dollar volume",
    "M7": "Log dollar volume",
    "M8": "Share volume",
    "M9": "Relative high-low range",
    "M10": "Roll implied spread",
    "M11": "Corwin-Schultz high-low spread (21-day mean)",
    "M12": "Abdi-Ranaldo CHLO spread (21-day mean)",
    "M14": "EDGE spread estimator (21-trading-day window)",
    "M16": "Lesmond-Ogden-Trzcinka LOT (252-day)",
    "M17": "Pastor-Stambaugh gamma (raw-return form)",
    "M18": "Zero-return proportion (21-day)",
    "M19": "Zero-volume proportion (21-day)",
    "M20": "Amivest ratio (21-day)",
    "M21": "Amihud 21-day trailing mean",
}

NUMERIC_FEATURES = [
    "meas_level", "meas_chg5", "meas_chg21", "meas_med21", "meas_med252",
    "meas_mad252", "ret1_bp", "ret5_bp", "absret1_bp", "logvol",
    "logvol_chg5", "vix_level", "tnx_chg5", "spy_ret5_bp",
]
HORIZONS = [1, 5]
MODELS = ["persistence", "trailing_median", "ridge", "gbm"]
LEARNED = ["ridge", "gbm"]


def panel_tickers() -> list[str]:
    elig = pd.read_csv(ELIGIBILITY_PATH)
    return list(elig["ticker"])


def eligible_funds() -> list[str]:
    return [t for t in panel_tickers() if t not in EXCLUDED_HISTORY]


def universe_for(measure: str) -> list[str]:
    elig = eligible_funds()
    if measure == "M16":
        return [t for t in elig if t not in M16_EXCLUDED]
    return elig


def grad_min_funds(measure: str) -> int:
    n = len(universe_for(measure))
    return n // 2 + 1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_hashes() -> dict[str, str]:
    """Spec section 11.1 item 5: SHA-256 of every input file used."""
    out: dict[str, str] = {}
    for t in eligible_funds():
        out[f"data/cache/{t}.parquet"] = sha256_file(PRICE_CACHE / f"{t}.parquet")
    out["data/cache/^VIX.parquet"] = sha256_file(PRICE_CACHE / "^VIX.parquet")
    out["data/cache/^TNX.parquet"] = sha256_file(PRICE_CACHE / "^TNX.parquet")
    out["reports/ml_target5b_measure_audit_2026-10-28.md"] = sha256_file(AUDIT_PATH)
    out["ml_target5b_dataset_card_2026-10-28.md"] = sha256_file(DATASET_CARD_PATH)
    out["reports/ml_target5a_eligibility_2026-10-28.csv"] = sha256_file(ELIGIBILITY_PATH)
    out["reports/ml_target5b_experiment_spec_2026-10-28.md"] = sha256_file(SPEC_PATH)
    return out


# --- Section 3 panel ------------------------------------------------------------

def load_fund_raw(ticker: str, end: pd.Timestamp | None = PANEL_END) -> pd.DataFrame:
    """One fund's cached panel (spec section 3.1): served OHLCV only,
    sorted by date, restricted to dates on or before the panel end."""
    df = pd.read_parquet(PRICE_CACHE / f"{ticker}.parquet")
    df = df.reset_index().rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)
    if end is not None:
        df = df.loc[df["date"] <= end].reset_index(drop=True)
    return df[["date", "open", "high", "low", "close", "volume"]].copy()


def load_market_series() -> dict[str, pd.DataFrame]:
    """Cache market series of spec section 7.2 plus the SPY return
    series used as the M16 market leg (spec section 6.16)."""
    out = {}
    for name, col in [("^VIX", "vix_close"), ("^TNX", "tnx_close"), ("SPY", "spy_close")]:
        df = pd.read_parquet(PRICE_CACHE / f"{name}.parquet")
        df = df.reset_index().rename(columns={"index": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        out[name] = (df[["date", "close"]].dropna().sort_values("date")
                     .reset_index(drop=True).rename(columns={"close": col}))
    spy = out["SPY"].copy()
    spy["spy_ret"] = spy["spy_close"] / spy["spy_close"].shift(1) - 1.0
    spy["spy_ret5_bp"] = (spy["spy_close"] / spy["spy_close"].shift(5) - 1.0) * 10_000.0
    out["SPY"] = spy
    tnx = out["^TNX"].copy()
    tnx["tnx_chg5"] = tnx["tnx_close"] - tnx["tnx_close"].shift(5)
    out["^TNX"] = tnx
    return out


# --- Section 6.14 EDGE estimator (as amended by Amendment A2, 2026-10-29) --------
#
# EDGE (Ardia, Guidotti & Kroencke 2024, Journal of Financial Economics
# 161, 103916) is a sample-level estimator. Amendment A2 re-freezes the
# M14 series construction: the EDGE estimate computed over the trailing
# 21 trading days of the fund's own OHLC sample, dated at the window's
# last trading day. The estimator below is transcribed from the
# authors' published pseudocode (repository eguidotti/bidask,
# pseudocode directory) and passes the frozen unit test against the
# authors' published worked example (see the driver's entry gate).

def edge_estimator(open_, high, low, close, sign=False):
    """EDGE estimator transcribed from the authors' published
    pseudocode (Ardia, Guidotti & Kroencke 2024; repository
    eguidotti/bidask, pseudocode directory, accessed 2026-10-28).
    Returns (estimate, intermediates dict)."""
    import math as _math
    o = np.log(np.asarray(open_, float))
    h = np.log(np.asarray(high, float))
    l = np.log(np.asarray(low, float))
    c = np.log(np.asarray(close, float))
    if len(o) < 3:
        return np.nan, {}
    m = (h + l) / 2.0
    lag = lambda x: np.concatenate([[np.nan], x[:-1]])  # noqa: E731
    h1, l1, c1, m1 = lag(h), lag(l), lag(c), lag(m)
    r1 = m - o
    r2 = o - m1
    r3 = m - c1
    r4 = c1 - m1
    r5 = o - c1
    with np.errstate(invalid="ignore", divide="ignore"):
        tau = np.where(np.isnan(h) | np.isnan(l) | np.isnan(c1), np.nan,
                       ((h != l) | (l != c1)).astype(float))
        ind = lambda cond, *arrs: np.where(  # noqa: E731
            np.isnan(tau) | np.any(np.isnan(np.column_stack(arrs)), axis=1)
            if arrs else np.isnan(tau), np.nan, cond.astype(float))
        po1 = ind((tau == 1) & (o != h), o, h)
        po2 = ind((tau == 1) & (o != l), o, l)
        pc1 = ind((tau == 1) & (c1 != h1), c1, h1)
        pc2 = ind((tau == 1) & (c1 != l1), c1, l1)
        pt = float(np.nanmean(tau))
        po = float(np.nanmean(po1) + np.nanmean(po2))
        pc = float(np.nanmean(pc1) + np.nanmean(pc2))
        if np.nansum(tau) < 2 or po == 0 or pc == 0 or pt == 0:
            return np.nan, {}
        d1 = r1 - np.nanmean(r1) / pt * tau
        d3 = r3 - np.nanmean(r3) / pt * tau
        d5 = r5 - np.nanmean(r5) / pt * tau
        x1 = -4.0 / po * d1 * r2 + -4.0 / pc * d3 * r4
        x2 = -4.0 / po * d1 * r5 + -4.0 / pc * d5 * r4
        e1 = float(np.nanmean(x1))
        e2 = float(np.nanmean(x2))
        v1 = float(np.nanmean(x1 * x1) - e1 * e1)
        v2 = float(np.nanmean(x2 * x2) - e2 * e2)
    vt = v1 + v2
    s2 = (v2 * e1 + v1 * e2) / vt if vt > 0 else (e1 + e2) / 2.0
    s = _math.sqrt(abs(s2))
    if sign and s2 < 0:
        s = -s
    return s, {"pt": pt, "po": po, "pc": pc, "e1": e1, "e2": e2,
               "v1": v1, "v2": v2, "s2": float(s2)}


def compute_edge_series(O, H, L, C) -> np.ndarray:
    """M14 series for one fund (Amendment A2): EDGE estimate over the
    trailing 21 trading days ending at each position j >= 20, dated at
    j; NaN where the window's EDGE estimate is undefined."""
    n = len(C)
    out = np.full(n, np.nan)
    for j in range(20, n):
        est, _ = edge_estimator(O[j - 20:j + 1], H[j - 20:j + 1],
                                L[j - 20:j + 1], C[j - 20:j + 1])
        out[j] = est
    return out


# --- Section 6.16 LOT estimation --------------------------------------------------

_LOT_BOUNDS = [(-0.01, 0.01), (-3.0, 3.0), (1e-6, 0.2),
               (-0.05, -1e-8), (1e-8, 0.05)]
_LOG_SQRT_2PI = 0.9189385332046727


def lot_loglik(params: np.ndarray, r: np.ndarray, rm: np.ndarray) -> float:
    """Spec section 6.16 log-likelihood of one window at the given
    parameters (a, b, sigma, t1, t2)."""
    a, b, sig, t1, t2 = params
    mu = a + b * rm
    ll = 0.0
    pos = r > 0
    neg = r < 0
    zer = ~(pos | neg)
    if pos.any():
        z = (r[pos] + t2 - mu[pos]) / sig
        ll += float(np.sum(-0.5 * z * z - np.log(sig) - _LOG_SQRT_2PI))
    if neg.any():
        z = (r[neg] + t1 - mu[neg]) / sig
        ll += float(np.sum(-0.5 * z * z - np.log(sig) - _LOG_SQRT_2PI))
    if zer.any():
        muz = mu[zer]
        d = ndtr((t2 - muz) / sig) - ndtr((t1 - muz) / sig)
        ll += float(np.sum(np.log(np.maximum(d, 1e-300))))
    return ll


def _lot_nll_grad(p, r, rm):
    """Negative log-likelihood and its analytic gradient (verified
    against central finite differences in the build's benchmark; the
    frozen estimator of spec section 6.16 is unchanged — the gradient
    is a computational device only)."""
    a, b, sig, t1, t2 = p
    mu = a + b * rm
    pos = r > 0
    neg = r < 0
    zer = ~(pos | neg)
    nll = 0.0
    g_mu = np.zeros_like(r)
    g_sig = 0.0
    g_t1 = 0.0
    g_t2 = 0.0
    if pos.any():
        z = (r[pos] + t2 - mu[pos]) / sig
        nll -= float(np.sum(-0.5 * z * z - np.log(sig) - _LOG_SQRT_2PI))
        g_mu[pos] = z / sig
        g_sig += float(np.sum((z * z - 1.0) / sig))
        g_t2 += float(np.sum(-z / sig))
    if neg.any():
        z = (r[neg] + t1 - mu[neg]) / sig
        nll -= float(np.sum(-0.5 * z * z - np.log(sig) - _LOG_SQRT_2PI))
        g_mu[neg] = z / sig
        g_sig += float(np.sum((z * z - 1.0) / sig))
        g_t1 += float(np.sum(-z / sig))
    if zer.any():
        muz = mu[zer]
        u2 = (t2 - muz) / sig
        u1 = (t1 - muz) / sig
        d = np.maximum(ndtr(u2) - ndtr(u1), 1e-300)
        nll -= float(np.sum(np.log(d)))
        ph2 = np.exp(-0.5 * u2 * u2) / (2.0 * np.sqrt(np.pi))
        ph1 = np.exp(-0.5 * u1 * u1) / (2.0 * np.sqrt(np.pi))
        # note: 1/sqrt(2*pi) = 1/(sqrt(2)*sqrt(pi)); use exact constant below
        ph2 = np.exp(-0.5 * u2 * u2) * 0.3989422804014327
        ph1 = np.exp(-0.5 * u1 * u1) * 0.3989422804014327
        g_mu[zer] = (-ph2 + ph1) / (sig * d)
        g_sig += float(np.sum((-u2 * ph2 + u1 * ph1) / (sig * d)))
        g_t2 += float(np.sum(ph2 / (sig * d)))
        g_t1 += float(np.sum(-ph1 / (sig * d)))
    grad_ll = np.array([g_mu.sum(), float((g_mu * rm).sum()), g_sig, g_t1, g_t2])
    return nll, -grad_ll


def _lot_nll(p, r, rm):
    """Value-only negative log-likelihood (spec section 6.16)."""
    a, b, sig, t1, t2 = p
    mu = a + b * rm
    ll = 0.0
    pos = r > 0
    neg = r < 0
    zer = ~(pos | neg)
    if pos.any():
        z = (r[pos] + t2 - mu[pos]) / sig
        ll += float(np.sum(-0.5 * z * z - np.log(sig) - _LOG_SQRT_2PI))
    if neg.any():
        z = (r[neg] + t1 - mu[neg]) / sig
        ll += float(np.sum(-0.5 * z * z - np.log(sig) - _LOG_SQRT_2PI))
    if zer.any():
        muz = mu[zer]
        d = ndtr((t2 - muz) / sig) - ndtr((t1 - muz) / sig)
        ll += float(np.sum(np.log(np.maximum(d, 1e-300))))
    return -ll


def lot_window_value(r: np.ndarray, rm: np.ndarray) -> tuple[float, bool]:
    """One frozen LOT window estimate (spec section 6.16): maximum
    likelihood with scipy.optimize.minimize, method L-BFGS-B, from
    the frozen starting values (a, b from the window's OLS regression
    of r on rm; sigma the regression's residual standard deviation;
    t1 = -0.005, t2 = 0.005) and the frozen bounds, exactly as the
    specification states them (scipy projects an out-of-bounds
    starting value into the bounds). Returns (t2 - t1, failed) where
    failed records optimiser failure or sigma on a bound (window
    undefined)."""
    X = np.column_stack([np.ones_like(rm), rm])
    beta, *_ = np.linalg.lstsq(X, r, rcond=None)
    resid = r - X @ beta
    sig0 = float(np.std(resid))
    if not np.isfinite(sig0):
        return np.nan, True
    x0 = np.array([float(beta[0]), float(beta[1]), sig0, -0.005, 0.005])
    out = minimize(_lot_nll, x0, args=(r, rm), method="L-BFGS-B",
                   bounds=_LOT_BOUNDS)
    if not out.success:
        return np.nan, True
    a, b, sig, t1, t2 = out.x
    if sig <= 1e-6 + 1e-12 or sig >= 0.2 - 1e-12:
        return np.nan, True
    return float(t2 - t1), False


def compute_lot_series(dates: pd.Series, closes: np.ndarray,
                       spy_ret_by_date: dict) -> tuple[np.ndarray, int]:
    """M16 series for one fund: 252-return rolling windows ending at
    each position j >= 252 (spec section 6.16). A window position
    whose date has no SPY return is dropped from the window; a window
    with fewer than 200 matched pairs is undefined. Returns the series
    and the count of failed windows (optimiser failure or sigma on a
    bound)."""
    n = len(closes)
    series = np.full(n, np.nan)
    failed = 0
    r = np.full(n, np.nan)
    r[1:] = closes[1:] / closes[:-1] - 1.0
    rm_all = np.array([spy_ret_by_date.get(d, np.nan) for d in dates])
    for j in range(252, n):
        # returns at positions j-251..j matched to SPY returns on the
        # same dates; the return at position t is dated t.
        idx = np.arange(j - 251, j + 1)
        rw = r[idx]
        mw = rm_all[idx]
        ok = np.isfinite(rw) & np.isfinite(mw)
        if ok.sum() < 200:
            continue
        val, fail = lot_window_value(rw[ok], mw[ok])
        if fail:
            failed += 1
        else:
            series[j] = val
    return series, failed


# --- Section 6 measure constructions (per fund) -----------------------------------

def compute_measures(fund: pd.DataFrame, spy_ret_by_date: dict | None,
                     with_lot: bool) -> pd.DataFrame:
    """All built measure series for one fund's panel, per spec section
    6. fund carries date/open/high/low/close/volume in panel order."""
    f = fund.reset_index(drop=True).copy()
    n = len(f)
    O = f["open"].to_numpy(dtype=float)
    H = f["high"].to_numpy(dtype=float)
    L = f["low"].to_numpy(dtype=float)
    C = f["close"].to_numpy(dtype=float)
    V = f["volume"].to_numpy(dtype=float)
    r = np.full(n, np.nan)
    r[1:] = C[1:] / C[:-1] - 1.0
    DV = C * V
    pos_vol = V > 0

    out = pd.DataFrame({"ticker": f["ticker"], "date": f["date"],
                        "pos": np.arange(n), "open": O, "high": H, "low": L,
                        "close": C, "volume": V})

    # M1-M5 (dollar-volume family; zero-volume days undefined).
    illiq = np.full(n, np.nan)
    m3 = np.full(n, np.nan)
    m4 = np.full(n, np.nan)
    m5 = np.full(n, np.nan)
    ok1 = pos_vol.copy()
    ok1[0] = False
    illiq[ok1] = np.abs(r[ok1]) / DV[ok1]
    m3[ok1] = np.abs(r[ok1]) / V[ok1]
    m4[ok1] = np.abs(C[ok1] / O[ok1] - 1.0) / DV[ok1]
    m5[ok1] = np.abs(O[ok1] / C[np.where(ok1)[0] - 1] - 1.0) / DV[ok1]
    out["M1"] = illiq
    out["M2"] = np.log1p(illiq)
    out["M3"] = m3
    out["M4"] = m4
    out["M5"] = m5
    # M6-M9.
    out["M6"] = DV
    out["M7"] = np.log1p(DV)
    out["M8"] = V
    out["M9"] = (H - L) / C
    # M10 Roll (spec 6.10): gamma = first-order autocovariance of the
    # 21 price changes in the window ending at j, computed as
    # numpy.cov(d[1:], d[:-1], ddof=1) over the 20 adjacent pairs —
    # identically the rolling covariance (ddof=1) of (d_t, d_{t-1})
    # over the 20 pairs ending at j.
    d = np.full(n, np.nan)
    d[1:] = C[1:] - C[:-1]
    ds = pd.Series(d)
    d_lag = ds.shift(1)
    gamma = ds.rolling(20).cov(d_lag).to_numpy()
    m10 = np.where(gamma < 0, 2.0 * np.sqrt(-np.minimum(gamma, 0.0)), 0.0)
    m10[:21] = np.nan
    m10[np.isnan(gamma)] = np.nan
    out["M10"] = m10
    # M11 Corwin-Schultz (spec 6.11).
    lh = np.log(H / L)
    beta = np.full(n, np.nan)
    beta[1:] = lh[:-1] ** 2 + lh[1:] ** 2
    h2 = np.maximum(H[:-1], H[1:])
    l2 = np.minimum(L[:-1], L[1:])
    gam = np.full(n, np.nan)
    gam[1:] = np.log(h2 / l2) ** 2
    denom = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denom - np.sqrt(gam / denom)
    exp_a = np.exp(alpha)
    s11 = 2.0 * (exp_a - 1.0) / (1.0 + exp_a)
    s11 = np.where(s11 < 0, 0.0, s11)
    out["M11"] = pd.Series(s11).rolling(21).mean().to_numpy()
    # M12 Abdi-Ranaldo CHLO (spec 6.12): daily flooring before the mean.
    m_log = (np.log(H) + np.log(L)) / 2.0
    rho = np.full(n, np.nan)
    rho[1:] = 0.5 * ((np.log(C[1:]) - m_log[1:]) * (np.log(C[1:]) - m_log[:-1])
                     + (np.log(O[1:]) - m_log[1:]) * (np.log(O[1:]) - m_log[:-1]))
    s12 = np.where(rho > 0, 2.0 * np.sqrt(np.where(rho > 0, rho, 0.0)), 0.0)
    s12[0] = np.nan
    out["M12"] = pd.Series(s12).rolling(21).mean().to_numpy()
    # M14 EDGE (spec 6.14 as amended by Amendment A2, 2026-10-29):
    # EDGE estimate over the trailing 21 trading days, dated at the
    # window's last trading day.
    out["M14"] = compute_edge_series(O, H, L, C)
    # M16 LOT (spec 6.16).
    if with_lot:
        lot, failed = compute_lot_series(f["date"], C, spy_ret_by_date)
        out["M16"] = lot
        out.attrs["lot_failed_windows"] = failed
    else:
        out["M16"] = np.nan
    # M17 Pastor-Stambaugh gamma, raw-return form (spec 6.17).
    out["M17"] = _pastor_stambaugh_step(f["date"], r, DV, V)
    # M18 / M19 (spec 6.18 / 6.19).
    zr = pd.Series(np.where(np.isnan(r), np.nan, (r == 0).astype(float)))
    out["M18"] = zr.rolling(21).mean().to_numpy()
    zv = pd.Series((V == 0).astype(float))
    out["M19"] = zv.rolling(21).mean().to_numpy()
    # M20 Amivest (spec 6.20): full 21-position window, >= 10 defined.
    q = np.full(n, np.nan)
    okq = pos_vol & np.isfinite(r) & (r != 0)
    q[okq] = DV[okq] / np.abs(r[okq])
    m20 = np.array(pd.Series(q).rolling(21, min_periods=10).mean())
    m20[:20] = np.nan
    out["M20"] = m20
    # M21 Amihud 21-day mean (spec 6.21): full window, >= 10 defined.
    m21 = np.array(pd.Series(illiq).rolling(21, min_periods=10).mean())
    m21[:20] = np.nan
    out["M21"] = m21
    return out


def _pastor_stambaugh_step(dates: pd.Series, r: np.ndarray, DV: np.ndarray,
                           V: np.ndarray) -> np.ndarray:
    """Spec section 6.17: monthly OLS gamma, dated to the month's last
    trading day and carried forward (step series)."""
    n = len(dates)
    step = np.full(n, np.nan)
    months = dates.dt.to_period("M").to_numpy()
    usable = (V > 0) & np.isfinite(r)
    # month -> (last position, gamma or nan)
    month_gamma: dict = {}
    uniq, start_idx = np.unique(months, return_index=True)
    bounds = list(start_idx) + [n]
    for mi, mo in enumerate(uniq):
        lo, hi = bounds[mi], bounds[mi + 1]  # positions [lo, hi)
        idx = np.arange(lo, hi)
        use_idx = idx[usable[idx]]
        if len(use_idx) < 15:
            continue
        # adjacent pairs (d, d+1): both usable, in this month, d+1 the
        # next panel position after d.
        pairs = [(d, d + 1) for d in use_idx
                 if d + 1 < hi and usable[d + 1]]
        if len(pairs) < 15:
            continue
        d_idx = np.array([p[0] for p in pairs])
        y = r[d_idx + 1]
        x1 = r[d_idx]
        x2 = np.sign(r[d_idx]) * DV[d_idx] / 1e9
        X = np.column_stack([np.ones(len(pairs)), x1, x2])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        month_gamma[hi - 1] = float(beta[2])
    last = np.nan
    for j in range(n):
        if j in month_gamma:
            last = month_gamma[j]
        step[j] = last
    return step


# --- Section 7 features -------------------------------------------------------------

def _rolling_median_defined(s: pd.Series, window: int, min_defined: int) -> np.ndarray:
    return s.rolling(window, min_periods=min_defined).median().to_numpy()


def _rolling_mad(s: np.ndarray, center: np.ndarray, window: int,
                 min_defined: int) -> np.ndarray:
    """Feature 6: median of |s_t - center_i| over the defined s_t in
    positions i-window+1..i (center_i = feature 5's value at i),
    requiring at least min_defined defined values."""
    n = len(s)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    # Partial windows at the start (positions i < window - 1), matching
    # the truncated window of feature 5's rolling median.
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
        W = sliding_window_view(s, window)  # row k covers s[k .. k+window-1]
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


def build_fund_features(panel_fund: pd.DataFrame, s: np.ndarray,
                        common: pd.DataFrame) -> pd.DataFrame:
    """The 14 frozen numeric features of spec section 7 for one fund
    and one measure, plus labels, label end dates and baseline inputs.
    common carries the fund's date-indexed common features (features
    7-14 inputs and calendar columns)."""
    f = common.reset_index(drop=True).copy()
    ss = pd.Series(s)
    f["meas_level"] = s
    f["meas_chg5"] = s - np.concatenate([np.full(5, np.nan), s[:-5]]) if len(s) >= 5 else np.nan
    f["meas_chg21"] = s - np.concatenate([np.full(21, np.nan), s[:-21]]) if len(s) >= 21 else np.nan
    f["meas_med21"] = _rolling_median_defined(ss, 21, 10)
    med252 = _rolling_median_defined(ss, 252, 20)
    f["meas_med252"] = med252
    f["meas_mad252"] = _rolling_mad(s, med252, 252, 20)
    for h in HORIZONS:
        lab = np.full(len(s), np.nan)
        if len(s) > h:
            lab[:-h] = s[h:]
        f[f"y_{h}"] = lab
        end = f["date"].to_numpy().copy()
        end_shifted = np.full(len(s), np.datetime64("NaT"), dtype="datetime64[ns]")
        if len(s) > h:
            end_shifted[:-h] = end[h:]
        f[f"label_end_date_{h}"] = pd.to_datetime(end_shifted)
    return f


def build_common_features(panel_fund: pd.DataFrame,
                          market: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Features 7-14 and calendar columns for one fund (measure-
    independent), spec section 7."""
    f = panel_fund[["ticker", "date", "pos"]].reset_index(drop=True).copy()
    C = panel_fund["close"].to_numpy(dtype=float)
    V = panel_fund["volume"].to_numpy(dtype=float)
    n = len(f)
    r = np.full(n, np.nan)
    r[1:] = C[1:] / C[:-1] - 1.0
    f["ret1_bp"] = r * 10_000.0
    ret5 = np.full(n, np.nan)
    ret5[5:] = C[5:] / C[:-5] - 1.0
    f["ret5_bp"] = ret5 * 10_000.0
    f["absret1_bp"] = np.abs(r) * 10_000.0
    lv = np.log1p(V)
    f["logvol"] = lv
    lv_s = pd.Series(lv)
    f["logvol_chg5"] = (lv_s.rolling(5).mean()
                        - lv_s.rolling(10).mean().shift(5)).to_numpy()
    # Market-state features by as-of join (spec section 7.2).
    f = pd.merge_asof(f.sort_values("date"),
                      market["^VIX"][["date", "vix_close"]], on="date",
                      direction="backward").rename(columns={"vix_close": "vix_level"})
    f = pd.merge_asof(f.sort_values("date"),
                      market["^TNX"][["date", "tnx_chg5"]], on="date",
                      direction="backward")
    f = pd.merge_asof(f.sort_values("date"),
                      market["SPY"][["date", "spy_ret5_bp"]], on="date",
                      direction="backward")
    f = f.sort_values("pos").reset_index(drop=True)
    wd = f["date"].dt.weekday
    assert set(wd.unique()) <= {0, 1, 2, 3, 4}, "weekend panel date"
    for k, name in [(1, "dow_tue"), (2, "dow_wed"), (3, "dow_thu"), (4, "dow_fri")]:
        f[name] = (wd == k).astype(float)
    return f


def design_columns(measure: str) -> list[str]:
    funds = universe_for(measure)
    return (NUMERIC_FEATURES + [f"fund_{t}" for t in funds]
            + ["dow_tue", "dow_wed", "dow_thu", "dow_fri"])


def build_sample(panel: pd.DataFrame, measure: str, horizon: int,
                 market: dict[str, pd.DataFrame],
                 common_by_fund: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Spec section 8 sample for one measure x horizon: every numeric
    feature non-NaN and the label computable. Sorted by (date,
    ticker)."""
    funds = universe_for(measure)
    parts = []
    for t in funds:
        pf = panel[panel["ticker"] == t]
        s = pf[measure].to_numpy(dtype=float)
        feats = build_fund_features(pf, s, common_by_fund[t])
        parts.append(feats)
    d = pd.concat(parts, ignore_index=True)
    ycol = f"y_{horizon}"
    d = d[d[NUMERIC_FEATURES].notna().all(axis=1) & d[ycol].notna()].copy()
    d["y"] = d[ycol].astype(float)
    d["label_end_date"] = d[f"label_end_date_{horizon}"]
    for t in funds:
        d[f"fund_{t}"] = (d["ticker"] == t).astype(float)
    d = d.sort_values(["date", "ticker"]).reset_index(drop=True)
    assert d[NUMERIC_FEATURES + ["y"]].notna().all().all()
    return d


# --- Section 9 pooled folds ---------------------------------------------------------

def make_pooled_folds(sample: pd.DataFrame, horizon: int):
    """Expanding walk-forward folds counted in pooled sample dates
    (spec section 9.1): test dates D[252 + 63k ... 252 + 63(k+1) - 1];
    training candidates dated strictly before the first test date;
    purge drops a training row whose label end date is on or after
    the first test date; embargo keeps a training row only if its
    feature date's position in D is at most (first test date
    position) - 6. Assertions raise (spec section 11.1 item 3)."""
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
    return folds


# --- Section 10 model ladder -----------------------------------------------------------

def ridge_fit_predict_t5b(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray):
    """Ridge per spec section 10 item 3: the 14 numeric features are
    standardised inside the pipeline (statistics from the training
    fold only); fund and day-of-week indicators are appended unscaled;
    alpha selected from the frozen grid by inner TimeSeriesSplit(3)
    on the training fold only, then refit on the full training fold —
    the Target 1 / Target 4a / Target 5a policy and grid of
    engine/ml_models.py."""
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


# --- Ridge via Gram assembly (identical procedure, efficient evaluation) ----------
#
# The frozen ridge procedure (spec section 10 item 3) is: standardise
# the 14 numeric features with statistics from the training subset in
# question, append fund and day-of-week indicators unscaled, fit
# sklearn Ridge (intercept by centering, penalty alpha * ||w||^2 on all
# coefficients), select alpha from the frozen grid by inner
# TimeSeriesSplit(3) cross-validation (lowest mean CV mean squared
# error), refit on the full training fold. Because the design has 155
# columns but the indicator block is one-hot, the normal equations of
# every candidate fit are assembled exactly from group aggregates
# (Gram matrix and X'y), which evaluates the identical procedure
# without repeatedly copying the pooled design matrix. The driver
# verifies this evaluation against the sklearn pipeline above on a
# sample fold before the run (agreement recorded in the results JSON).

def _ridge_gram(Xn, fund_idx, dow_idx, y, rows, n_funds):
    """Gram matrix, X'y, column means and y mean of the transformed
    design over the given rows, with the numeric block standardised by
    this row subset's own statistics (the pipeline's scaler step)."""
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
    # numeric block
    G[:n_num, :n_num] = Z.T @ Z
    xty[:n_num] = Z.T @ yy
    # numeric x fund
    ZF = np.zeros((n_funds, n_num))
    np.add.at(ZF, fi, Z)
    G[:n_num, n_num:n_num + n_funds] = ZF.T
    G[n_num:n_num + n_funds, :n_num] = ZF
    xty[n_num:n_num + n_funds] = np.bincount(fi, weights=yy, minlength=n_funds)
    # numeric x dow (columns for dow 1..4; dow 0 = Monday reference)
    ZD = np.zeros((4, n_num))
    m_dow = di > 0
    np.add.at(ZD, di[m_dow] - 1, Z[m_dow])
    G[:n_num, n_num + n_funds:] = ZD.T
    G[n_num + n_funds:, :n_num] = ZD
    xty[n_num + n_funds:] = np.bincount(di[m_dow] - 1, weights=yy[m_dow],
                                        minlength=4)
    # fund x fund (diagonal counts), fund x dow counts, dow x dow
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
    """The frozen ridge procedure evaluated by Gram assembly (see the
    note above). train_rows / test_rows are positional indices into
    the sample arrays, in sample order."""
    Xtr_n = Xn[train_rows]
    splits = list(TimeSeriesSplit(3).split(Xtr_n))
    # CV: scaler and ridge fitted on each CV training subset only.
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
