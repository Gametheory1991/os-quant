"""Institutional stats sheet for strategy evaluation.

``report(returns, benchmark=None, turnover=None, risk_free=0.0, label="",
n_trials=None, tnx=None, vix=None, factors=None)`` takes a daily net-return
Series and produces the full dict of metrics plus a formatted text block. All
annualizations assume 252 trading days.

Definitions (documented so the numbers are auditable):
  * Sharpe/Sortino: mean/std of daily returns, annualized, vs ``risk_free``
    (annual, simple). Sortino uses downside deviation only.
  * Max drawdown: from the NAV curve; dates = trough date and the peak date
    the drawdown started from.
  * Win rate: fraction of positive-return months (and days, reported separately).
  * Win/loss ratio: mean positive monthly return / |mean negative monthly return|.
  * Profit factor: sum of positive monthly returns / |sum of negative|.
  * VaR 95/99: historical one-day VaR, quoted as a positive loss number.
  * CVaR 95/99: expected shortfall -- mean of daily returns at/below the VaR
    threshold, quoted as a positive loss number (1-day).
  * Beta/alpha: OLS of strategy excess returns on benchmark excess returns;
    alpha annualized. Tracking error / information ratio vs the benchmark.
  * Turnover: mean monthly turnover (fraction of NAV traded), when supplied.
  * Skew / excess kurtosis: Fisher moments of daily returns, computed from
    population moments (ddof=0): skew = m3 / m2**1.5,
    excess kurtosis = m4 / m2**2 - 3.
  * PSR -- Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012):
      PSR(SR*) = Phi( (SR - SR*) * sqrt(T-1)
                      / sqrt(1 - g3*SR + (g4-1)/4 * SR^2) )
    where SR is the annualized Sharpe, T the number of daily observations,
    g3/g4 the skewness/kurtosis of daily (excess) returns, and Phi the
    standard normal cdf. report() emits PSR(0): the probability the true
    Sharpe exceeds zero, under the paper's moment-adjusted normal
    approximation. Always computable.
  * DSR -- Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014): the PSR form
    above with SR* replaced by SR0, the expected Sharpe under H0 given
    selection over K = ``n_trials`` backtest trials:
      SR0 = sqrt(Vhat) * ((1-gamma) * Phi^{-1}(1 - 1/K)
                          + gamma * Phi^{-1}(1 - 1/(K*e)))
    gamma = 0.5772156649 (Euler-Mascheroni). Vhat is the variance of the
    trial Sharpe estimates; with only the trial count known (the trial Sharpe
    vector is not retained) it is approximated as 252/T -- the variance of
    the annualized Sharpe estimator under H0 with T daily observations.
    K <= 1 -> SR0 = 0 (no selection bias, DSR collapses to PSR(0)).
    Requires ``n_trials`` (the strategy's free-parameter count, documented by
    the caller); without it the field is "pending engine upgrade".
  * Breakeven TC: the one-way cost in bps per unit of daily turnover that
    drives the net Sharpe to zero. With cost drag c/1e4 * turnover_daily,
    mean(r - drag) = 0  =>  c_bps = mean(r) / mean(turnover_daily) * 1e4.
    Needs ``turnover``; otherwise "pending engine upgrade".
  * Drawdown duration: calendar days from the max-drawdown peak to the first
    date with NAV >= the prior peak (1e-9 relative tolerance). Unrecovered ->
    dd_status is "still in DD", dd_duration_days/dd_recovery_date are None,
    and dd_days_since_peak counts peak -> sample end.
  * Expectancy per rebalance: mean net P&L per rebalance interval per $1 NAV,
    computed as the mean of monthly-resampled net returns x 100 (in % per $1).
    The current strategies rebalance monthly, so one calendar month = one
    rebalance interval; a strategy on a different cadence would need the
    resampling changed to match.
  * Newey-West t-stat: t = mean(daily returns) / SE_NW, where SE_NW is the
    Bartlett-kernel Newey-West standard error of the mean with
    lag = floor(4 * (n/100)^(2/9)). Computed on raw daily net returns (not
    excess). Implemented manually -- requirements.txt has no statsmodels and
    none is added.
  * Regimes: dated subperiod stats (Sharpe, CAGR, n_days, explicit start/end)
    for (i) ^TNX rising vs falling -- sign of the 63-trading-day change in
    ``tnx``; (ii) ^VIX high vs low -- ``vix`` above/below its trailing
    252-day median; (iii) crisis vs calm -- crisis is the union of the dated
    episodes GFC 2008-09-01->2008-12-31, Covid 2020-02-20->2020-03-23 and
    2022 hikes 2022-06-01->2022-10-31, everything else calm. Needs ``tnx``
    and ``vix`` (daily closes, aligned to the returns index); without both,
    the block is "pending engine upgrade".
  * Walk-forward: the sample split in calendar halves (midpoint of the date
    range); per half: Sharpe, CAGR, max drawdown, n_days, start/end.
  * Factor correlations: Pearson correlation of daily strategy returns to each
    caller-supplied daily factor return series (e.g. value/momentum/carry
    book returns, TLT excess, HYG-SHY). Needs ``factors``; otherwise
    "pending engine upgrade", plus a note naming the proxies.
  * Capacity/crowding: out of scope by design -- there is no free source for
    volume/borrow data, so nothing is computed here.

JSON safety: every new field is a python float/int/str/dict/None. Statistically
uncomputable values are None (never NaN, NaT, Timestamps or DataFrames). The
literal string "pending engine upgrade" is reserved for inputs the engine was
not given (n_trials, turnover, tnx/vix, factors) -- never for estimates.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

TRADING_DAYS = 252

_PHI = NormalDist()  # stdlib normal cdf / inverse cdf; no scipy dependency
_EULER_MASCHERONI = 0.5772156649

#: Dated crisis episodes for the regime block (acute-phase start -> finish).
CRISIS_EPISODES = (
    ("2008-09-01", "2008-12-31"),  # GFC
    ("2020-02-20", "2020-03-23"),  # Covid crash
    ("2022-06-01", "2022-10-31"),  # 2022 rate-hike drawdown
)

#: Literal emitted when the engine was not given an input a stat needs.
PENDING = "pending engine upgrade"


def _ann_factor(periods_per_year: int = TRADING_DAYS) -> float:
    return float(periods_per_year)


def _degenerate(x: pd.Series) -> bool:
    """True when a series has no usable variation.

    std() of a constant series can come back as a ~1e-19 float residue
    rather than exactly 0, so degeneracy needs a relative guard.
    """
    v = x.to_numpy(dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return True
    s = float(v.std(ddof=1))
    scale = float(np.sqrt((v ** 2).mean()))
    return (not np.isfinite(s)) or s <= 1e-9 * scale


def _ann_sharpe(excess: pd.Series) -> float | None:
    """Annualized Sharpe of an excess-return series; None if degenerate."""
    if _degenerate(excess):
        return None
    v = excess.dropna().to_numpy(dtype=float)
    return float(v.mean() / v.std(ddof=1) * math.sqrt(TRADING_DAYS))


def _moments(x: pd.Series) -> tuple[float | None, float | None]:
    """Fisher skewness and (non-excess) kurtosis via population moments."""
    v = x.to_numpy(dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 3:
        return None, None
    m = v.mean()
    m2 = ((v - m) ** 2).mean()
    if not (m2 > 0):
        return None, None
    m3 = ((v - m) ** 3).mean()
    m4 = ((v - m) ** 4).mean()
    return float(m3 / m2 ** 1.5), float(m4 / m2 ** 2)


def _psr_z(sr: float, t: int, g3: float, g4: float, sr_star: float = 0.0) -> float | None:
    """Moment-adjusted z-statistic shared by PSR and DSR."""
    inner = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2
    if inner <= 0 or t < 2:
        return None
    return float((sr - sr_star) * math.sqrt(t - 1) / math.sqrt(inner))


def probabilistic_sharpe(
    returns: pd.Series, sr_star: float = 0.0, risk_free: float = 0.0
) -> float | None:
    """Probabilistic Sharpe Ratio: P(true Sharpe > sr_star).

    Bailey & Lopez de Prado (2012):
    PSR(SR*) = Phi((SR - SR*) * sqrt(T-1) / sqrt(1 - g3*SR + (g4-1)/4*SR^2)).
    """
    r = returns.dropna().astype(float)
    if len(r) < 2:
        return None
    excess = r - risk_free / TRADING_DAYS
    sr = _ann_sharpe(excess)
    g3, g4 = _moments(excess)
    if sr is None or g3 is None or g4 is None:
        return None
    z = _psr_z(sr, len(excess), g3, g4, sr_star)
    if z is None:
        return None
    return float(min(max(_PHI.cdf(z), 0.0), 1.0))


def _dsr_sr0(t: int, n_trials: int) -> float:
    """Expected Sharpe under H0 given selection over n_trials backtests."""
    if n_trials <= 1:
        return 0.0
    k = float(n_trials)
    m_k = ((1 - _EULER_MASCHERONI) * _PHI.inv_cdf(1 - 1 / k)
           + _EULER_MASCHERONI * _PHI.inv_cdf(1 - 1 / (k * math.e)))
    v_hat = TRADING_DAYS / t  # Var(annualized Sharpe estimator) under H0
    return float(math.sqrt(v_hat) * m_k)


def deflated_sharpe(
    returns: pd.Series, n_trials: int | None, risk_free: float = 0.0
) -> float | None:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

    DSR = Phi((SR - SR0) * sqrt(T-1) / sqrt(1 - g3*SR + (g4-1)/4*SR^2)),
    with SR0 = _dsr_sr0(T, n_trials). Returns None when n_trials is missing
    or the moments are degenerate; report() maps that to PENDING.
    """
    if n_trials is None or n_trials < 1:
        return None
    r = returns.dropna().astype(float)
    if len(r) < 2:
        return None
    excess = r - risk_free / TRADING_DAYS
    t = len(excess)
    sr = _ann_sharpe(excess)
    g3, g4 = _moments(excess)
    if sr is None or g3 is None or g4 is None:
        return None
    z = _psr_z(sr, t, g3, g4, _dsr_sr0(t, int(n_trials)))
    if z is None:
        return None
    return float(min(max(_PHI.cdf(z), 0.0), 1.0))


def newey_west_tstat(returns: pd.Series, risk_free: float = 0.0) -> float | None:
    """t-stat of mean daily returns with a Bartlett Newey-West SE.

    lag = floor(4 * (n/100)^(2/9)). Implemented manually (no statsmodels).
    Computed on raw daily net returns, not excess.
    """
    x = returns.dropna().astype(float).to_numpy() - risk_free / TRADING_DAYS
    n = len(x)
    if n < 2:
        return None
    lag = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    mu = float(x.mean())
    d = x - mu
    s = float((d * d).sum() / n)
    for j in range(1, lag + 1):
        gj = float((d[j:] * d[:-j]).sum() / n)
        s += 2.0 * (1.0 - j / (lag + 1.0)) * gj
    if not (s > 0):
        return None
    return float(mu / math.sqrt(s / n))


def _breakeven_tc_bps(returns: pd.Series, turnover: pd.Series) -> float | None:
    """One-way bps per unit daily turnover that zeroes the net Sharpe."""
    r = returns.dropna().astype(float)
    t = turnover.reindex(r.index).dropna().astype(float)
    mean_t = float(t.mean()) if len(t) else 0.0
    if not (mean_t > 0):
        return None
    return float(r.mean() / mean_t * 1e4)


def max_drawdown(nav: pd.Series) -> dict:
    peak = nav.cummax()
    dd = nav / peak - 1.0
    trough_date = dd.idxmin()
    peak_date = nav.loc[:trough_date].idxmax()
    return {
        "max_drawdown": float(dd.min()),
        "dd_start": str(peak_date.date()) if hasattr(peak_date, "date") else str(peak_date),
        "dd_trough": str(trough_date.date()) if hasattr(trough_date, "date") else str(trough_date),
    }


def drawdown_duration(nav: pd.Series) -> dict:
    """Calendar-day duration of the max-drawdown event.

    Peak -> first date with NAV >= prior peak (1e-9 relative tolerance).
    Unrecovered -> dd_status "still in DD" with dd_days_since_peak counting
    peak -> sample end.
    """
    nav = nav.dropna().astype(float)
    if len(nav) < 2:
        raise ValueError("need at least 2 NAV observations")
    peak = nav.cummax()
    dd = nav / peak - 1.0
    trough_idx = dd.idxmin()
    peak_idx = nav.loc[:trough_idx].idxmax()
    peak_val = float(nav.loc[peak_idx])
    peak_day = pd.Timestamp(peak_idx).date()
    end_day = pd.Timestamp(nav.index[-1]).date()
    if dd.min() >= 0:
        return {
            "dd_duration_days": 0,
            "dd_status": "no drawdown",
            "dd_recovery_date": str(peak_day),
            "dd_days_since_peak": 0,
        }
    after = nav.loc[peak_idx:]
    rec = after[(after >= peak_val * (1.0 - 1e-9)) & (after.index > peak_idx)]
    if len(rec):
        rec_day = pd.Timestamp(rec.index[0]).date()
        return {
            "dd_duration_days": int((rec_day - peak_day).days),
            "dd_status": "recovered",
            "dd_recovery_date": str(rec_day),
            "dd_days_since_peak": None,
        }
    return {
        "dd_duration_days": None,
        "dd_status": "still in DD",
        "dd_recovery_date": None,
        "dd_days_since_peak": int((end_day - peak_day).days),
    }


def monthly_pnl(returns: pd.Series) -> pd.DataFrame:
    """Monthly P&L table: rows = years, columns = months, values = % return."""
    m = (1 + returns).resample("ME").prod() - 1
    tbl = m.groupby([m.index.year, m.index.month]).first().unstack()
    tbl.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return (tbl * 100).round(2)


def yearly_returns(returns: pd.Series) -> pd.Series:
    return ((1 + returns).resample("YE").prod() - 1) * 100


def _subperiod_stats(r: pd.Series, mask: pd.Series, rf_d: float) -> dict:
    """Sharpe/CAGR/n_days/start/end for a boolean regime mask."""
    sub = r[mask]
    n = int(len(sub))
    out = {
        "n_days": n,
        "start": str(sub.index[0].date()) if n else None,
        "end": str(sub.index[-1].date()) if n else None,
        "sharpe": None,
        "cagr": None,
    }
    if n >= 2:
        out["sharpe"] = _ann_sharpe(sub - rf_d)
        nav = (1 + sub).cumprod()
        out["cagr"] = float(nav.iloc[-1] ** (1 / (n / TRADING_DAYS)) - 1)
    return out


def regime_stats(
    returns: pd.Series,
    tnx: pd.Series,
    vix: pd.Series,
    risk_free: float = 0.0,
) -> dict:
    """Dated subperiod stats for rates / volatility / crisis regimes.

    (i) ^TNX rising vs falling: sign of the 63-trading-day change in ``tnx``
    (exactly-zero changes belong to neither leg);
    (ii) ^VIX high vs low: ``vix`` above / at-or-below its trailing 252-day
    median (days without a full 252-day window are excluded);
    (iii) crisis vs calm: crisis = union of CRISIS_EPISODES.
    """
    r = returns.dropna().astype(float)
    rf_d = risk_free / TRADING_DAYS
    tnx = tnx.reindex(r.index).astype(float)
    vix = vix.reindex(r.index).astype(float)

    chg63 = tnx - tnx.shift(63)
    rising = chg63 > 0
    falling = chg63 < 0

    med252 = vix.rolling(252).median()
    vix_high = vix > med252
    vix_low = vix <= med252

    dates = r.index.normalize()
    crisis_mask = pd.Series(False, index=r.index)
    for s, e in CRISIS_EPISODES:
        crisis_mask |= (dates >= pd.Timestamp(s)) & (dates <= pd.Timestamp(e))
    calm_mask = ~crisis_mask

    return {
        "rates": {
            "definition": "sign of ^TNX 63-trading-day change",
            "rising": _subperiod_stats(r, rising, rf_d),
            "falling": _subperiod_stats(r, falling, rf_d),
        },
        "volatility": {
            "definition": "^VIX above/below trailing 252-day median",
            "high": _subperiod_stats(r, vix_high, rf_d),
            "low": _subperiod_stats(r, vix_low, rf_d),
        },
        "crisis": {
            "definition": "crisis = " + ", ".join(f"{s}->{e}" for s, e in CRISIS_EPISODES),
            "crisis": _subperiod_stats(r, crisis_mask, rf_d),
            "calm": _subperiod_stats(r, calm_mask, rf_d),
        },
    }


def walk_forward(returns: pd.Series, risk_free: float = 0.0) -> dict:
    """Split the sample in calendar halves; per half: Sharpe, CAGR, max DD."""
    r = returns.dropna().astype(float)
    if len(r) < 2:
        raise ValueError("need at least 2 return observations")
    rf_d = risk_free / TRADING_DAYS
    mid = r.index[0] + (r.index[-1] - r.index[0]) / 2
    out = {}
    for name, sub in (("first_half", r[r.index <= mid]),
                      ("second_half", r[r.index > mid])):
        n = int(len(sub))
        nav = (1 + sub).cumprod()
        out[name] = {
            "n_days": n,
            "start": str(sub.index[0].date()) if n else None,
            "end": str(sub.index[-1].date()) if n else None,
            "sharpe": _ann_sharpe(sub - rf_d) if n >= 2 else None,
            "cagr": float(nav.iloc[-1] ** (1 / (n / TRADING_DAYS)) - 1) if n >= 2 else None,
            "max_drawdown": float((nav / nav.cummax() - 1).min()) if n else None,
        }
    return out


def parameter_sensitivity(
    run_fn,
    base_params: dict,
    perturbations: dict | None = None,
) -> dict:
    """One-at-a-time Sharpe sensitivity to parameter perturbations.

    ``run_fn(params_dict)`` -> daily returns Series. Each named numeric
    parameter is scaled by (1+delta) for delta in [-0.5, -0.2, 0.2, 0.5]
    (override per parameter via ``perturbations={name: [deltas]}``), holding
    the others at base. Returns {"base_sharpe",
    "table": {param: {delta: sharpe}},
    "sharpe_degradation": {param: {delta: base_sharpe - perturbed_sharpe}}}.
    Helper for the runner layer -- not used inside report().
    """
    deltas_default = [-0.5, -0.2, 0.2, 0.5]
    base_sharpe = _ann_sharpe(run_fn(dict(base_params)).dropna().astype(float))
    table, degr = {}, {}
    for param, base_val in base_params.items():
        try:
            bv = float(base_val)
        except (TypeError, ValueError):
            raise TypeError(
                f"parameter_sensitivity: param {param!r} is not numeric "
                f"(value {base_val!r}); only numeric lookbacks/thresholds "
                "can be scaled"
            )
        deltas = (perturbations or {}).get(param, deltas_default)
        trow, drow = {}, {}
        for d in deltas:
            p = dict(base_params)
            p[param] = bv * (1.0 + d)
            s = _ann_sharpe(run_fn(p).dropna().astype(float))
            trow[d] = s
            drow[d] = (base_sharpe - s) if (base_sharpe is not None and s is not None) else None
        table[param] = trow
        degr[param] = drow
    return {"base_sharpe": base_sharpe, "table": table, "sharpe_degradation": degr}


def factor_correlations(returns: pd.Series, factors: dict) -> dict:
    """Pearson correlation of daily strategy returns to each daily factor
    return series (index-aligned). Uncomputable pairs -> None."""
    r = returns.dropna().astype(float)
    out = {}
    for name, f in factors.items():
        f = pd.Series(f).reindex(r.index).dropna().astype(float)
        common = r.index.intersection(f.index)
        corr = None
        if len(common) > 2 and not _degenerate(r.loc[common]) and not _degenerate(f.loc[common]):
            c = r.loc[common].corr(f.loc[common])
            corr = float(c) if np.isfinite(c) else None
        out[name] = corr
    return out


def strategy_correlations(returns_by_name: dict) -> dict:
    """Pairwise Pearson correlations among the books in the set.

    ``returns_by_name`` maps a book key to its daily net-return Series.
    Returns {book: {other_book: corr}} — each book's row, computed on the
    pairwise overlap of the two series (books have different inception
    dates, so overlaps differ by pair). A book's correlation with itself
    is 1.0 when it has more than 2 observations. Uncomputable pairs ->
    None. Helper for the runner layer -- not used inside report(),
    because report() sees one book at a time.
    """
    out: dict = {}
    for name, r in returns_by_name.items():
        r = pd.Series(r).dropna().astype(float)
        row: dict = {}
        for other, s in returns_by_name.items():
            s = pd.Series(s).dropna().astype(float)
            if other == name:
                row[other] = 1.0 if len(r) > 2 else None
                continue
            common = r.index.intersection(s.index)
            corr = None
            if (len(common) > 2 and not _degenerate(r.loc[common])
                    and not _degenerate(s.loc[common])):
                c = r.loc[common].corr(s.loc[common])
                corr = float(c) if np.isfinite(c) else None
            row[other] = corr
        out[name] = row
    return out


def report(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    turnover: pd.Series | None = None,
    risk_free: float = 0.0,
    label: str = "strategy",
    n_trials: int | None = None,
    tnx: pd.Series | None = None,
    vix: pd.Series | None = None,
    factors: dict | None = None,
) -> dict:
    """Full stats dict; see module docstring for definitions.

    New optional kwargs: ``n_trials`` (drives ``deflated_sharpe``),
    ``tnx``/``vix`` (daily closes, drive ``regimes``), ``factors``
    (dict of daily factor return series, drives ``factor_correlations``).
    Any of them omitted -> the corresponding field(s) emit
    "pending engine upgrade".
    """
    r = returns.dropna().astype(float)
    if len(r) < 2:
        raise ValueError("need at least 2 return observations")
    rf_d = risk_free / TRADING_DAYS
    excess = r - rf_d
    nav = (1 + r).cumprod()

    n_years = len(r) / TRADING_DAYS
    total_return = float(nav.iloc[-1] - 1)
    cagr = float(nav.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else float("nan")
    vol = float(excess.std() * np.sqrt(TRADING_DAYS))
    sharpe = float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)) if excess.std() > 0 else float("nan")
    downside = excess[excess < 0]
    sortino = (
        float(excess.mean() / downside.std() * np.sqrt(TRADING_DAYS))
        if len(downside) and downside.std() > 0 else float("nan")
    )
    dd = max_drawdown(nav)
    calmar = float(cagr / abs(dd["max_drawdown"])) if dd["max_drawdown"] < 0 else float("nan")

    monthly = (1 + r).resample("ME").prod() - 1
    monthly = monthly.dropna()
    win_rate_m = float((monthly > 0).mean()) if len(monthly) else float("nan")
    win_rate_d = float((r > 0).mean())
    pos_m, neg_m = monthly[monthly > 0], monthly[monthly < 0]
    win_loss = float(pos_m.mean() / abs(neg_m.mean())) if len(neg_m) and neg_m.mean() != 0 else float("nan")
    profit_factor = float(pos_m.sum() / abs(neg_m.sum())) if len(neg_m) and neg_m.sum() != 0 else float("nan")

    var95 = float(-r.quantile(0.05))
    var99 = float(-r.quantile(0.01))

    stats: dict = {
        "label": label,
        "start": str(r.index[0].date()),
        "end": str(r.index[-1].date()),
        "n_days": int(len(r)),
        "n_years": round(n_years, 2),
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        **dd,
        "win_rate_daily": win_rate_d,
        "win_rate_monthly": win_rate_m,
        "win_loss_ratio_monthly": win_loss,
        "profit_factor_monthly": profit_factor,
        "var95_daily": var95,
        "var99_daily": var99,
        "avg_monthly_turnover": float(turnover.resample("ME").sum().mean()) if turnover is not None else None,
    }

    if benchmark is not None:
        b = benchmark.reindex(r.index).dropna()
        common = r.index.intersection(b.index)
        rs, bs = r.loc[common] - rf_d, b.loc[common] - rf_d
        if len(common) > 2 and bs.std() > 0:
            beta = float(rs.cov(bs) / bs.var())
            alpha_d = float(rs.mean() - beta * bs.mean())
            stats["beta"] = beta
            stats["alpha_ann"] = float((1 + alpha_d) ** TRADING_DAYS - 1)
            # R^2 of the regression
            stats["r_squared"] = float(rs.corr(bs) ** 2)
            active = rs - bs
            stats["tracking_error"] = float(active.std() * np.sqrt(TRADING_DAYS))
            stats["information_ratio"] = (
                float(active.mean() / active.std() * np.sqrt(TRADING_DAYS))
                if active.std() > 0 else float("nan")
            )
        else:
            stats.update(beta=float("nan"), alpha_ann=float("nan"),
                         r_squared=float("nan"), tracking_error=float("nan"),
                         information_ratio=float("nan"))

    # ---- expanded Results block ----
    g3, g4 = _moments(excess)
    stats["skew_daily"] = g3
    stats["excess_kurtosis_daily"] = (g4 - 3.0) if g4 is not None else None

    tail95 = r[r <= r.quantile(0.05)]
    tail99 = r[r <= r.quantile(0.01)]
    stats["cvar95_daily"] = float(-tail95.mean()) if len(tail95) else None
    stats["cvar99_daily"] = float(-tail99.mean()) if len(tail99) else None

    stats["psr"] = probabilistic_sharpe(r, sr_star=0.0, risk_free=risk_free)
    if n_trials is None or n_trials < 1:
        stats["n_trials"] = None
        stats["deflated_sharpe"] = PENDING
        stats["deflated_sharpe_sr0"] = PENDING
    else:
        stats["n_trials"] = int(n_trials)
        stats["deflated_sharpe"] = deflated_sharpe(r, n_trials, risk_free=risk_free)
        stats["deflated_sharpe_sr0"] = _dsr_sr0(len(r), int(n_trials))

    stats["breakeven_tc_bps"] = (
        _breakeven_tc_bps(r, turnover) if turnover is not None else PENDING
    )

    stats.update(drawdown_duration(nav))

    stats["expectancy_per_rebalance"] = (
        float(monthly.mean() * 100) if len(monthly) else None
    )
    stats["tstat_newey_west"] = newey_west_tstat(r, risk_free=risk_free)

    stats["walkforward"] = walk_forward(r, risk_free=risk_free)

    if tnx is None or vix is None:
        stats["regimes"] = PENDING
    else:
        stats["regimes"] = regime_stats(r, tnx, vix, risk_free=risk_free)

    if factors is None:
        stats["factor_correlations"] = PENDING
        stats["factor_corr_note"] = PENDING
    else:
        stats["factor_correlations"] = factor_correlations(r, factors)
        stats["factor_corr_note"] = (
            "Factor proxies supplied by caller: "
            + ", ".join(str(k) for k in factors.keys())
            + ". Correlations are Pearson on daily returns; interpretation "
            "depends on how each proxy was constructed."
        )

    stats["monthly_pnl"] = monthly_pnl(r)
    stats["yearly_returns"] = yearly_returns(r)
    return stats


def _pct(x: float | None) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.2f}%"


def _num(x: float | None) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def _f2(x) -> str:
    """Format a stat that may be a float, None/NaN, or the PENDING literal."""
    if isinstance(x, str):
        return x
    return _num(x)


def _f2pct(x) -> str:
    """Format an already-in-percent stat (expectancy); PENDING passes through."""
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.2f}%"


def format_report(s: dict) -> str:
    """One-page text summary of ``report()`` output."""
    L = [
        f"=== {s['label']}  ({s['start']} -> {s['end']}, {s['n_years']}y) ===",
        f"Total return   {_pct(s['total_return'])}   CAGR {_pct(s['cagr'])}",
        f"Ann vol        {_pct(s['ann_vol'])}   Sharpe {_num(s['sharpe'])}   Sortino {_num(s['sortino'])}",
        f"Max drawdown   {_pct(s['max_drawdown'])}  ({s['dd_start']} -> {s['dd_trough']})   Calmar {_num(s['calmar'])}",
        f"Win rate       {_pct(s['win_rate_monthly'])} monthly / {_pct(s['win_rate_daily'])} daily",
        f"W/L ratio      {_num(s['win_loss_ratio_monthly'])} (monthly)   Profit factor {_num(s['profit_factor_monthly'])}",
        f"VaR 95 / 99    {_pct(s['var95_daily'])} / {_pct(s['var99_daily'])} (1-day, historical)",
    ]
    if "beta" in s:
        L.append(
            f"Beta {_num(s['beta'])}   Alpha(ann) {_pct(s['alpha_ann'])}   R2 {_num(s['r_squared'])}"
            f"   TE {_pct(s['tracking_error'])}   IR {_num(s['information_ratio'])}"
        )
    if s.get("avg_monthly_turnover") is not None:
        L.append(f"Avg monthly turnover  {_pct(s['avg_monthly_turnover'])} of NAV")
    L.append("--- robustness ---")
    L.append(
        f"DSR {_f2(s.get('deflated_sharpe'))}"
        f" (SR0 {_f2(s.get('deflated_sharpe_sr0'))}, n_trials={s.get('n_trials')})"
        f"   PSR(0) {_f2(s.get('psr'))}   NW t-stat {_f2(s.get('tstat_newey_west'))}"
    )
    L.append(f"Breakeven TC   {_f2(s.get('breakeven_tc_bps'))} bps one-way per unit turnover")
    L.append(
        f"CVaR 95 / 99   {_pct(s.get('cvar95_daily'))} / {_pct(s.get('cvar99_daily'))} (1-day)"
        f"   skew {_f2(s.get('skew_daily'))}   excess kurt {_f2(s.get('excess_kurtosis_daily'))}"
    )
    dur = s.get("dd_duration_days")
    ddline = (f"DD duration    {dur}d" if dur is not None else "DD duration    n/a")
    ddline += f"   status {s.get('dd_status')}"
    if s.get("dd_recovery_date"):
        ddline += f"   recovered {s['dd_recovery_date']}"
    if s.get("dd_status") == "still in DD":
        ddline += f"   ({s.get('dd_days_since_peak')}d since peak)"
    L.append(ddline)
    L.append(f"Expectancy     {_f2pct(s.get('expectancy_per_rebalance'))} per rebalance per $1 NAV")
    wf = s.get("walkforward") or {}
    L.append(
        f"Walk-forward   1H Sharpe {_f2(wf.get('first_half', {}).get('sharpe'))}"
        f" / 2H Sharpe {_f2(wf.get('second_half', {}).get('sharpe'))}"
    )
    rg = s.get("regimes")
    if isinstance(rg, dict):
        L.append(
            f"Regimes        rates rising {_f2(rg['rates']['rising']['sharpe'])}"
            f" vs falling {_f2(rg['rates']['falling']['sharpe'])};"
            f" VIX high {_f2(rg['volatility']['high']['sharpe'])}"
            f" vs low {_f2(rg['volatility']['low']['sharpe'])};"
            f" crisis {_f2(rg['crisis']['crisis']['sharpe'])}"
            f" vs calm {_f2(rg['crisis']['calm']['sharpe'])} (Sharpe)"
        )
    else:
        L.append(f"Regimes        {rg}")
    fc = s.get("factor_correlations")
    if isinstance(fc, dict):
        L.append("Factor corr    " + ", ".join(f"{k} {_f2(v)}" for k, v in fc.items()))
    else:
        L.append(f"Factor corr    {fc}")
    return "\n".join(L)
