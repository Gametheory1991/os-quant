"""Institutional stats sheet for strategy evaluation.

``report(returns, benchmark=None, turnover=None, risk_free=0.0, label="")``
takes a daily net-return Series and produces the full dict of metrics plus a
formatted text block. All annualizations assume 252 trading days.

Definitions (documented so the numbers are auditable):
  * Sharpe/Sortino: mean/std of daily returns, annualized, vs ``risk_free``
    (annual, simple). Sortino uses downside deviation only.
  * Max drawdown: from the NAV curve; dates = trough date and the peak date
    the drawdown started from.
  * Win rate: fraction of positive-return months (and days, reported separately).
  * Win/loss ratio: mean positive monthly return / |mean negative monthly return|.
  * Profit factor: sum of positive monthly returns / |sum of negative|.
  * VaR 95/99: historical one-day VaR, quoted as a positive loss number.
  * Beta/alpha: OLS of strategy excess returns on benchmark excess returns;
    alpha annualized. Tracking error / information ratio vs the benchmark.
  * Turnover: mean monthly turnover (fraction of NAV traded), when supplied.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _ann_factor(periods_per_year: int = TRADING_DAYS) -> float:
    return float(periods_per_year)


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


def monthly_pnl(returns: pd.Series) -> pd.DataFrame:
    """Monthly P&L table: rows = years, columns = months, values = % return."""
    m = (1 + returns).resample("ME").prod() - 1
    tbl = m.groupby([m.index.year, m.index.month]).first().unstack()
    tbl.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return (tbl * 100).round(2)


def yearly_returns(returns: pd.Series) -> pd.Series:
    return ((1 + returns).resample("YE").prod() - 1) * 100


def report(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    turnover: pd.Series | None = None,
    risk_free: float = 0.0,
    label: str = "strategy",
) -> dict:
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

    stats["monthly_pnl"] = monthly_pnl(r)
    stats["yearly_returns"] = yearly_returns(r)
    return stats


def _pct(x: float | None) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.2f}%"


def _num(x: float | None) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


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
    return "\n".join(L)
