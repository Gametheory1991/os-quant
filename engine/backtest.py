"""Vectorized daily backtester.

Convention
----------
* ``prices``: DataFrame of total-return prices (adjusted closes), indexed by
  date, one column per asset. NaN = untradeable that day.
* ``weights``: DataFrame of *target* portfolio weights on rebalance dates,
  indexed by date, one column per asset. Rows on non-rebalance dates may be
  absent or NaN — positions are forward-filled (buy-and-hold between
  rebalances). Weights are fractions of NAV (e.g. 0.5 long / -0.5 short for a
  dollar-neutral book). They need not sum to anything in particular; residual
  cash earns zero by default (pass ``cash_returns`` to change that).

Accounting
-----------
For each day t:
    r_p,t = sum_i w_{i,t-1} * r_{i,t}            (holdings from yesterday)
    turnover_t = sum_i |w_{i,t} - w_drifted_{i,t}|
    cost_t = turnover_t * costs_bps / 1e4
    strat_t = r_p,t - cost_t
where w_drifted is yesterday's weight drifted by relative performance:
    w_drifted_{i,t} = w_{i,t-1} * (1 + r_{i,t}) / (1 + r_p,t)

Short positions pay an optional borrow cost: ``borrow_bps_annual`` applied to
the absolute short notional each day (actual/365). This matters for long-short
ETF implementations (Strategy 1).

NaN handling: an asset with a NaN price on day t is treated as untradeable —
its weight is set to 0 that day (no phantom marks) and its return contributes
nothing; when prices resume, the target weight is reinstated at the next valid
price with no fictitious re-entry cost. Weights explicitly set to NaN on a
rebalance date are treated as 0 (exit the position).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    costs_bps: float = 10.0,
    borrow_bps_annual: float = 0.0,
    cash_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Run the backtest. Returns a DataFrame with columns:
    ret (gross), cost, borrow, turnover, strat (net), nav, plus per-asset
    weights columns are NOT returned — see ``weights`` input for those."""
    prices = prices.sort_index()
    rets = prices.pct_change()

    # target weights on the price calendar; NaN target = 0 (exit)
    tgt = weights.reindex(prices.index)
    tgt = tgt.fillna(0.0)
    # hold positions between rebalances: on rebalance dates use the new
    # target row (NaN target = exit = 0); otherwise carry yesterday forward.
    rebalance_mask = weights.reindex(prices.index).notna().any(axis=1)
    held = tgt.where(rebalance_mask).ffill().fillna(0.0)
    w = held.copy()
    # assets with no price get no weight and no return
    valid = prices.notna()
    w = w.where(valid, 0.0)
    r = rets.fillna(0.0)

    # portfolio return from yesterday's holdings
    w_prev = w.shift(1).fillna(0.0)
    gross = (w_prev * r).sum(axis=1)

    # drifted weights just before today's rebalance
    denom = (1 + gross).replace(0, np.nan)
    w_drift = w_prev.mul(1 + r).div(denom, axis=0).fillna(0.0)

    # turnover only counts on rebalance dates (else target == drifted by construction)
    turnover = (w - w_drift).abs().sum(axis=1)
    turnover = turnover.where(rebalance_mask.reindex(prices.index, fill_value=False), 0.0)

    cost = turnover * costs_bps / 1e4
    short_notional = w_prev.clip(upper=0).abs().sum(axis=1)
    borrow = short_notional * borrow_bps_annual / 1e4 / 365.0

    strat = gross - cost - borrow
    if cash_returns is not None:
        cash_w = 1 - w_prev.abs().sum(axis=1).clip(upper=1)
        strat = strat + cash_w.clip(lower=0) * cash_returns.reindex(prices.index).fillna(0.0)

    nav = (1 + strat.fillna(0.0)).cumprod()
    out = pd.DataFrame(
        {
            "ret": gross,
            "cost": cost,
            "borrow": borrow,
            "turnover": turnover,
            "strat": strat,
            "nav": nav,
        },
        index=prices.index,
    )
    return out.dropna(subset=["strat"])
