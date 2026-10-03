"""Strategy 1 — "Value and Momentum Everywhere" (Asness, Moskowitz, Pedersen, JoF 2013).

Paper construction (multi-asset classes: equities, bonds, currencies, commodities):
  * VALUE: for non-equity classes, the negative of the past 5-year return
    (De Bondt-Thaler long-term reversal); for equities, book-to-market.
  * MOMENTUM: cumulative return over months t-12..t-2 ("12-1", Jegadeesh-Titman).
  * Long/short the extremes, 50/50 value/momentum combo, monthly rebalance.

Free-data ETF implementation (this file):
  * Universe: 11 liquid ETFs (5 equity, 4 rates, 2 credit).
  * VALUE_i(t) = -(48m cumulative return ending 12m before t), i.e. the 5y
    reversal window t-60..t-13. Skipping the most recent 12m keeps value
    orthogonal to the 12-1 momentum window (this is also what the paper's
    t-59..t-12 convention does).
  * MOMENTUM_i(t) = 11m cumulative return over t-12..t-2.
  * Cross-sectional z-score of each; combo score = z_val + z_mom (50/50).
  * Long top tercile / short bottom tercile, equal-weighted within each leg,
    $0.50 long / $0.50 short per $1 NAV (dollar-neutral).
  * Monthly rebalance. Signals use month-end closes; positions are dated one
    trading day later (documented execution lag — no lookahead).
  * The fixed-income sleeve (TLT/IEF/SHY/TIP/LQD/HYG) is built identically as
    a standalone long-short book and reported separately.

Deviations from the paper (documented honestly):
  * ETF proxies instead of futures/spot indices (no leverage, no carry/roll
    yield, includes ETF fees — ~5-15 bps/yr drag vs the paper's excess returns).
  * No book-to-market for equities (no free fundamentals feed in phase 1);
    the 5y-reversal value proxy is used for ALL assets, as the paper does for
    non-equity classes.
  * Shorting ETFs assumed feasible at ``borrow_bps_annual`` (config); real
    borrow can be wider for HYG in stress.
  * Universe is 11 ETFs, not the paper's 48 instruments across 4 classes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Month-end total-return panel -> monthly simple returns."""
    m = prices.resample("ME").last()
    return m.pct_change()


def _cumret(rets: pd.DataFrame, end_pos: int, window: int) -> pd.Series:
    """Cumulative simple return over ``window`` months ending ``end_pos``
    (positional, end-exclusive)."""
    return (1 + rets.iloc[end_pos - window:end_pos]).prod() - 1


def value_momentum_weights(
    prices: pd.DataFrame,
    tickers: list[str] | None = None,
    value_window: int = 48,
    value_skip: int = 12,
    mom_window: int = 11,
    mom_skip: int = 1,
    n_legs: int | None = None,
) -> pd.DataFrame:
    """Cross-sectional value+momentum long-short weights.

    Returns a DataFrame of target weights indexed by (month-end + 1 trading
    day) — the documented execution lag. Only rebalance dates have rows.
    """
    px = prices if tickers is None else prices[tickers]
    px = px.dropna(axis=1, how="all")
    tickers = list(px.columns)
    mret = monthly_returns(px)
    # need value_window + value_skip months of returns
    need = value_window + value_skip
    n_per_leg = n_legs or max(1, len(tickers) // 3)

    rows, dates = [], []
    for pos in range(need, len(mret)):
        date_t = mret.index[pos]
        # need full windows for every asset (no partial-history assets)
        win_v = mret.iloc[pos - need:pos - value_skip]
        win_m = mret.iloc[pos - mom_window - mom_skip:pos - mom_skip]
        if win_v.isna().any().any() or win_m.isna().any().any():
            continue
        value = -((1 + win_v).prod() - 1)
        momentum = (1 + win_m).prod() - 1
        z = lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else 1.0)  # noqa: E731
        combo = z(value) + z(momentum)
        ranks = combo.rank(method="first")
        n = len(combo)
        longs = ranks[ranks > n - n_per_leg].index
        shorts = ranks[ranks <= n_per_leg].index
        w = pd.Series(0.0, index=tickers)
        w[longs] = 0.5 / len(longs)
        w[shorts] = -0.5 / len(shorts)
        rows.append(w)
        dates.append(date_t)

    weights = pd.DataFrame(rows, index=pd.DatetimeIndex(dates))
    # execution lag: trade at the next trading day's close
    lagged_idx = [prices.index[prices.index > d][0] for d in weights.index
                  if (prices.index > d).any()]
    weights = weights.iloc[:len(lagged_idx)].copy()
    weights.index = pd.DatetimeIndex(lagged_idx)
    return weights
