"""Strategy 2 — FI rates time-series momentum (the "trend in bonds" trade).

This is the rates-desk staple: time-series momentum a la Moskowitz, Ooi and
Pedersen (2012), "Time Series Momentum" (JFE), applied to US duration.

Construction:
  * Signal each month-end: 12-1m *excess* return of long Treasuries (TLT)
    over cash (SHY). sign = +1 if positive, -1 if negative, 0 if exactly zero.
  * Position: volatility-targeted. w_t = sign * target_vol / sigma_60d,
    where sigma_60d is the annualized realized vol of the TLT/SHY excess
    return over the prior 60 trading days, capped at +/- ``max_leverage``.
    Default: 10% vol target, 1.5x cap.
  * Rebalanced monthly; signal dated with the same 1-trading-day execution
    lag as Strategy 1.
  * Benchmark: monthly-rebalanced 50/50 TLT/IEF total-return blend, the
    standard free-data proxy for the Bloomberg US Treasury index.

Why vol targeting: duration vol regimes differ enormously (2022 vs 2019);
sizing by inverse vol keeps the strategy's risk budget stationary, which is
what every FI quant desk does. The fixed-weight variant (always +/-100%) is
left in the code path via target_vol=None for comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fi_rates_momentum_weights(
    prices: pd.DataFrame,
    long_ticker: str = "TLT",
    cash_ticker: str = "SHY",
    target_vol: float | None = 0.10,
    max_leverage: float = 1.5,
    mom_window: int = 11,
    mom_skip: int = 1,
    vol_window_days: int = 60,
) -> pd.DataFrame:
    """Time-series momentum weights on duration.

    Returns target weights (single column ``long_ticker``) indexed by
    (month-end + 1 trading day). Positive = long duration, negative = short.
    """
    daily = prices[[long_ticker, cash_ticker]].dropna()
    excess_daily = daily[long_ticker].pct_change() - daily[cash_ticker].pct_change()
    month_ends = daily.resample("ME").last().index

    rows, dates = [], []
    for t in month_ends:
        hist = daily.loc[:t]
        if len(hist) < 252 + 30:  # need 12m signal + 60d vol
            continue
        # 12-1m excess return
        p_now_1m = hist[long_ticker].iloc[-22:].iloc[0] / hist[cash_ticker].iloc[-22:].iloc[0]
        p_now_12m = hist[long_ticker].iloc[-252:].iloc[0] / hist[cash_ticker].iloc[-252:].iloc[0]
        mom = p_now_1m / p_now_12m - 1
        sign = float(np.sign(mom))
        if target_vol is None:
            w = sign
        else:
            sigma = excess_daily.loc[:t].iloc[-vol_window_days:].std() * np.sqrt(252)
            w = sign * (target_vol / sigma if sigma > 0 else 0.0)
            w = float(np.clip(w, -max_leverage, max_leverage))
        rows.append(w)
        dates.append(t)

    weights = pd.DataFrame({long_ticker: rows}, index=pd.DatetimeIndex(dates))
    lagged_idx = [prices.index[prices.index > d][0] for d in weights.index
                  if (prices.index > d).any()]
    weights = weights.iloc[:len(lagged_idx)].copy()
    weights.index = pd.DatetimeIndex(lagged_idx)
    return weights


def treasury_benchmark(prices: pd.DataFrame, tickers: tuple[str, str] = ("TLT", "IEF")) -> pd.Series:
    """Monthly-rebalanced 50/50 blend total-return series (benchmark)."""
    px = prices[list(tickers)].dropna()
    mret = px.resample("ME").last().pct_change()
    blend_m = (mret[tickers[0]] + mret[tickers[1]]) / 2
    return (1 + blend_m).cumprod()
