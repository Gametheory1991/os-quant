"""Unit tests for engine.backtest — accounting verified by hand, no network."""
import numpy as np
import pandas as pd

from engine.backtest import run_backtest


def _prices():
    idx = pd.bdate_range("2020-01-01", periods=10)
    # asset A: +1% every day ; asset B: flat
    a = 100 * 1.01 ** np.arange(10)
    b = np.full(10, 50.0)
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def test_buy_and_hold_matches_asset_returns():
    px = _prices()
    w = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=[px.index[0]])
    out = run_backtest(px, w, costs_bps=0)
    # day 0: entry, no return yet (position set at close). From day 1 on:
    # strat == A's daily return, turnover only on entry day
    assert out["turnover"].iloc[0] == 1.0
    assert (out["turnover"].iloc[1:] == 0).all()
    assert abs(out["strat"].iloc[1] - 0.01) < 1e-12
    assert abs(out["nav"].iloc[-1] - (1.01 ** 9)) < 1e-9


def test_costs_charged_on_turnover():
    px = _prices()
    w = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=[px.index[0]])
    out = run_backtest(px, w, costs_bps=100)  # 1% per unit turnover
    # entry turnover = 1.0 -> cost 0.01 on day 0; strat day 0 = 0 - 0.01
    assert abs(out["cost"].iloc[0] - 0.01) < 1e-12
    assert abs(out["strat"].iloc[0] - (-0.01)) < 1e-12
    assert (out["cost"].iloc[1:] == 0).all()


def test_rebalance_turnover_and_drift():
    px = _prices()
    # day 0: 100% A. day 5: switch to 100% B.
    w = pd.DataFrame(
        {"A": [1.0, 0.0], "B": [0.0, 1.0]}, index=[px.index[0], px.index[5]]
    )
    out = run_backtest(px, w, costs_bps=0)
    # on day 5 we sell all of A (drifted weight ~1.0) and buy B: turnover ~2.0
    assert abs(out["turnover"].iloc[5] - 2.0) < 1e-6
    # after the switch, strategy is flat (B never moves)
    assert (out["strat"].iloc[6:].abs() < 1e-12).all()


def test_dollar_neutral_long_short():
    idx = pd.bdate_range("2020-01-01", periods=5)
    px = pd.DataFrame(
        {"A": [100, 101, 102, 103, 104], "B": [100, 99, 98, 97, 96]}, index=idx
    )
    w = pd.DataFrame({"A": [0.5], "B": [-0.5]}, index=[idx[0]])
    out = run_backtest(px, w, costs_bps=0)
    # long A (+~1%/d) short B (B falls ~1%/d -> short gains): both legs earn
    assert (out["strat"].iloc[1:] > 0).all()
    # borrow cost applies to the short leg
    out_b = run_backtest(px, w, costs_bps=0, borrow_bps_annual=10000)  # 100%/yr
    assert (out_b["borrow"].iloc[1:] > 0).all()  # day 0: no position held yet
    # borrow per day = 0.5 short notional * 100% / 365
    assert abs(out_b["borrow"].iloc[1] - 0.5 / 365) < 1e-12


def test_nan_price_is_untradeable():
    idx = pd.bdate_range("2020-01-01", periods=5)
    px = pd.DataFrame({"A": [100, 101, np.nan, np.nan, 104.0]}, index=idx)
    w = pd.DataFrame({"A": [1.0]}, index=[idx[0]])
    out = run_backtest(px, w, costs_bps=0)
    # NaN days contribute zero return and don't crash
    assert np.isfinite(out["strat"]).all()
    assert out["strat"].iloc[2] == 0.0 and out["strat"].iloc[3] == 0.0
