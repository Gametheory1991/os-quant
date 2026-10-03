"""Unit tests for engine.stats — hand-computed fixtures, no network."""
import numpy as np
import pandas as pd

from engine.stats import max_drawdown, monthly_pnl, report


def _series(vals, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(vals))
    return pd.Series(vals, index=idx)


def test_max_drawdown_hand_computed():
    nav = _series([1.0, 1.1, 1.05, 1.2, 1.15])
    dd = max_drawdown(nav)
    assert abs(dd["max_drawdown"] - (1.05 / 1.1 - 1)) < 1e-12
    assert dd["dd_start"] == "2020-01-02"   # peak at 1.1
    assert dd["dd_trough"] == "2020-01-03"  # trough at 1.05


def test_cagr_and_total_return():
    # 252 days, each +0.1% -> total = 1.001^252 - 1, CAGR = 1.001^252 - 1
    r = _series([0.001] * 252)
    s = report(r)
    expected = 1.001 ** 252 - 1
    assert abs(s["total_return"] - expected) < 1e-9
    assert abs(s["cagr"] - expected) < 1e-9  # exactly 1 year
    assert s["max_drawdown"] == 0.0  # monotonic up


def test_sharpe_known_values():
    # alternating +2% / -1%: mean=0.005, std=0.015 (ddof=1 -> sample std)
    r = _series([0.02, -0.01] * 126)
    s = report(r)
    exp = (0.005 / r.std()) * np.sqrt(252)
    assert abs(s["sharpe"] - exp) < 1e-9


def test_beta_alpha_against_benchmark():
    # strategy = 2x benchmark exactly -> beta 2, alpha 0
    b = _series([0.01, -0.005, 0.02, -0.01] * 63)
    r = b * 2
    s = report(r, benchmark=b)
    assert abs(s["beta"] - 2.0) < 1e-9
    assert abs(s["alpha_ann"]) < 1e-9
    assert abs(s["r_squared"] - 1.0) < 1e-9


def test_var_is_positive_loss_number():
    r = _series(np.random.RandomState(42).normal(0.0005, 0.01, 500))
    s = report(r)
    assert s["var95_daily"] > 0 and s["var99_daily"] > 0
    assert s["var99_daily"] >= s["var95_daily"]


def test_win_loss_and_profit_factor():
    # 12 months: 9 up months (+~2.1%), 3 down months (~-3.1%)
    idx = pd.bdate_range("2020-01-01", periods=252)
    vals = [0.001] * (21 * 9) + [-0.0015] * (21 * 3)
    r = pd.Series(vals, index=idx)
    s = report(r)
    assert abs(s["win_rate_monthly"] - 0.75) < 1e-9
    assert s["win_loss_ratio_monthly"] > 0
    assert s["profit_factor_monthly"] > 0


def test_monthly_pnl_shape():
    r = _series([0.001] * 252, start="2021-01-01")
    tbl = monthly_pnl(r)
    assert tbl.shape[0] >= 1 and tbl.shape[1] == 12
    assert abs(tbl.iloc[0]["Jan"] - ((1.001 ** 21 - 1) * 100)) < 0.5  # ~21 trading days


def test_report_rejects_tiny_input():
    try:
        report(_series([0.01]))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
