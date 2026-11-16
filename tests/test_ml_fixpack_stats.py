"""Unit tests for the v2 fix-pack statistics helpers
(engine/ml_fixpack_stats.py; spec:
reports/ml_v2_fixpack_tests_spec_2026-10-24.md). Synthetic data only.
"""
import numpy as np

from engine import ml_fixpack_stats as S
from engine import ml_harness as H


def test_decluster_collapses_a_run_and_keeps_gap5():
    # Alerts on consecutive days collapse to one; a gap of exactly 5
    # full-panel days starts a new event; a gap of 4 does not.
    pos = np.array([100, 101, 102, 107, 111, 130])
    y = np.array([1, 1, 0, 1, 0, 1])
    out = S.decluster_alerts(pos, y, blackout=5)
    assert out["kept_positions"].tolist() == [100, 107, 130]
    assert out["n_events"] == 3
    assert out["hits"] == 3  # y at 100, 107, 130 are all 1
    assert out["hit_rate"] == 1.0


def test_decluster_single_and_empty_gap_cases():
    out = S.decluster_alerts(np.array([10, 20]), np.array([0, 1]))
    assert out["n_events"] == 2 and out["hits"] == 1
    out = S.decluster_alerts(np.array([10]), np.array([1]))
    assert out["n_events"] == 1 and out["hit_rate"] == 1.0


def test_binomial_matches_harness_alert_test():
    rng = np.random.default_rng(3)
    y = (rng.random(262) < 0.21).astype(int)
    p = rng.random(262)
    ref = H.alert_budget_test(y, p, null_rate=0.21)
    sel = S.alert_selection(y, p)
    assert sel["n_alert"] == ref["n_alert"] == 27
    assert sel["hits"] == ref["alert_hits"]
    assert S.binomial_greater_p(sel["hits"], sel["n_alert"], 0.21) == ref["binomial_p"]


def test_alert_selection_picks_highest_probs():
    y = np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 1])
    p = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.05])
    sel = S.alert_selection(y, p, budget=0.10)  # ceil(1.0) = 1 alert
    assert sel["n_alert"] == 1
    assert sel["alert_idx"].tolist() == [0]
    assert sel["hits"] == 0


def test_circular_block_indices_wrap_and_length():
    rng = np.random.default_rng(13)
    idx = S.circular_block_indices(25, 10, rng)
    assert len(idx) == 25
    assert idx.min() >= 0 and idx.max() < 25
    # Blocks are runs of 10 consecutive (wrapping) positions.
    for b in range(2):
        blk = idx[b * 10:(b + 1) * 10]
        assert ((np.diff(blk) % 25) == 1).all()


def test_block_bootstrap_deterministic_under_seed():
    a = S.block_bootstrap(50, 10, 20, seed=13)
    b = S.block_bootstrap(50, 10, 20, seed=13)
    assert a.shape == (20, 50)
    assert np.array_equal(a, b)
    c = S.block_bootstrap(50, 10, 20, seed=14)
    assert not np.array_equal(a, c)


def test_paired_brier_diffs_mean_equals_brier_difference():
    rng = np.random.default_rng(5)
    y = (rng.random(200) < 0.3).astype(int)
    pm = rng.random(200) * 0.6
    pc = np.full(200, 0.3)
    d = S.paired_brier_diffs(y, pm, pc)
    assert np.isclose(d.mean(), H.brier(y, pm) - H.brier(y, pc))
    # A perfect model has diff = -(climatology Brier) every day y=0/1 mix.
    d2 = S.paired_brier_diffs(y, y.astype(float), pc)
    assert np.isclose(d2.mean(), -H.brier(y, pc))


def test_percentile_interval_orders_bounds():
    lo, hi = S.percentile_interval(np.arange(101, dtype=float))
    assert lo == 2.5 and hi == 97.5
