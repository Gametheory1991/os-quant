"""Tests for engine/ml_event_study.py — Phase 1 event rule and helpers.

The load-bearing property is point-in-time discipline: the event flag at
date t is computed from data dated <= t only, so truncating (or changing)
everything after t must not change the flag at or before t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.ml_event_study import (
    binomial_p_ge,
    decluster_events,
    event_flags,
    find_clusters,
    forward_delta,
    trailing_threshold,
)


def _series(values, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(np.asarray(values, dtype=float), index=idx)


def _long_series(n=400, seed=7):
    rng = np.random.default_rng(seed)
    return _series(rng.lognormal(mean=0.0, sigma=0.5, size=n), start="2022-01-03")


# ---------------------------------------------------------------------------
# Event rule: no future data
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cut", [130, 200, 252, 317, 399])
def test_truncation_does_not_change_earlier_flags(cut):
    """Flags from a truncated input equal full-sample flags up to the cut."""
    s = _long_series()
    full = event_flags(s)
    truncated = event_flags(s.iloc[:cut])
    pd.testing.assert_series_equal(truncated, full.iloc[:cut])


@pytest.mark.parametrize("cut", [150, 300])
def test_truncation_does_not_change_earlier_thresholds(cut):
    s = _long_series()
    full = trailing_threshold(s)
    truncated = trailing_threshold(s.iloc[:cut])
    pd.testing.assert_series_equal(truncated, full.iloc[:cut])


def test_mutating_the_future_does_not_change_earlier_flags():
    s = _long_series()
    full = event_flags(s)
    mutated = s.copy()
    mutated.iloc[250:] = mutated.iloc[250:] * 1000.0  # extreme future values
    mutated_flags = event_flags(mutated)
    pd.testing.assert_series_equal(mutated_flags.iloc[:250], full.iloc[:250])


# ---------------------------------------------------------------------------
# Event rule: semantics
# ---------------------------------------------------------------------------
def test_flags_na_until_min_periods():
    s = _series([1.0] * 130)
    flags = event_flags(s)  # window 252, min_periods 126
    assert flags.iloc[:125].isna().all()
    assert flags.iloc[125:].notna().all()


def test_flag_hand_computed_small_window():
    # window=5, min_periods=3, q=0.90
    s = _series([1.0, 2.0, 3.0, 4.0, 100.0])
    flags = event_flags(s, window=5, min_periods=3, quantile=0.90)
    assert flags.iloc[:2].isna().all()
    # trailing [1,2,3]: q90 = 2.8 -> 3 is an event
    assert bool(flags.iloc[2]) is True
    # trailing [1,2,3,4]: q90 = 3.7 -> 4 is an event
    assert bool(flags.iloc[3]) is True
    # trailing [1,2,3,4,100]: q90 = 61.6 -> 100 is an event
    assert bool(flags.iloc[4]) is True


def test_flag_false_when_below_trailing_quantile():
    s = _series([100.0, 1.0, 1.0, 1.0, 1.0])
    flags = event_flags(s, window=5, min_periods=3, quantile=0.90)
    # trailing [100,1,1]: q90 = 80.2 -> 1 is not an event
    assert bool(flags.iloc[2]) is False


def test_nan_illiq_is_unclassifiable_not_a_nonevent():
    values = np.ones(130)
    values[129] = np.nan
    flags = event_flags(_series(values))
    assert pd.isna(flags.iloc[129])


def test_threshold_is_trailing_not_full_sample():
    # A huge value late in the sample must not raise the early threshold.
    s = _series([1.0] * 130 + [1e6])
    thr = trailing_threshold(s)
    assert thr.iloc[129] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def test_decluster_keeps_first_of_each_cluster():
    trading = pd.bdate_range("2024-01-01", periods=40)
    events = trading[[0, 3, 10, 13, 14, 30]]  # gaps 3 | 7,3,1 | 16
    clusters = find_clusters(events, trading, min_gap=5)
    assert [len(c) for c in clusters] == [2, 3, 1]
    kept = decluster_events(events, trading, min_gap=5)
    assert list(kept) == [trading[0], trading[10], trading[30]]


def test_decluster_gap_of_exactly_five_starts_new_cluster():
    trading = pd.bdate_range("2024-01-01", periods=20)
    events = trading[[0, 5]]
    kept = decluster_events(events, trading, min_gap=5)
    assert list(kept) == [trading[0], trading[5]]


def test_decluster_no_events():
    trading = pd.bdate_range("2024-01-01", periods=10)
    assert decluster_events(pd.DatetimeIndex([]), trading).empty


# ---------------------------------------------------------------------------
# Binomial test + forward delta
# ---------------------------------------------------------------------------
def test_binomial_p_ge_exact_values():
    assert binomial_p_ge(0, 10, 0.5) == pytest.approx(1.0)
    assert binomial_p_ge(10, 10, 0.5) == pytest.approx(0.5**10)
    assert binomial_p_ge(9, 10, 0.5) == pytest.approx(11 / 1024)
    assert binomial_p_ge(5, 10, 0.0) == 0.0


def test_forward_delta_uses_oas_calendar_positions():
    oas = _series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    d = forward_delta(oas, horizon=5)
    assert d.iloc[0] == pytest.approx(5.0)
    assert d.iloc[1] == pytest.approx(5.0)
    assert d.iloc[2:].isna().all()
