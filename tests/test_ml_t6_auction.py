"""Tests for engine/ml_t6_auction.py (Stage C-1, proposal T6)."""
import numpy as np
import pandas as pd

from engine import ml_t6_auction as T6


def _auctions():
    return pd.DataFrame({
        "auction_date": pd.to_datetime(
            ["2026-01-05", "2026-01-05", "2026-01-06", "2026-01-07"]),
        "security_type": ["Bill", "Note", "Bond", "Bill"],
        "offering_amt": [100e9, 40e9, 30e9, 90e9],
    })


def test_coupon_calendar_counts_notes_and_bonds_only():
    cal = T6.coupon_calendar(_auctions())
    # 2026-01-05: one Note (40) + Bill excluded -> coupon day, amt 40
    assert cal.loc["2026-01-05", "coupon_day"] == 1
    assert cal.loc["2026-01-05", "coupon_amt"] == 40.0
    # 2026-01-06: one Bond (30)
    assert cal.loc["2026-01-06", "coupon_amt"] == 30.0
    # 2026-01-07: Bill only -> not a coupon date at all
    assert pd.Timestamp("2026-01-07") not in cal.index


def test_coupon_features_alignment_and_zeros():
    idx = pd.bdate_range("2026-01-05", "2026-01-08")
    out = T6.coupon_features_on(idx, _auctions())
    assert out.loc["2026-01-05", "coupon_day"] == 1
    assert out.loc["2026-01-07", "coupon_day"] == 0
    assert out.loc["2026-01-07", "coupon_amt"] == 0.0
    assert out.loc["2026-01-08", "coupon_day"] == 0


def test_auction_leg_window_is_strictly_forward():
    idx = pd.bdate_range("2026-01-05", periods=10)
    sample = pd.DataFrame({"full_pos": np.arange(10)}, index=idx)
    coupon_dates = pd.DatetimeIndex(["2026-01-06"])  # position 1
    a = T6.auction_leg(sample, idx, coupon_dates, horizon=5)
    # t at position 0: window positions 1..5 contains the coupon -> 1
    assert a[0] == 1
    # t at position 1 (the coupon day itself): window 2..6 -> 0
    assert a[1] == 0
    # t at position 5: window 6..10 -> 0
    assert a[5] == 0


def test_trailing_median_uses_no_future_data():
    s = pd.Series([1.0] * 130 + [100.0],
                  index=pd.bdate_range("2026-01-01", periods=131))
    med = T6.trailing_median(s)
    # median at the spike day includes the spike (window ends at t) but
    # the day before is unaffected by it
    assert med.iloc[-2] == 1.0
    assert med.iloc[-1] == 1.0  # 130 ones vs one 100 -> median still 1
