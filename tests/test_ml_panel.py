"""Tests for engine/ml_panel.py — ML Phase 0 liquidity panel + feature store."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine import ml_panel as mp

T_CUT = pd.Timestamp("2024-06-30")  # a Sunday: no partial refunding week

# Floor for the standing panel: it only grows (daily updater), so tests
# check "at least this much, and every row comes from the source file"
# instead of an exact count that breaks on every legitimate extension.
PANEL_MIN_ROWS = 849               # 2026-11-08 weekly roll-up
PANEL_MIN_LAST_DAY = "2026-11-06"


def _is_stub_holiday(d: pd.Timestamp) -> bool:
    """SIFMA full-close days on which FINRA still publishes a near-empty
    daily Treasury file: Columbus Day (second Monday of October) and
    Veterans Day (Nov 11; observed Fri Nov 10 / Mon Nov 12 when it falls
    on a weekend)."""
    if d.month == 10 and d.weekday() == 0 and 8 <= d.day <= 14:
        return True
    if d.month == 11:
        obs = pd.Timestamp(d.year, 11, 11)
        if obs.weekday() == 5:
            obs -= pd.Timedelta(days=1)
        elif obs.weekday() == 6:
            obs += pd.Timedelta(days=1)
        return d.normalize() == obs
    return False


@pytest.fixture(scope="module")
def built():
    return mp.build_all()


@pytest.fixture(scope="module")
def panel(built):
    return built[0]


@pytest.fixture(scope="module")
def fs(built):
    return built[1]


@pytest.fixture(scope="module")
def integrity(built):
    return built[2]


# ---------------------------------------------------------------------------
# Panel shape / semantics
# ---------------------------------------------------------------------------
def test_panel_shape_and_range(panel):
    # Standing panel: grows via the daily updater (849 rows as of the
    # 2026-11-08 weekly roll-up; 834 as of the 2026-10-18 weekly
    # roll-up; 831 as of the 2026-10-13 standing-issuance unlock; was
    # 824 at the Phase 0 build).  Assert a floor plus one row per source
    # day, not an exact count, so a legitimate extension doesn't fail.
    assert len(panel) >= PANEL_MIN_ROWS
    assert len(panel) == len(mp.load_trace_daily())
    assert str(panel.index.min().date()) == "2023-02-14"
    assert str(panel.index.max().date()) >= PANEL_MIN_LAST_DAY
    assert panel.index.is_monotonic_increasing
    assert not panel.index.duplicated().any()
    # every row is a weekday (no weekend rows sneak in with an extension)
    assert (panel.index.weekday < 5).all()


def test_six_vector_semantics_known_day(fs):
    """2026-10-02, hand-checked against the source XLSX: the 6-vector is
    [ATS & Interdealer trades, ATS & Interdealer par $B, Dealer-to-Customer
    trades, Dealer-to-Customer par $B, Total trades, Total par $B]."""
    row = fs.loc["2026-10-02"]
    assert row["adt_nominal"] == 519760.0          # total trades, nominal
    assert row["adv_nominal"] == pytest.approx(1093.1)   # total par, $B
    assert row["adv_bills"] == pytest.approx(210.1)
    assert row["adt_agg"] == pytest.approx(611637.0, abs=700)
    # ATS & Interdealer share of computed aggregate par
    assert row["ats_share_agg"] == pytest.approx(721.7 / 1322.0, abs=0.002)
    # ticket size: $M per trade
    assert row["ticket_agg"] == pytest.approx(1322.0 * 1000 / 611637, rel=0.01)
    # on-the-run share of the <=2y bucket (onrun par 129.1 / total 167.3)
    assert row["onrun_share_le2"] == pytest.approx(129.1 / 167.3, abs=0.002)


def test_adv_is_in_billions(panel):
    """Volumes are $ billions (source header), so aggregate ADV is
    O(1000), not O(1) ($M misreading) or O(1e6)."""
    med = panel["adv_agg"].median()
    assert 500 < med < 2500


def test_reconciliation_within_tolerance(panel, integrity):
    assert integrity["recon_failures"] == []
    non_stub = panel[~panel["is_holiday_stub"]]
    assert (non_stub["recon_diff_pct"] <= 0.005).all()


def test_holiday_stub_days(panel):
    stub_idx = panel.index[panel["is_holiday_stub"]]
    stubs = [str(d.date()) for d in stub_idx]
    # the known stubs stay stubs ...
    assert {"2024-10-14", "2024-11-11", "2025-10-13", "2025-11-11",
            "2026-10-12"} <= set(stubs)
    # ... and any new stub must fall on Columbus or Veterans Day, so a
    # bad near-empty row on an ordinary day still fails here
    assert all(_is_stub_holiday(d) for d in stub_idx), stubs
    assert (panel.loc[panel["is_holiday_stub"], "adv_total_reported"] < 1.0).all()


def test_coverage_finding_recorded(integrity):
    # sparse Feb-Oct 2023 in the source file; dense (holidays only) after
    assert integrity["panel_days_by_year"]["2023"] == 132
    assert integrity["panel_days_by_year"]["2024"] == 252
    assert len(integrity["missing_weekdays_nonholiday"]) == 88
    assert max(integrity["missing_weekdays_nonholiday"]) <= "2023-10-31"
    assert integrity["duplicate_dates"] == 0


# ---------------------------------------------------------------------------
# ILLIQ construction
# ---------------------------------------------------------------------------
def test_illiq_formula(fs):
    row = fs.loc["2026-10-02"]
    assert row["illiq_le2"] == pytest.approx(
        abs(row["ret_shy"]) / row["adv_le2"], rel=1e-9)
    assert row["illiq_gt20"] == pytest.approx(
        abs(row["ret_tlt"]) / row["adv_gt20"], rel=1e-9)
    assert row["illiq_tips_le5"] == pytest.approx(
        abs(row["ret_tip"]) / row["adv_tips_le5"], rel=1e-9)


def test_illiq_nan_only_where_expected(fs):
    # NaN exactly on the holiday stubs (which grow by about two a year)
    # + 2025-01-09 (NYSE closed for the Carter national day of mourning;
    # bond market open, so no ETF return exists)
    nan_days = set(fs.index[fs["illiq_agg"].isna()])
    stub_days = set(fs.index[fs["is_holiday_stub"]])
    assert nan_days == stub_days | {pd.Timestamp("2025-01-09")}
    assert pd.isna(fs.loc["2025-01-09", "illiq_agg"])
    assert fs.loc[fs["is_holiday_stub"], "illiq_agg"].isna().all()


def test_illiq_agg_is_volume_weighted(fs):
    d = "2026-10-02"
    slugs = list(mp.NOMINAL_BUCKETS.values())
    num = sum(fs.loc[d, f"illiq_{s}"] * fs.loc[d, f"adv_{s}"] for s in slugs)
    den = sum(fs.loc[d, f"adv_{s}"] for s in slugs)
    assert fs.loc[d, "illiq_agg"] == pytest.approx(num / den, rel=1e-9)


# ---------------------------------------------------------------------------
# Lags / point-in-time behaviour of individual features
# ---------------------------------------------------------------------------
def test_oas_availability(fs):
    assert fs.loc[fs.index < "2023-10-16", "oas_hy"].isna().all()
    assert fs.loc[fs.index >= "2023-10-16", "oas_hy"].notna().all()


def test_dealer_publication_lag(fs):
    dealer = mp.load_dealer_tot()
    checked = 0
    for (w0, v0), (w1, v1) in zip(dealer, dealer[1:]):
        if v0 == v1:
            continue
        usable = w1 + pd.Timedelta(days=7)
        if usable <= fs.index.min() or w1 < fs.index.min():
            continue
        window = fs.loc[(fs.index >= w1) & (fs.index < usable), "dealer_net_tot"]
        if len(window) and (window == v0).all():
            day_after = fs.loc[fs.index >= usable, "dealer_net_tot"]
            if len(day_after) and day_after.iloc[0] == v1:
                checked += 1
        if checked >= 3:
            break
    assert checked >= 1


def test_corp_adv_publication_lag(fs):
    cdf = mp.monthly_corp_adv(mp.load_monthly())
    apr = cdf[cdf["month"] == "2024-04"].iloc[0]
    mar = cdf[cdf["month"] == "2024-03"].iloc[0]
    # April month-end (Apr 30) + 45d = Jun 14: before that, March's ADV
    assert fs.loc["2024-06-13", "corp_adv_lag"] == pytest.approx(mar["corp_adv"])
    assert fs.loc["2024-06-14", "corp_adv_lag"] == pytest.approx(apr["corp_adv"])
    # FINRA cross-check: Sep 2026 implied ADV = published average daily par
    chk = cdf[cdf["month"] == "2026-09"].iloc[0]
    assert chk["trading_days"] == 21
    assert chk["corp_adv"] * 1000 == pytest.approx(60196.1, abs=0.15)


def test_auction_features(fs):
    assert fs["auction_day"].sum() > 500
    assert fs["refunding_week"].sum() > 0
    # btc_last is the mean BTC of the most recent reported auction date
    a = mp.fetch_auctions()
    a = a.dropna(subset=["bid_to_cover_ratio"])
    last_day = a["auction_date"].dt.normalize().max()
    t = fs.index[fs.index >= last_day]
    if len(t):
        expected = a[a["auction_date"].dt.normalize() == last_day][
            "bid_to_cover_ratio"].mean()
        assert fs.loc[t[0], "btc_last"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Trailing-only helpers
# ---------------------------------------------------------------------------
def test_trailing_helpers_are_past_only():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    z = mp.trailing_zscore(s, window=10, min_periods=3)
    assert pd.isna(z.iloc[0]) and pd.isna(z.iloc[1])
    manual = (3.0 - np.mean([1.0, 2.0, 3.0])) / np.std([1.0, 2.0, 3.0], ddof=1)
    assert z.iloc[2] == pytest.approx(manual)
    v = mp.velocity(pd.Series([1.0, 1.0, 2.0, 2.0, 2.0]), lag=2,
                    window=10, min_periods=1)
    assert v.iloc[2] == pytest.approx(1.0)   # 1 -> 2 over 2 steps = +100%
    assert v.iloc[4] == pytest.approx(1.0)   # max over the trailing window


# ---------------------------------------------------------------------------
# THE point-in-time test: truncating every input at T must not change
# any feature value on rows <= T.
# ---------------------------------------------------------------------------
def test_point_in_time_truncation_invariance(built):
    _, fs_full, _ = built
    trace = mp.load_trace_daily()
    etfs = mp.load_etf_prices(["SHY", "IEF", "TLT", "TIP", "^TNX", "^VIX"])
    oas_ig = mp.fetch_fred_series("BAMLC0A0CM")
    oas_hy = mp.fetch_fred_series("BAMLH0A0HYM2")
    t10y2y = mp.fetch_fred_series("T10Y2Y")
    dealer = mp.load_dealer_tot()
    monthly = mp.load_monthly()
    auctions = mp.fetch_auctions()
    cut = str(T_CUT.date())
    _, fs_trunc, _ = mp.build_all(
        trace={d: v for d, v in trace.items() if d <= cut},
        etf_prices={t: s[s.index <= T_CUT] for t, s in etfs.items()},
        oas_ig=oas_ig[oas_ig.index <= T_CUT],
        oas_hy=oas_hy[oas_hy.index <= T_CUT],
        t10y2y=t10y2y[t10y2y.index <= T_CUT],
        dealer=[(d, v) for d, v in dealer if d <= T_CUT],
        monthly={ym: r for ym, r in monthly.items() if ym <= "2024-06"},
        auctions=auctions[auctions["auction_date"] <= T_CUT],
    )
    a = fs_full.loc[fs_full.index <= T_CUT]
    b = fs_trunc
    assert list(a.index) == list(b.index)
    pd.testing.assert_frame_equal(
        a.astype(float), b.astype(float), check_exact=False,
        rtol=1e-9, atol=1e-12)
