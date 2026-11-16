"""Unit tests for ML Stage C Target 2 Phase 1 (engine/ml_target2_auction).

All tests run on small synthetic frames with hand-computed expectations;
no model is fit anywhere in this suite. Coverage: point-in-time
trailing-median label construction (incl. strict prior-date rule and
same-day exclusion), weak-auction label, panel separation, tenor
mapping / reopening handling, refunding-boundary fold rule, embargo,
and the guarantee that no outcome-stamped field enters the features.
"""
import numpy as np
import pandas as pd
import pytest

from engine import ml_target2_auction as T2


def make_auctions(rows):
    """rows: (auction_date, security_type, security_term, offering_amt, btc, cusip)"""
    df = pd.DataFrame(rows, columns=["auction_date", "security_type",
                                     "security_term", "offering_amt",
                                     "bid_to_cover_ratio", "cusip"])
    df["auction_date"] = pd.to_datetime(df["auction_date"])
    df["issue_date"] = df["auction_date"] + pd.Timedelta(days=5)
    df = df.sort_values(["auction_date", "cusip"]).reset_index(drop=True)
    df["event_pos"] = np.arange(len(df))
    return df


def weekly_notes(n=10, start="2024-01-01", btc0=2.50, step=0.01,
                 term="3-Year", amt=50_000_000_000):
    """n same-tenor notes, 7 days apart, BTC rising by ``step``."""
    base = pd.Timestamp(start)
    return [(base + pd.Timedelta(days=7 * i), "Note", term, amt,
             round(btc0 + step * i, 2), f"C{i:03d}") for i in range(n)]


# ---------------------------------------------------------------------------
# Tenor mapping and panels
# ---------------------------------------------------------------------------
def test_tenor_mapping_covers_every_coupon_term_in_file():
    auctions = T2.load_auctions()
    panel = T2.coupon_panel(auctions)
    assert set(panel["tenor"]) == set(T2.COUPON_TENORS)
    assert len(panel) == 399  # Note 302 + Bond 97, dataset card §2


def test_reopening_terms_share_tenor_with_original():
    auctions = T2.load_auctions()
    panel = T2.coupon_panel(auctions)
    by_term = panel.groupby("security_term")["tenor"].first()
    assert by_term["9-Year 10-Month"] == by_term["10-Year"] == "10Y"
    assert by_term["4-Year 10-Month"] == by_term["5-Year"] == "5Y"
    assert by_term["29-Year 6-Month"] == by_term["30-Year"] == "30Y"
    assert by_term["1-Year 10-Month"] == by_term["2-Year"] == "2Y"


def test_panels_are_disjoint_and_bills_regular_only():
    auctions = T2.load_auctions()
    c = T2.coupon_panel(auctions)
    b = T2.bills_panel(auctions)
    assert set(c["security_type"]) == {"Note", "Bond"}
    assert set(b["security_type"]) == {"Bill"}
    assert set(b["security_term"]) <= set(T2.BILL_REGULAR_TERMS)
    assert len(b) == 1105  # dataset card §4 (regular terms only)
    assert len(c) + len(auctions[auctions["security_type"] == "Bill"]) \
        == len(auctions)


# ---------------------------------------------------------------------------
# Label construction — point-in-time trailing median of six
# ---------------------------------------------------------------------------
def test_benchmark_is_median_of_six_most_recent_prior():
    auctions = make_auctions(weekly_notes(8))
    panel = T2.coupon_panel(auctions)
    lab = T2.build_labels(panel)
    # First six auctions have < 6 priors: benchmark undefined.
    assert lab["bench_med6_label"].iloc[:6].isna().all()
    # 7th auction (index 6): priors are BTC 2.50..2.55 -> median 2.525.
    assert lab["bench_med6_label"].iloc[6] == pytest.approx(2.525)
    # 8th auction: priors' six most recent are 2.51..2.56 -> median 2.535.
    assert lab["bench_med6_label"].iloc[7] == pytest.approx(2.535)
    # Surprise = own BTC minus benchmark.
    assert lab["surprise"].iloc[7] == pytest.approx(2.57 - 2.535)


def test_benchmark_uses_strictly_prior_dates_only():
    rows = weekly_notes(6)
    d6 = pd.Timestamp("2024-01-01") + pd.Timedelta(days=42)
    # Two auctions on the same date: neither may enter the other's
    # benchmark; both see exactly the same six priors.
    rows += [(d6, "Note", "3-Year", 50_000_000_000, 9.99, "CX1"),
             (d6, "Note", "3-Year", 50_000_000_000, 1.01, "CX2")]
    auctions = make_auctions(rows)
    panel = T2.coupon_panel(auctions)
    lab = T2.build_labels(panel)
    same_day = lab.iloc[6:]
    expected = np.median([2.50, 2.51, 2.52, 2.53, 2.54, 2.55])
    assert same_day["bench_med6_label"].tolist() == \
        pytest.approx([expected, expected])


def test_benchmark_ignores_other_tenors():
    rows = weekly_notes(6, term="3-Year")
    rows += weekly_notes(6, term="7-Year", btc0=9.0, start="2024-01-03")
    rows.append((pd.Timestamp("2024-02-13"), "Note", "3-Year",
                 50_000_000_000, 2.60, "CTGT"))
    auctions = make_auctions(rows)
    panel = T2.coupon_panel(auctions)
    lab = T2.build_labels(panel)
    tgt = panel.index[panel["cusip"] == "CTGT"][0]
    pos = panel.index.get_loc(tgt)
    assert lab["bench_med6_label"].iloc[pos] == pytest.approx(2.525)


def test_missing_btc_priors_are_skipped_not_counted():
    rows = weekly_notes(7)
    rows[2] = (rows[2][0], "Note", "3-Year", 50_000_000_000, np.nan, "C002")
    rows.append((pd.Timestamp("2024-02-19"), "Note", "3-Year",
                 50_000_000_000, 2.60, "CNEW"))
    auctions = make_auctions(rows)
    panel = T2.coupon_panel(auctions)
    lab = T2.build_labels(panel)
    # Only 6 reported priors exist before CNEW (one of 7 is NaN), so
    # the benchmark is defined and equals their median.
    pos = panel.index.get_loc(panel.index[panel["cusip"] == "CNEW"][0])
    reported = [2.50, 2.51, 2.53, 2.54, 2.55, 2.56]
    assert lab["bench_med6_label"].iloc[pos] == pytest.approx(
        np.median(reported))


def test_weak_label_bottom_quintile_of_trailing_distribution():
    # 10 priors with BTC 2.0,2.1,...,2.9 -> 20th pct = 2.0+0.8*0.1*... :
    # quantile(0.20) linear = 2.18. BTC 2.10 -> weak; BTC 2.30 -> not.
    rows = [(pd.Timestamp("2024-01-01") + pd.Timedelta(days=7 * i),
             "Note", "5-Year", 50_000_000_000, round(2.0 + 0.1 * i, 2),
             f"W{i:02d}") for i in range(10)]
    rows.append((pd.Timestamp("2024-03-11"), "Note", "5-Year",
                 50_000_000_000, 2.10, "WLO"))
    rows.append((pd.Timestamp("2024-03-18"), "Note", "5-Year",
                 50_000_000_000, 2.30, "WHI"))
    auctions = make_auctions(rows)
    panel = T2.coupon_panel(auctions)
    lab = T2.build_labels(panel)
    lo = lab.iloc[panel.index.get_loc(
        panel.index[panel["cusip"] == "WLO"][0])]
    hi = lab.iloc[panel.index.get_loc(
        panel.index[panel["cusip"] == "WHI"][0])]
    assert lo["weak"] == 1.0
    assert hi["weak"] == 0.0


# ---------------------------------------------------------------------------
# Features — no outcome-stamped field may enter
# ---------------------------------------------------------------------------
def test_features_do_not_read_own_btc():
    auctions = make_auctions(weekly_notes(9))
    panel = T2.coupon_panel(auctions)
    X1 = T2.build_features(panel, auctions, "coupon")
    shocked = auctions.copy()
    shocked.loc[shocked.index[-1], "bid_to_cover_ratio"] = 99.0
    panel2 = T2.coupon_panel(shocked)
    X2 = T2.build_features(panel2, shocked, "coupon")
    pd.testing.assert_frame_equal(X1.iloc[:-1], X2.iloc[:-1])
    # The last row's features use only PRIOR auctions, so even its own
    # feature row is unchanged when its own BTC is altered.
    pd.testing.assert_series_equal(X1.iloc[-1], X2.iloc[-1])


def test_feature_columns_are_exactly_the_frozen_list():
    auctions = make_auctions(weekly_notes(9))
    panel = T2.coupon_panel(auctions)
    X = T2.build_features(panel, auctions, "coupon")
    assert list(X.columns) == T2.FEATURE_NAMES
    for forbidden in ("bid_to_cover_ratio", "btc", "surprise", "weak",
                      "refunding_week", "auction_day", "auction_amt"):
        assert forbidden not in X.columns


def test_demand_history_features_hand_computed():
    auctions = make_auctions(weekly_notes(8))
    panel = T2.coupon_panel(auctions)
    X = T2.build_features(panel, auctions, "coupon")
    last = X.iloc[7]
    assert last["bench_med6"] == pytest.approx(2.535)
    assert last["btc_last"] == pytest.approx(2.56)
    assert last["btc_gap_last"] == pytest.approx(2.56 - 2.535)
    assert last["days_since_same_tenor"] == 7
    assert last["is_reopening"] == 0.0


def test_reopening_flag_from_term_string():
    rows = weekly_notes(2, term="10-Year")
    rows.append((pd.Timestamp("2024-01-20"), "Note", "9-Year 11-Month",
                 30_000_000_000, 2.40, "ROP"))
    auctions = make_auctions(rows)
    panel = T2.coupon_panel(auctions)
    X = T2.build_features(panel, auctions, "coupon")
    pos = panel.index.get_loc(panel.index[panel["cusip"] == "ROP"][0])
    assert X["is_reopening"].iloc[pos] == 1.0


# ---------------------------------------------------------------------------
# Folds — event time, embargo, refunding-boundary rule
# ---------------------------------------------------------------------------
def _synthetic_sample(n_weeks=40, refunding_mondays=()):
    """One coupon auction per week on Wednesdays, plus flag Mondays."""
    wed = pd.date_range("2024-01-03", periods=n_weeks, freq="7D")
    df = pd.DataFrame({
        "auction_date": wed,
        "week_monday": wed - pd.to_timedelta(wed.weekday, unit="D"),
    })
    df["sample_pos"] = np.arange(len(df))
    refunding = {pd.Timestamp(m) for m in refunding_mondays}
    return df, refunding


def test_folds_expanding_with_embargo_gap():
    sample, refunding = _synthetic_sample(40)
    folds = T2.make_folds(sample, initial_train=15, test_block=10,
                          embargo_days=7, refunding=refunding)
    assert len(folds) >= 2
    for f in folds:
        tr_dates = sample["auction_date"].iloc[f.train_idx]
        te_dates = sample["auction_date"].iloc[f.test_idx]
        assert tr_dates.max() < te_dates.min()
        assert (te_dates.min() - tr_dates.max()).days > 7
        assert len(np.intersect1d(f.train_idx, f.test_idx)) == 0


def test_refunding_week_never_split_across_boundary():
    # Weeks start Mon 2024-01-01; Wednesdays indexed 0..39. Make the
    # week of index 15 a refunding week and place the raw boundary so
    # it would fall mid-week in a denser calendar: add a second
    # auction (Friday) inside that week.
    sample, _ = _synthetic_sample(40)
    monday15 = sample["week_monday"].iloc[15]
    extra = pd.DataFrame({
        "auction_date": [monday15 + pd.Timedelta(days=4)],
        "week_monday": [monday15],
    })
    sample = pd.concat([sample, extra], ignore_index=True) \
        .sort_values("auction_date").reset_index(drop=True)
    sample["sample_pos"] = np.arange(len(sample))
    refunding = {monday15}
    # Raw boundary at the Friday event (position of extra row): without
    # the snap, Wednesday of the refunding week trains, Friday tests.
    fri_pos = int(sample.index[sample["auction_date"]
                               == monday15 + pd.Timedelta(days=4)][0])
    folds = T2.make_folds(sample, initial_train=fri_pos, test_block=5,
                          embargo_days=3, refunding=refunding)
    f0 = folds[0]
    test_mondays = {pd.Timestamp(m)
                    for m in sample["week_monday"].iloc[f0.test_idx]}
    train_mondays = {pd.Timestamp(m)
                     for m in sample["week_monday"].iloc[f0.train_idx]}
    assert monday15 in test_mondays
    assert monday15 not in train_mondays


def test_snap_suppressed_when_week_already_in_prior_test_block():
    # Synthetic reproduction of the Amendment A1 boundary case (the
    # 2025-08-04 refunding week at the fold boundary, in miniature):
    # a refunding week with three events; the first event falls in
    # the EARLIER fold's test block and the raw boundary of the next
    # fold falls on the week's second event. The retreat must NOT
    # re-pull the already-tested event: the block starts at the raw
    # boundary, every position is tested exactly once, and the week
    # is absent from the later fold's training set (the embargo
    # removes the week's earlier event, which is 1 day before the
    # first test date).
    mon = pd.Timestamp("2024-02-05")  # refunding Monday
    dates = [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-10"),
             pd.Timestamp("2024-01-17"), pd.Timestamp("2024-01-24"),
             pd.Timestamp("2024-01-31"),
             mon + pd.Timedelta(days=1),   # pos 5: Tue, refunding week
             mon + pd.Timedelta(days=2),   # pos 6: Wed, refunding week
             mon + pd.Timedelta(days=3),   # pos 7: Thu, refunding week
             pd.Timestamp("2024-02-14"), pd.Timestamp("2024-02-21"),
             pd.Timestamp("2024-02-28"), pd.Timestamp("2024-03-06")]
    sample = pd.DataFrame({
        "auction_date": dates,
        "week_monday": [d - pd.Timedelta(days=d.weekday()) for d in dates],
    })
    sample["sample_pos"] = np.arange(len(sample))
    folds = T2.make_folds(sample, initial_train=4, test_block=2,
                          embargo_days=7, refunding={mon})
    # Cross-fold uniqueness over the whole assignment.
    assert T2.assert_cross_fold_test_uniqueness(folds) == \
        sum(len(f.test_idx) for f in folds)
    f0, f1 = folds[0], folds[1]
    assert f0.test_idx.tolist() == [4, 5]        # pos 5 tested here
    assert f1.test_idx.tolist() == [6, 7]        # NOT re-pulled
    assert f1.meta["snap_suppressed"] is True
    train_mondays = {pd.Timestamp(m)
                     for m in sample["week_monday"].iloc[f1.train_idx]}
    assert mon not in train_mondays


def test_cross_fold_uniqueness_assertion_fires_on_duplicate():
    sample, _ = _synthetic_sample(12)
    dup = [T2.Fold(fold_id=0, train_idx=np.arange(0, 3),
                   test_idx=np.arange(3, 6)),
           T2.Fold(fold_id=1, train_idx=np.arange(0, 5),
                   test_idx=np.arange(5, 8))]
    with pytest.raises(AssertionError):
        T2.assert_cross_fold_test_uniqueness(dup)


def test_fold_assertion_fires_on_refunding_split():
    # Daily rows; rows 4 and 5 share the refunding Monday. Putting row
    # 4 in train and row 5 in test splits that week and must raise.
    days = pd.date_range("2024-01-01", periods=10, freq="D")  # Mon..Wed
    sample = pd.DataFrame({
        "auction_date": days,
        "week_monday": days - pd.to_timedelta(days.weekday, unit="D"),
    })
    sample["sample_pos"] = np.arange(len(sample))
    refunding_monday = sample["week_monday"].iloc[4]
    assert sample["week_monday"].iloc[5] == refunding_monday
    with pytest.raises(AssertionError):
        T2._assert_fold_integrity(
            sample, np.arange(0, 5), np.arange(5, 9), 0,
            {refunding_monday})


# ---------------------------------------------------------------------------
# Real-file sample accounting (dataset card numbers)
# ---------------------------------------------------------------------------
def test_real_sample_sizes_match_dataset_card():
    auctions = T2.load_auctions()
    assert len(auctions) == 1621
    c = T2.coupon_panel(auctions)
    b = T2.bills_panel(auctions)
    lab_c = T2.build_labels(c)
    labelled_c = (lab_c["btc"].notna() & lab_c["bench_med6_label"].notna())
    assert int(labelled_c.sum()) == 357
    lab_b = T2.build_labels(b)
    labelled_b = (lab_b["btc"].notna() & lab_b["bench_med6_label"].notna())
    assert int(labelled_b.sum()) == 1060
    mondays = T2.refunding_mondays(auctions)
    assert len(mondays) == 15


def test_real_fold_boundary_2025_08_04_no_double_count():
    # The exact gate case (Amendment A1): the refunding week of
    # Monday 2025-08-04 holds sample positions 228 (2025-08-05,
    # 3-Year, CUSIP 91282CNU1), 229 (2025-08-06, 10-Year, CUSIP
    # 91282CNT4) and 230 (2025-08-07, 30-Year). The block ending at
    # 229 already tests 228 and 229; the next fold's raw boundary at
    # 230 must not re-pull them.
    auctions = T2.load_auctions()
    refunding = T2.refunding_mondays(auctions)
    assert pd.Timestamp("2025-08-04") in refunding
    sample = T2.build_sample(T2.coupon_panel(auctions), auctions,
                             "coupon")
    r228, r229, r230 = (sample.iloc[p] for p in (228, 229, 230))
    assert (str(r228["auction_date"].date()), r228["cusip"]) == \
        ("2025-08-05", "91282CNU1")
    assert (str(r229["auction_date"].date()), r229["cusip"]) == \
        ("2025-08-06", "91282CNT4")
    assert str(r230["auction_date"].date()) == "2025-08-07"
    folds = T2.make_folds(sample, initial_train=150, test_block=40,
                          embargo_days=7, refunding=refunding)
    owner = {}
    for f in folds:
        for p in f.test_idx.tolist():
            assert p not in owner, f"position {p} tested twice"
            owner[p] = f.fold_id
    assert owner[228] == owner[229] == 1     # the 190–229 block only
    assert owner[230] == 2
    f2 = folds[2]
    assert f2.test_idx[0] == 230 and len(f2.test_idx) == 40
    assert f2.meta["snap_suppressed"] is True
    train_mondays = {pd.Timestamp(m)
                     for m in sample["week_monday"].iloc[f2.train_idx]}
    assert pd.Timestamp("2025-08-04") not in train_mondays
    # Restated pooled counts (Amendment A1): 207 distinct test
    # auctions and 48 distinct test weak events (not 209 / 49).
    assert T2.assert_cross_fold_test_uniqueness(folds) == 207
    assert sum(len(f.test_idx) for f in folds) == 207
    assert int(sum(sample["weak"].iloc[f.test_idx].sum()
                   for f in folds)) == 48
    # The same suppression fires at the next shifted boundary: the
    # refunding week of Monday 2026-05-11 (position 309 already
    # tested by the preceding block; the next block starts at 310).
    assert owner[309] == 3 and owner[310] == 4
    assert folds[4].meta["snap_suppressed"] is True


def test_real_bills_folds_have_no_cross_fold_duplication():
    auctions = T2.load_auctions()
    refunding = T2.refunding_mondays(auctions)
    sample = T2.build_sample(T2.bills_panel(auctions), auctions,
                             "bills")
    folds = T2.make_folds(sample, initial_train=400, test_block=100,
                          embargo_days=7, refunding=refunding)
    assert T2.assert_cross_fold_test_uniqueness(folds) == 660
    assert int(sum(sample["weak"].iloc[f.test_idx].sum()
                   for f in folds)) == 158
