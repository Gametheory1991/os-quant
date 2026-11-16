"""ML Stage C — Target 2: Treasury auction bid-to-cover (BTC) prediction.

Phase 1 build (field audit + frozen specification + implementation; NO
fitting in this phase). Frozen spec:
``reports/ml_target2_experiment_spec_2026-10-27.md``; dataset card:
``reports/ml_target2_dataset_card_2026-10-27.md``.

This module provides ONLY:

- loading of ``data/ml/auctions.csv`` with the frozen tenor mapping;
- panel separation (coupon primary panel; bills secondary panel,
  regular terms only; the two panels are never pooled);
- point-in-time label construction (T2a BTC level, T2b BTC surprise
  against the median of the six most recent prior same-tenor auctions,
  T2c weak-auction indicator against the bottom quintile of the
  trailing same-tenor distribution);
- point-in-time feature construction (every feature for auction t is
  computed from auctions dated strictly before t, plus the announced
  schedule quantities of date t itself: this auction's announced
  offering amount and the announced same-day coupon totals);
- the event-time walk-forward harness (expanding train, blocked test,
  calendar embargo, refunding-week boundary rule) with hard assertions.

It contains NO model fitting. The model ladder (ridge / logistic /
gradient boosting) is reused unmodified from ``engine/ml_models.py``
by the separate backtest phase; nothing in this module calls it.

Point-in-time rules enforced here:

- "Prior" always means ``auction_date`` STRICTLY before the auction's
  own date. Auctions sharing a date never enter each other's
  benchmarks, features, or labels, because the file carries no
  intraday ordering.
- The auction's own ``bid_to_cover_ratio`` is label-side only. The
  feature builder never reads the current row's BTC (a unit test
  permutes it and requires identical features).
- The stored ``refunding_week`` column of the feature store is never
  read. Refunding weeks are identified ex post from the realised
  calendar ONLY to partition folds (sample splitting, never a
  feature), exactly as frozen in the specification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "data" / "ml"
AUCTIONS_CSV = ML_DIR / "auctions.csv"

SEED = 13  # program convention (engine/ml_harness.SEED); unused in Phase 1

# ---------------------------------------------------------------------------
# Frozen tenor taxonomy (spec §3). ``security_term`` in the persisted file
# is the REMAINING maturity at auction, so a reopening carries a shorter
# term string than its original auction. The tenor group is the ORIGINAL
# maturity bucket; membership is validated against CUSIP chains in the
# dataset card (a reopening shares its CUSIP with the original auction).
# The file carries no field separating nominal notes/bonds from Treasury
# Inflation-Protected Securities (TIPS) or Floating Rate Notes (FRN), so
# each tenor group pools those instruments; that limitation is carried
# in the card and the spec and is not repaired by any proxy here.
# ---------------------------------------------------------------------------
COUPON_TENOR_MAP = {
    "2-Year": "2Y",
    "1-Year 11-Month": "2Y",   # 2Y reopening (incl. 2Y FRN reopenings)
    "1-Year 10-Month": "2Y",   # 2Y reopening (incl. 2Y FRN reopenings)
    "3-Year": "3Y",
    "5-Year": "5Y",
    "4-Year 10-Month": "5Y",   # 5Y reopening (CUSIP-verified)
    "7-Year": "7Y",
    "10-Year": "10Y",
    "9-Year 11-Month": "10Y",
    "9-Year 10-Month": "10Y",
    "9-Year 8-Month": "10Y",
    "20-Year": "20Y",
    "19-Year 11-Month": "20Y",
    "19-Year 10-Month": "20Y",
    "30-Year": "30Y",
    "29-Year 11-Month": "30Y",
    "29-Year 10-Month": "30Y",
    "29-Year 6-Month": "30Y",
}
COUPON_TENORS = ["2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
# Original-issue term string per tenor (a row whose security_term equals
# this string is a new issue; any other member string is a reopening).
ORIGINAL_TERM = {
    "2Y": "2-Year", "3Y": "3-Year", "5Y": "5-Year", "7Y": "7-Year",
    "10Y": "10-Year", "20Y": "20-Year", "30Y": "30-Year",
}
# Bills panel: the seven standing weekly/monthly auction terms only.
# Every other bill term is a cash-management or one-off bill with no
# stable same-term history (dataset card §4) and is excluded from the
# bills modelling panel.
BILL_REGULAR_TERMS = ["4-Week", "6-Week", "8-Week", "13-Week",
                      "17-Week", "26-Week", "52-Week"]

BENCH_N = 6        # trailing same-tenor auctions in the T2b benchmark
WEAK_WINDOW = 24   # trailing same-tenor auctions in the T2c reference set
WEAK_QUANTILE = 0.20

# Feature columns, in frozen order (spec §5). Label columns are separate
# and never appear here.
FEATURE_NAMES = [
    "bench_med6",          # median BTC of 6 most recent prior same-tenor
    "btc_last",            # BTC of most recent prior same-tenor auction
    "btc_gap_last",        # btc_last minus bench_med6
    "btc_std6",            # std of the same 6 prior BTCs
    "log_offering_amt",    # log of this auction's announced amount ($)
    "offering_amt_chg",    # amount vs previous same-tenor amount (ratio-1)
    "is_reopening",        # 1 if security_term is a reopening term string
    "days_since_same_tenor",
    "days_since_any_coupon_or_bill",  # panel-specific; see build_features
    "coupon_day_context",  # bills: stored-form coupon_day; coupons: same-day other-coupon flag
    "coupon_amt_day",      # announced coupon amount dated t ($B)
    "n_same_day_panel",    # announced auctions of this panel dated t
]
LABEL_NAMES = ["btc", "bench_med6_label", "surprise", "weak"]


# ---------------------------------------------------------------------------
# Loading and panels
# ---------------------------------------------------------------------------
def load_auctions(path: Path | None = None) -> pd.DataFrame:
    """Load auctions.csv sorted in event order.

    Event order is (auction_date, cusip): the file carries no intraday
    timestamp, so the CUSIP is the deterministic tie-break for auctions
    sharing a date. The sort never affects point-in-time content, which
    is governed by strict date inequalities throughout this module.
    """
    df = pd.read_csv(path or AUCTIONS_CSV,
                     parse_dates=["auction_date", "issue_date"])
    df["auction_date"] = df["auction_date"].dt.normalize()
    df["issue_date"] = df["issue_date"].dt.normalize()
    df["offering_amt"] = pd.to_numeric(df["offering_amt"], errors="coerce")
    df["bid_to_cover_ratio"] = pd.to_numeric(df["bid_to_cover_ratio"],
                                             errors="coerce")
    df = df.sort_values(["auction_date", "cusip"]).reset_index(drop=True)
    df["event_pos"] = np.arange(len(df))
    return df


def coupon_panel(auctions: pd.DataFrame) -> pd.DataFrame:
    """Primary panel: Note and Bond records, with tenor group attached."""
    c = auctions[auctions["security_type"].isin(["Note", "Bond"])].copy()
    c["tenor"] = c["security_term"].map(COUPON_TENOR_MAP)
    assert c["tenor"].notna().all(), (
        "unmapped coupon security_term: "
        f"{sorted(set(c.loc[c['tenor'].isna(), 'security_term']))}")
    c["group"] = c["tenor"]
    return c.reset_index(drop=True)


def bills_panel(auctions: pd.DataFrame) -> pd.DataFrame:
    """Secondary panel: Bill records in the seven regular terms only."""
    b = auctions[(auctions["security_type"] == "Bill")
                 & (auctions["security_term"].isin(BILL_REGULAR_TERMS))]
    b = b.copy()
    b["group"] = b["security_term"]
    return b.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Point-in-time trailing statistics (strictly prior dates only)
# ---------------------------------------------------------------------------
def _prior_rows(panel: pd.DataFrame, row: pd.Series,
                group_col: str = "group") -> pd.DataFrame:
    """All panel rows in the same group dated strictly before ``row``."""
    m = (panel[group_col] == row[group_col]) \
        & (panel["auction_date"] < row["auction_date"])
    return panel.loc[m].sort_values(["auction_date", "cusip"])


def trailing_benchmark(panel: pd.DataFrame, row: pd.Series) -> float:
    """Median BTC of the BENCH_N most recent prior same-group auctions.

    Uses only auctions dated strictly before the row's date and only
    their reported (non-null) BTC values. NaN until BENCH_N reported
    prior BTCs exist.
    """
    prior = _prior_rows(panel, row)["bid_to_cover_ratio"].dropna()
    if len(prior) < BENCH_N:
        return float("nan")
    return float(prior.tail(BENCH_N).median())


def trailing_quantile(panel: pd.DataFrame, row: pd.Series,
                      window: int = WEAK_WINDOW,
                      q: float = WEAK_QUANTILE) -> float:
    """q-quantile of up to ``window`` most recent prior same-group BTCs."""
    prior = _prior_rows(panel, row)["bid_to_cover_ratio"].dropna()
    if len(prior) < BENCH_N:
        return float("nan")
    return float(prior.tail(window).quantile(q))


# ---------------------------------------------------------------------------
# Labels (label side only; never features)
# ---------------------------------------------------------------------------
def build_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """T2a / T2b / T2c labels for every panel row, indexed like ``panel``.

    - ``btc`` (T2a): the auction's realised bid-to-cover ratio.
    - ``bench_med6_label``: the point-in-time benchmark (median of the
      six most recent prior same-group BTCs, strictly prior dates).
    - ``surprise`` (T2b): ``btc - bench_med6_label``.
    - ``weak`` (T2c): 1 if ``btc`` is at or below the 20th percentile
      of the trailing same-group BTC distribution (up to 24 most
      recent prior reported BTCs; benchmark defined only once at
      least 6 prior reported BTCs exist), else 0.

    A row is label-complete only when its own BTC is reported and its
    benchmark is defined.
    """
    out = pd.DataFrame(index=panel.index)
    bench = np.full(len(panel), np.nan)
    weak = np.full(len(panel), np.nan)
    for i, (_, row) in enumerate(panel.iterrows()):
        bench[i] = trailing_benchmark(panel, row)
        thr = trailing_quantile(panel, row)
        btc = row["bid_to_cover_ratio"]
        if np.isfinite(thr) and pd.notna(btc):
            weak[i] = float(btc <= thr)
    out["btc"] = panel["bid_to_cover_ratio"].to_numpy(dtype=float)
    out["bench_med6_label"] = bench
    out["surprise"] = out["btc"] - out["bench_med6_label"]
    out["weak"] = weak
    return out


# ---------------------------------------------------------------------------
# Features (knowable before the auction result at t)
# ---------------------------------------------------------------------------
def _coupon_calendar(auctions: pd.DataFrame) -> pd.DataFrame:
    """Announced coupon totals per date, T6 definitional form.

    Coupon := security_type in {Note, Bond}; amounts are the summed
    offering amounts of coupon auctions dated exactly that date. In
    Target 2 these are used on the announced-schedule basis frozen in
    the specification (auction amounts are announced before the
    auction date); no auction dated after t contributes.
    """
    c = auctions[auctions["security_type"].isin(["Note", "Bond"])]
    g = c.groupby("auction_date")["offering_amt"].agg(["size", "sum"])
    out = pd.DataFrame(index=g.index)
    out["n_coupon"] = g["size"].to_numpy(dtype=float)
    out["coupon_amt"] = g["sum"].to_numpy(dtype=float) / 1e9
    return out


def build_features(panel: pd.DataFrame, auctions: pd.DataFrame,
                   panel_kind: str) -> pd.DataFrame:
    """Frozen feature matrix for one panel, indexed like ``panel``.

    ``panel_kind`` is "coupon" or "bills". Demand-history features use
    only same-group auctions dated strictly before t. Context features
    use the announced schedule of date t (this auction's announced
    amount; announced same-day totals). The current row's BTC is never
    read: ``bid_to_cover_ratio`` is accessed only on strictly prior
    rows.
    """
    assert panel_kind in ("coupon", "bills")
    cal = _coupon_calendar(auctions)
    dates = auctions["auction_date"]
    rows: list[dict] = []
    for _, row in panel.iterrows():
        t = row["auction_date"]
        prior = _prior_rows(panel, row)
        prior_btc = prior["bid_to_cover_ratio"].dropna()
        last6 = prior_btc.tail(BENCH_N)
        bench = float(last6.median()) if len(last6) == BENCH_N else np.nan
        btc_last = float(prior_btc.iloc[-1]) if len(prior_btc) else np.nan
        btc_std6 = (float(last6.std(ddof=1)) if len(last6) == BENCH_N
                    else np.nan)
        prev_amt = (float(prior["offering_amt"].iloc[-1])
                    if len(prior) else np.nan)
        # Announced same-day context (date t schedule quantities).
        day_coupon_amt = (float(cal["coupon_amt"].loc[t])
                          if t in cal.index else 0.0)
        day_n_coupon = (float(cal["n_coupon"].loc[t])
                        if t in cal.index else 0.0)
        if panel_kind == "coupon":
            coupon_day_context = float(day_n_coupon > 1)
            prior_dates = panel.loc[panel["auction_date"] < t,
                                    "auction_date"]
        else:
            coupon_day_context = float(day_n_coupon >= 1)
            prior_dates = dates[dates < t]
        days_since_any = (float((t - prior_dates.max()).days)
                          if len(prior_dates) else np.nan)
        rows.append({
            "bench_med6": bench,
            "btc_last": btc_last,
            "btc_gap_last": (btc_last - bench
                             if np.isfinite(btc_last) and np.isfinite(bench)
                             else np.nan),
            "btc_std6": btc_std6,
            "log_offering_amt": float(np.log(row["offering_amt"])),
            "offering_amt_chg": (float(row["offering_amt"] / prev_amt - 1.0)
                                 if np.isfinite(prev_amt) and prev_amt > 0
                                 else np.nan),
            "is_reopening": float(
                row["security_term"] != ORIGINAL_TERM.get(row["group"],
                                                          row["group"])
                if panel_kind == "coupon" else 0.0),
            "days_since_same_tenor": (
                float((t - prior["auction_date"].max()).days)
                if len(prior) else np.nan),
            "days_since_any_coupon_or_bill": days_since_any,
            "coupon_day_context": coupon_day_context,
            "coupon_amt_day": day_coupon_amt,
            "n_same_day_panel": float(
                (panel["auction_date"] == t).sum()),
        })
    X = pd.DataFrame(rows, index=panel.index)[FEATURE_NAMES]
    return X


def build_sample(panel: pd.DataFrame, auctions: pd.DataFrame,
                 panel_kind: str) -> pd.DataFrame:
    """Label-complete modelling sample for one panel.

    Keeps rows whose labels are defined (own BTC reported; benchmark
    defined) and whose features are all non-NaN. Carries ``event_pos``
    (position inside the panel, event time) and ``week_monday`` (the
    Monday of the auction's calendar week, used only by the fold
    rule). Panels are built and sampled separately; there is no code
    path that pools them.
    """
    X = build_features(panel, auctions, panel_kind)
    lab = build_labels(panel)
    df = panel[["auction_date", "security_type", "security_term",
                "group", "cusip", "offering_amt"]].copy()
    df = df.join(X).join(lab)
    # Panels are reset-indexed at construction, so each row's index IS
    # its position inside the parent panel (event time within panel).
    df["event_pos"] = df.index.to_numpy(dtype=int)
    df["week_monday"] = (df["auction_date"]
                         - pd.to_timedelta(df["auction_date"].dt.weekday,
                                            unit="D"))
    keep = (df["btc"].notna() & df["bench_med6_label"].notna()
            & df["weak"].notna() & df[FEATURE_NAMES].notna().all(axis=1))
    sample = df.loc[keep].copy()
    sample["sample_pos"] = np.arange(len(sample))
    return sample


# ---------------------------------------------------------------------------
# Refunding weeks (fold partitioning only — never a feature)
# ---------------------------------------------------------------------------
def refunding_mondays(auctions: pd.DataFrame) -> set[pd.Timestamp]:
    """Monday of each realised refunding week in the auction file.

    A refunding week is a Monday–Sunday week whose coupon auctions
    include a new 3-Year note, a new 10-Year note, and a new 30-Year
    bond (exact original term strings) — the quarterly refunding
    pattern, identical in form to the stored construction audited in
    Stage C-1. This identification is ex post and is used ONLY to keep
    a refunding week on one side of any train/test boundary.
    """
    a = auctions[auctions["security_type"].isin(["Note", "Bond"])].copy()
    a["monday"] = (a["auction_date"]
                   - pd.to_timedelta(a["auction_date"].dt.weekday, unit="D"))
    terms = a.groupby("monday")["security_term"].agg(
        lambda s: set(s.dropna()))
    return {m for m, ts in terms.items()
            if {"3-Year", "10-Year", "30-Year"} <= ts}


# ---------------------------------------------------------------------------
# Walk-forward folds in event time (spec §7)
# ---------------------------------------------------------------------------
@dataclass
class Fold:
    fold_id: int
    train_idx: np.ndarray  # sample positions
    test_idx: np.ndarray   # sample positions
    meta: dict = field(default_factory=dict)


def make_folds(sample: pd.DataFrame, initial_train: int,
               test_block: int, embargo_days: int = 7,
               refunding: set[pd.Timestamp] | None = None) -> list[Fold]:
    """Expanding-train walk-forward folds over auction events.

    Boundary construction, in order (spec §7, as amended 2026-10-27 —
    see Amendment A1 in the specification):
    1. The raw boundary is the sample position ``start`` (the first
       ``initial_train`` events train; blocks of ``test_block`` test;
       ``start`` is always one past the previous block's last test
       event).
    2. Refunding snap: if the first test event falls inside a
       refunding week, the boundary retreats to that week's Monday —
       every event of that week becomes a test event, so no refunding
       week is ever split across the boundary — UNLESS any event the
       retreat would pull in has already been assigned to an earlier
       fold's test block. In that case the retreat is suppressed and
       the block starts at the raw boundary: the week's earlier
       events are already test events of the earlier fold, and the
       7-day embargo (step 3) removes every earlier event of the
       same calendar week from this fold's training set (two dates
       in one Monday–Sunday week differ by at most 6 days), so the
       week still lies wholly on the test side of this fold's
       boundary and no event is tested twice.
    3. Embargo: training events dated within ``embargo_days`` calendar
       days before the first test date are dropped from training.
    4. Assertions: per fold — train strictly before test in calendar
       time; no intersection; no refunding Monday present on both
       sides; the calendar gap exceeds the embargo. Across folds —
       no sample position is assigned to more than one test block
       (asserted against the running set of assigned test positions);
       a retreat whose pulled events were already assigned is
       impossible by construction under step 2 and is asserted
       against directly.

    Why both required properties hold. (i) Cross-fold test
    uniqueness: a new block's positions are either all >= ``start``
    (hence past every earlier block, because ``start`` advances to
    one past the previous block's last test event) or, under a
    retained retreat, positions in [week Monday, ``start``) that
    step 2 has verified were assigned to no earlier test block.
    (ii) No refunding week split across a train/test boundary: if a
    boundary does not fall inside a refunding week, no refunding
    week contains events on both sides of it within the fold; if it
    does, either the retreat places the week's first event at the
    head of the test block (no event of the week precedes the
    block), or the retreat is suppressed and the embargo removes
    every earlier event of that week from training, as argued in
    step 2. The per-fold assertion in step 4 verifies (ii) on every
    realised fold.
    """
    if refunding is None:
        refunding = set()
    n = len(sample)
    dates = sample["auction_date"].to_numpy()
    mondays = sample["week_monday"].to_numpy()
    folds: list[Fold] = []
    assigned_test: set[int] = set()
    start = initial_train
    fid = 0
    while start < n:
        raw_start = start
        eff_start = raw_start
        pulled = 0
        snap_suppressed = False
        first_monday = pd.Timestamp(mondays[raw_start])
        if first_monday in refunding:
            snapped = raw_start
            while (snapped > 0
                   and pd.Timestamp(mondays[snapped - 1]) == first_monday):
                snapped -= 1
            pulled_positions = list(range(snapped, raw_start))
            if any(p in assigned_test for p in pulled_positions):
                # Amendment A1: do not re-pull events already assigned
                # to a prior test block; start at the raw boundary.
                snap_suppressed = True
            else:
                eff_start = snapped
                pulled = raw_start - snapped
        # A retained retreat extends the block end by the number of
        # events pulled in, capped at n (the frozen construction).
        test_idx = np.arange(
            eff_start, min(raw_start + test_block + pulled, n))
        overlap = assigned_test.intersection(test_idx.tolist())
        assert not overlap, (
            f"fold {fid}: test positions already assigned to an "
            f"earlier fold: {sorted(overlap)}")
        test_start_date = pd.Timestamp(dates[test_idx[0]])
        cand = np.arange(0, test_idx[0])
        cand_dates = pd.to_datetime(dates[cand])
        keep_embargo = (test_start_date - cand_dates).days > embargo_days
        train_idx = cand[keep_embargo]
        assert len(train_idx) > 0, f"fold {fid}: empty training set"
        _assert_fold_integrity(sample, train_idx, test_idx,
                               embargo_days, refunding)
        assigned_test.update(test_idx.tolist())
        meta = {
            "train_start": str(pd.Timestamp(dates[train_idx[0]]).date()),
            "train_end": str(pd.Timestamp(dates[train_idx[-1]]).date()),
            "test_start": str(test_start_date.date()),
            "test_end": str(pd.Timestamp(dates[test_idx[-1]]).date()),
            "test_pos_start": int(test_idx[0]),
            "test_pos_end": int(test_idx[-1]),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "snap_suppressed": bool(snap_suppressed),
        }
        if "weak" in sample.columns:
            meta["n_test_weak"] = int(
                sample["weak"].iloc[test_idx].sum())
        folds.append(Fold(fold_id=fid, train_idx=train_idx,
                          test_idx=test_idx, meta=meta))
        fid += 1
        start = int(test_idx[-1]) + 1
    return folds


def assert_cross_fold_test_uniqueness(folds: list[Fold]) -> int:
    """Assert no sample position appears in more than one test block.

    Returns the number of DISTINCT pooled test positions. Added under
    Amendment A1 (2026-10-27): the per-fold assertions cannot see a
    position assigned by two different folds, so the pooled
    assignment is checked as a whole.
    """
    seen: dict[int, int] = {}
    for f in folds:
        for p in f.test_idx.tolist():
            assert p not in seen, (
                f"test position {p} assigned to both fold "
                f"{seen[p]} and fold {f.fold_id}")
            seen[p] = f.fold_id
    return len(seen)


def _assert_fold_integrity(sample, train_idx, test_idx, embargo_days,
                           refunding):
    dates = sample["auction_date"]
    mondays = sample["week_monday"]
    assert dates.iloc[train_idx].max() < dates.iloc[test_idx].min(), \
        "train/test order violated"
    assert len(np.intersect1d(train_idx, test_idx)) == 0, \
        "train/test overlap"
    gap = (dates.iloc[test_idx].min()
           - dates.iloc[train_idx].max()).days
    assert gap > embargo_days, f"embargo violated: gap={gap} <= {embargo_days}"
    train_weeks = {pd.Timestamp(m) for m in mondays.iloc[train_idx]}
    test_weeks = {pd.Timestamp(m) for m in mondays.iloc[test_idx]}
    split = (train_weeks & test_weeks) & set(refunding)
    assert not split, f"refunding week split across boundary: {split}"


# ---------------------------------------------------------------------------
# Deterministic baselines (comparators for the backtest phase; no fitting)
# ---------------------------------------------------------------------------
def baseline_t2a(sample_rows: pd.DataFrame) -> np.ndarray:
    """T2a baseline forecast: the point-in-time trailing median itself."""
    return sample_rows["bench_med6_label"].to_numpy(dtype=float)


def baseline_t2b(sample_rows: pd.DataFrame) -> np.ndarray:
    """T2b baseline forecast: zero surprise."""
    return np.zeros(len(sample_rows), dtype=float)


def climatology_t2c(y_train: np.ndarray, n_test: int) -> np.ndarray:
    """T2c baseline: the training-fold weak-auction frequency."""
    return np.full(n_test, float(np.mean(y_train)))
