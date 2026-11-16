"""Tests for the ML Target 2 T2c BILLS shadow forward log
(approved by Harry 2026-10-27).

Covers: the issuance definition (one announced upcoming
regular-term bill auction, not one calendar day — parse_upcoming
admits only announced, amount-published, still-ahead, regular-term
bill auctions); the batch shape (exactly two entries per auction —
the T2c bills logistic model plus climatology; no coupon
configuration and no other Target 2 configuration); t2s- ID
parsing through the broadened scorecard regex; the inputs snapshot
carrying exactly the frozen twelve Target 2 features; duplicate
append refusal; scoring that waits for the realised bid-to-cover
ratio to be published and grades against the threshold stored at
issue time; the frozen grading / promote / kill criteria in the
scorecard header; and the failure isolation of run_ml_daily
step 5e (a T2 failure never breaks the v1 pass).
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from engine import ml_scorecard as SC
from engine import ml_t2_shadow as S
from engine import ml_target2_auction as T2

TODAY = date(2026, 10, 27)


def _td_record(**kw):
    base = {
        "securityType": "Bill",
        "securityTerm": "17-Week",
        "cashManagementBill": "No",
        "offeringAmount": "75000000000",
        "announcementDate": "2026-10-27T00:00:00",
        "competitiveAuctionCloseDate": "2026-10-28T00:00:00",
        "issueDate": "2026-11-03T00:00:00",
        "cusip": "912797XB8",
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# Issuance definition: what an "issuance" is for an auction model
# ---------------------------------------------------------------------------
def test_parse_upcoming_admits_only_announced_ahead_regular_bills():
    records = [
        _td_record(),  # the one issuable auction
        _td_record(securityType="Note", securityTerm="2-Year",
                   cusip="91282CRS2"),  # coupon: never issued by this log
        _td_record(offeringAmount="", cusip="912797VV6",
                   securityTerm="13-Week"),  # tentative, no amount yet
        _td_record(announcementDate="2026-10-29T00:00:00",
                   cusip="912797WX1", securityTerm="26-Week"),  # not yet announced
        _td_record(competitiveAuctionCloseDate="2026-10-26T00:00:00",
                   cusip="912797OLD"),  # already held: never retroactive
        _td_record(cashManagementBill="Yes", securityTerm="42-Day",
                   cusip="912797CMB"),  # cash-management bill: excluded panel
    ]
    out = S.parse_upcoming(records, today=TODAY)
    assert len(out) == 1
    u = out[0]
    assert u["security_term"] == "17-Week"
    assert u["cusip"] == "912797XB8"
    assert str(u["auction_date"].date()) == "2026-10-28"
    assert u["offering_amt"] == 75_000_000_000.0


@pytest.fixture(scope="module")
def settled():
    return T2.load_auctions()


@pytest.fixture(scope="module")
def built(settled):
    upcoming = S.parse_upcoming([_td_record()], today=TODAY)
    return S.build_t2_entries(settled, upcoming)


# ---------------------------------------------------------------------------
# Batch shape + frozen snapshot
# ---------------------------------------------------------------------------
def test_batch_shape_two_entries_logistic_plus_climatology(built):
    entries, meta = built
    assert len(entries) == 2
    assert [e["model_version"] for e in entries] == [
        S.LOGISTIC_VERSION, S.CLIM_VERSION]
    assert not any("coupon" in e["model_version"] for e in entries)
    assert len(meta["auctions"]) == 1
    a = meta["auctions"][0]
    assert a["auction_date"] == "2026-10-28"
    assert a["term"] == "17-Week"
    assert a["cusip"] == "912797XB8"
    assert 0.0 < a["logistic_probability"] < 1.0
    assert 0.0 < a["climatology_probability"] < 1.0
    for e in entries:
        assert e["prediction_id"].startswith("t2s-2026-10-28-17Week-912797XB8-")
        prob = e["prediction"]["probability"]
        assert prob == pytest.approx(
            a["logistic_probability"] if e["model_version"] == S.LOGISTIC_VERSION
            else a["climatology_probability"])
        # The weak-auction threshold is stored at issue time.
        assert e["prediction"]["threshold_btc"] == pytest.approx(a["threshold_btc"])


def test_snapshot_features_exactly_the_frozen_twelve(built):
    entries, _meta = built
    for e in entries:
        feats = e["inputs_snapshot"]["features"]
        assert list(feats.keys()) == list(T2.FEATURE_NAMES)
        assert len(feats) == 12


def test_t2_id_parsing(built):
    entries, _meta = built
    e = entries[0]
    text = "# header\n\n" + S.format_entry_t2(e) + "\n"
    parsed = SC.parse_entries(text)
    assert len(parsed) == 1
    p = parsed[0]
    assert p["prediction_id"] == e["prediction_id"]
    assert p["target_key"] == "T2c-Bills"
    assert p["asof"] == "2026-10-28"  # the auction date
    assert p["outcome"] == "pending"


def test_same_day_coupon_context_uses_announced_schedule(settled):
    # A coupon auction announced for the same date must enter the
    # bill auction's same-day context features (coupon_day_context,
    # coupon_amt_day) on the announced-schedule basis.
    upcoming = S.parse_upcoming([_td_record()], today=TODAY)
    calendar = S.parse_upcoming_calendar([
        _td_record(),
        _td_record(securityType="Note", securityTerm="2-Year",
                   cusip="91282CRS2", offeringAmount="30000000000"),
    ], today=TODAY)
    assert len(calendar) == 2  # bill + coupon on 2026-10-28
    entries, _meta = S.build_t2_entries(settled, upcoming, calendar=calendar)
    assert len(entries) == 2
    feats = entries[0]["inputs_snapshot"]["features"]
    assert feats["coupon_day_context"] == 1.0
    assert feats["coupon_amt_day"] == pytest.approx(30.0)


def test_no_upcoming_auctions_issues_nothing(settled):
    entries, meta = S.build_t2_entries(settled, [])
    assert entries == []
    assert meta["auctions"] == []


# ---------------------------------------------------------------------------
# Header criteria frozen before the first batch
# ---------------------------------------------------------------------------
def test_header_carries_frozen_grading_and_kill_criteria(tmp_path):
    path = tmp_path / "prediction_scorecard_t2_shadow.md"
    S.ensure_t2_file(path)
    text = path.read_text()
    assert "after 63 issued bill auctions" in text
    assert "counting unit is the issued auction" in text
    assert "worse than climatology's at that grading" in text
    assert "extension only by Harry's explicit decision" in text
    assert "p = 0.0001" in text
    assert "COUPON logistic configuration FAILED" in text
    assert "- last_t2_run: none" in text


# ---------------------------------------------------------------------------
# Append / duplicate refusal / scoring settlement
# ---------------------------------------------------------------------------
def test_duplicate_append_refused(built, tmp_path):
    entries, _meta = built
    path = tmp_path / "scorecard_t2.md"
    S.ensure_t2_file(path)
    assert S.append_entries_t2(entries, path=path) == 2
    before = path.read_text()
    with pytest.raises(ValueError, match="duplicate prediction_id"):
        S.append_entries_t2(entries, path=path)
    assert path.read_text() == before
    assert len(SC.parse_entries(path.read_text())) == 2


def test_scoring_waits_for_publication_then_grades_on_stored_threshold(
        built, settled, tmp_path):
    entries, meta = built
    path = tmp_path / "scorecard_t2.md"
    S.ensure_t2_file(path)
    S.append_entries_t2(entries, path=path)
    # The auction is not in the settled record yet: nothing scores.
    assert S.score_pending_t2(settled, path=path) == []
    # Publish the auction with a bid-to-cover exactly at the stored
    # threshold: weak = 1 by the frozen "at or below" definition.
    thr = meta["auctions"][0]["threshold_btc"]
    new_row = pd.DataFrame([{
        "auction_date": pd.Timestamp("2026-10-28"),
        "issue_date": pd.Timestamp("2026-11-03"),
        "security_type": "Bill", "security_term": "17-Week",
        "offering_amt": 75_000_000_000.0, "bid_to_cover_ratio": thr,
        "cusip": "912797XB8", "event_pos": len(settled),
    }])
    settled2 = pd.concat([settled, new_row], ignore_index=True)
    scored = S.score_pending_t2(settled2, path=path)
    assert len(scored) == 2
    parsed = SC.parse_entries(path.read_text())
    for e in parsed:
        prob, _t = S._pred_prob_thr(e["fields"]["prediction"])
        assert e["outcome"] == f"btc={thr!r}, weak=1"
        assert e["error"] == f"brier={(prob - 1) ** 2!r}"
    rec = {r["model_version"]: r for r in S.running_record_t2(path=path)}
    assert set(rec) == {S.LOGISTIC_VERSION, S.CLIM_VERSION}
    assert all(r["n_scored"] == 1 for r in rec.values())


# ---------------------------------------------------------------------------
# Failure isolation: run_ml_daily step 5e must never break the v1 pass
# ---------------------------------------------------------------------------
def test_run_live_t2_failure_isolated(monkeypatch, tmp_path, capsys):
    import run_ml_daily
    from engine import ml_c3_shadow, ml_panel, ml_shadow, ml_t4a_daily

    path = tmp_path / "prediction_scorecard.md"
    path.write_text(Path(SC.SCORECARD_PATH).read_text())
    monkeypatch.setattr(SC, "SCORECARD_PATH", path)

    monkeypatch.setattr(run_ml_daily, "_refresh_trace_updater", lambda: None)
    monkeypatch.setattr(run_ml_daily, "_refresh_yahoo_cache", lambda: None)
    monkeypatch.setattr(ml_panel, "main", lambda: None)
    monkeypatch.setattr(run_ml_daily, "build_scorecard_entries",
                        lambda panel, oas, asof=None: [])
    monkeypatch.setattr(SC, "score_pending", lambda panel, oas: [])
    monkeypatch.setattr(SC, "append_entries", lambda entries: 0)
    monkeypatch.setattr(SC, "running_record", lambda: [])
    monkeypatch.setattr(ml_shadow, "run_shadow", lambda panel, oas: {
        "scored_count": 0, "issued_count": 0, "issued_ids": [],
        "asof": "2026-10-23"})
    monkeypatch.setattr(ml_t4a_daily, "run_t4a", lambda: {
        "scored_count": 0, "issued_count": 0, "issued_ids": [],
        "asof": "2026-10-23"})
    monkeypatch.setattr(ml_t4a_daily, "record_t4a_error", lambda msg: None)
    monkeypatch.setattr(ml_c3_shadow, "run_c3", lambda: {
        "scored_count": 0, "issued_count": 0, "issued_ids": [],
        "asof": "2026-10-23"})
    monkeypatch.setattr(ml_c3_shadow, "record_c3_error", lambda msg: None)

    def _t2_boom():
        raise RuntimeError("t2 boom (test)")

    monkeypatch.setattr(S, "run_t2", _t2_boom)
    monkeypatch.setattr(S, "record_t2_error", lambda msg: None)

    rc = run_ml_daily.run_live()
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["shadow"]["status"] == "ok"
    assert summary["t4a"]["status"] == "ok"
    assert summary["c3"]["status"] == "ok"
    assert summary["t2"]["status"] == "failed"
    assert "t2 boom" in summary["t2"]["error"]
