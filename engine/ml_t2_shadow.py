"""ML Target 2 — T2c BILLS shadow forward log (approved by Harry 2026-10-27).

Wired configuration ONLY: the T2c (weak-auction classification)
BILLS logistic configuration from the frozen specification
``reports/ml_target2_experiment_spec_2026-10-27.md`` (including
Amendment A1), which passed the unchanged Target 1 classification
bar within the experiment (pooled Brier score 0.180117 against
climatology 0.183674; alert binomial p = 0.0001; 660 pooled test
auctions). The T2c COUPON logistic configuration FAILED both legs
(Brier 0.179694 against climatology 0.178742; alert p = 0.9733) and
issues nothing here, as does every other Target 2 configuration.

Issuance definition (specification section 11, adapted to the
bills-only approval and frozen in the scorecard header before the
first batch): an issuance is ONE announced upcoming regular-term
bill auction, not one calendar day. Each pass:

  1. Pulls the current auction record from the same FiscalData
     auctions_query source the dataset card audited (same endpoint,
     fields, and date filter as ``engine.ml_panel.fetch_auctions``),
     IN MEMORY ONLY. The persisted ``data/ml/auctions.csv`` is the
     frozen Target 2 dataset-card file (1,621 records through
     2026-10-15) on which the backtest sample and its assertions
     rest; this step never overwrites it. (An earlier form of this
     step refreshed that file in place; that mutated the frozen
     dataset and broke the Target 2 sample assertions, and was
     corrected the same day, 2026-10-27, with the frozen file
     restored and verified against every aggregate in the dataset
     card.)
  2. Scores every pending entry whose auction's realised
     bid-to-cover ratio has now been published in the refreshed
     record (weak / not weak against the threshold stored in the
     entry at issue time; Brier component filled in).
  3. Fetches TreasuryDirect's published upcoming-auctions
     announcements and issues two entries (the frozen bills
     logistic model + climatology) for each regular-term bill
     auction that is announced (announcement date on or before the
     pass date, published offering amount present), still ahead
     (auction / competitive-close date on or after the pass date),
     and not already issued in this log. A pass with no such
     auction issues nothing — no prediction is ever made for a
     non-existent auction, and no auction is ever issued twice.

The logistic model is retrained at each issuing pass on ALL
label-complete bills sample rows (the Target 4a daily precedent),
with the backtest's own pipeline (``ml_target2_backtest.
logistic_fit_predict_df``: twelve frozen numeric features
standardised, term dummies appended unscaled, C selected from
{0.01, 0.1, 1, 10} by inner TimeSeriesSplit(3) on log-loss, seed
13). Features for an upcoming auction are built by the Phase 1
implementation (``ml_target2_auction.build_features``) on the
combined frame of settled auctions plus the announced upcoming
auctions, so every demand-history quantity uses strictly prior
dates and same-day context uses announced schedule quantities
only, exactly as the frozen specification prescribes. Trailing
windows skip unreported bid-to-cover ratios; nothing is imputed.

Scorecard: ``knowledge/prediction_scorecard_t2_shadow.md`` — a NEW
append-only file (schema mirrors engine/ml_scorecard.py section
3c; parsing and duplicate detection reuse ml_scorecard.parse_entries
via the broadened ID regex: prefix ``t2s-``). The Target 1, v2
shadow, and Target 4a scorecards are never written by this module.

Failure isolation: run_ml_daily.py calls run_t2() inside a
try/except exactly like the shadow (5b) and Target 4a (5c) steps;
a failure is recorded in this scorecard's header state and the
summary's "t2" key and never breaks Target 1 issuance or the
daily report. This module has no daily message of its own.
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from engine import ml_scorecard as SC
from engine import ml_target2_auction as T2

ROOT = Path(__file__).resolve().parent.parent
SCORECARD_T2_PATH = (
    Path.home()
    / "workspace/goals/quant-research-backtest-engine/knowledge/prediction_scorecard_t2_shadow.md"
)
TD_UPCOMING_URL = "https://www.treasurydirect.gov/TA_WS/securities/upcoming"

LOGISTIC_VERSION = "logistic:t2c-bills"
CLIM_VERSION = "baseline:climatology"
TARGET_DESCR = (
    "T2c-Bills: weak auction — realised bid-to-cover ratio at or "
    "below the 20th percentile of the trailing same-term "
    "distribution (frozen Target 2 definition)"
)
HORIZON_DESCR = "auction settlement (outcome filled when the realised bid-to-cover ratio is published)"
GRADING_AUCTIONS = 63

_HEADER_TEMPLATE = """# Prediction Scorecard — ML Target 2 SHADOW (T2c: weak bill auctions)

Shadow forward log per Harry's 2026-10-27 approval. Configuration
wired ONLY: the T2c BILLS logistic configuration from the frozen
specification `quant-engine/reports/ml_target2_experiment_spec_2026-10-27.md`
(including Amendment A1); results:
`quant-engine/reports/ml_target2_results_2026-10-27.md`.
Within-experiment result for this configuration: pooled Brier
score 0.180117 against climatology 0.183674, alert binomial
p = 0.0001 (660 pooled test auctions, test span 2024-09-03 to
2026-10-13). The T2c COUPON logistic configuration FAILED both
legs of the bar (Brier 0.179694 against climatology 0.178742;
alert binomial p = 0.9733) and issues NOTHING in this log. No
other Target 2 configuration (T2a, T2b, or any coupon
configuration) issues anything in this log.

This scorecard is NOT part of the daily report. v1 remains the
only version in Harry's daily report. This log is a shadow
forward log, not a live signal: no alerting and no external
distribution.

Issuance definition (frozen before the first batch): an issuance
is ONE announced upcoming regular-term bill auction — not one
calendar day. On each daily pass, this step issues predictions
only for bill auctions that (a) appear in TreasuryDirect's
published upcoming-auctions announcements with an announcement
date on or before the pass date, a published offering amount,
and an auction (competitive close) date on or after the pass
date, (b) are in one of the seven regular bill terms (4-Week,
6-Week, 8-Week, 13-Week, 17-Week, 26-Week, 52-Week),
cash-management bills excluded, and (c) have not already been
issued in this log. A pass on a day with no such auction issues
nothing. Each issued auction carries exactly two entries: the
frozen bills logistic model (retrained on all label-complete
bills sample rows at the pass, hyperparameters selected exactly
as in the backtest) and climatology (the weak-auction frequency
of that same training sample). The prediction is the probability
that the auction is weak under the frozen T2c definition: its
realised bid-to-cover ratio at or below the 20th percentile of
the trailing same-term distribution (up to 24 most recent prior
reported same-term ratios, strictly prior dates). That threshold
is computed from prior auctions only and is stored in the entry
at issue time. Outcomes are filled only after the auction's
realised bid-to-cover ratio is published in the refreshed
auction record; an auction whose ratio is not yet published
stays pending — never scored early.

Schema and rules mirror the Target 1 scorecard (section 3c,
engine/ml_scorecard.py): every prediction is logged at issue
time, before its outcome is knowable; no retroactive edits;
outcome/error are filled only after settlement. Prediction IDs
carry the literal prefix "t2s-".

Grading criteria (single grading — frozen in this header before
the first batch was issued, not after results):
{grading_bullet}
- Promotion-eligible ONLY if, at that single grading, BOTH legs
  hold on forward data alone: (1) the logistic model's
  cumulative forward Brier score is strictly below
  climatology's, AND (2) the alert leg — the top 10% of issued
  auctions by logistic predicted probability, ties by
  auction-date order — beats its null rate (the mean
  climatology probability over the issued auctions) at the
  one-sided binomial bar p < 0.05, unchanged from the backtest.
- Kill: if the logistic model's cumulative forward Brier score
  is worse than climatology's at that grading, the
  configuration is killed and this log is retired.
- Extend: extension only by Harry's explicit decision.
  Promotion itself is Harry's decision.

Qualifications that travel with every reading of this file:
- The bills panel is the secondary panel of Target 2: a
  bills-only forward record does not settle the coupon
  question, and the two panels' verdicts are never combined.
- The within-experiment pass has not passed a settling stage;
  this forward log is the deciding evidence, per the Stage A
  protocol.
- Offering amounts enter the features on the announced-amount
  basis without an in-file announcement timestamp (dataset
  card section 6), here taken from TreasuryDirect's published
  announcement for the specific auction.
- Trailing windows skip unreported bid-to-cover ratios;
  nothing is imputed (frozen specification section 4).

- last_t2_run: none
- last_t2_error: none
"""


# ---------------------------------------------------------------------------
# Header / grading window
# ---------------------------------------------------------------------------
def _now_et() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")


def _today_et() -> date:
    return datetime.now(ZoneInfo("America/New_York")).date()


def _grading_bullet(first: dict | None) -> str:
    base = (
        f"- Grading happens once, after {GRADING_AUCTIONS} issued bill "
        "auctions. The counting unit is the issued auction (each issued "
        "auction carries one logistic entry and one climatology entry), "
        "not the issuance day and not the entry. Grading occurs after "
        f"the {GRADING_AUCTIONS}rd issued auction's outcome has settled "
        "(its realised bid-to-cover ratio published in the refreshed "
        "auction record), and the grading window is the 7 calendar "
        "days beginning on that settlement date."
    )
    if first is None:
        return base + (
            " First issuance: pending; the first issued auction is "
            "filled in from the first batch."
        )
    return base + (
        f" First issuance: auction dated {first['auction_date']} "
        f"({first['term']}, CUSIP {first['cusip']}), issued "
        f"{first['issued_date']}."
    )


def ensure_t2_file(path: Path = SCORECARD_T2_PATH) -> Path:
    """Create the T2 shadow scorecard with its frozen header if absent."""
    path = Path(path)
    if not path.exists():
        path.write_text(_HEADER_TEMPLATE.format(grading_bullet=_grading_bullet(None)))
    return path


def _header_block(lines: list[str]) -> tuple[list[str], list[str]]:
    first_entry = next(
        (i for i, ln in enumerate(lines) if ln.startswith("## ")), len(lines)
    )
    return lines[:first_entry], lines[first_entry:]


def _first_issuance(path: Path) -> dict | None:
    entries = SC.parse_entries(Path(path).read_text())
    if not entries:
        return None
    e = min(entries, key=lambda x: (x["asof"], x["prediction_id"]))
    snap = e["fields"].get("inputs_snapshot", "")
    m_term = re.search(r"term=([^;]+);", snap)
    m_cusip = re.search(r"cusip=([0-9A-Z]+)", snap)
    return {
        "auction_date": e["asof"],
        "term": m_term.group(1) if m_term else "?",
        "cusip": m_cusip.group(1) if m_cusip else "?",
        "issued_date": e["fields"].get("issued_at", "?")[:10],
    }


def _set_grading_window(path: Path = SCORECARD_T2_PATH) -> None:
    """(Re)write the header's grading bullet from the earliest entry.
    Deterministic in the first issuance; header bookkeeping only —
    never edits an issued entry."""
    path = Path(path)
    if not path.exists():
        return
    first = _first_issuance(path)
    if first is None:
        return
    bullet = _grading_bullet(first)
    lines = path.read_text().splitlines()
    header, rest = _header_block(lines)
    for i, ln in enumerate(header):
        if ln.startswith("- Grading happens once,"):
            header[i] = bullet
            break
    path.write_text("\n".join(header + rest).rstrip("\n") + "\n")


def _set_header_state(error: str | None, path: Path = SCORECARD_T2_PATH) -> None:
    path = Path(path)
    ensure_t2_file(path)
    lines = path.read_text().splitlines()
    header, rest = _header_block(lines)
    run_line = f"- last_t2_run: {_now_et()}"
    err_line = f"- last_t2_error: {error if error else 'none'}"
    seen_run = seen_err = False
    for i, ln in enumerate(header):
        if ln.startswith("- last_t2_run:"):
            header[i] = run_line
            seen_run = True
        elif ln.startswith("- last_t2_error:"):
            header[i] = err_line
            seen_err = True
    if not seen_run:
        header.append(run_line)
    if not seen_err:
        header.append(err_line)
    path.write_text("\n".join(header + rest).rstrip("\n") + "\n")


def record_t2_error(error: str, path: Path = SCORECARD_T2_PATH) -> None:
    """Best-effort recording of a T2 failure into the header state."""
    _set_header_state(str(error), path=path)


# ---------------------------------------------------------------------------
# Data refresh + upcoming announcements
# ---------------------------------------------------------------------------
AUCTIONS_QUERY_URL = (
    "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
    "/v1/accounting/od/auctions_query"
    "?filter=auction_date:gte:2023-01-25&sort=-auction_date"
    "&fields=auction_date,issue_date,security_type,security_term,"
    "offering_amt,bid_to_cover_ratio,cusip&page[size]=5000&format=json"
)


def refresh_auctions() -> pd.DataFrame:
    """Pull the current auction record from FiscalData IN MEMORY
    (the same endpoint, fields, and filter as
    ``engine.ml_panel.fetch_auctions``) and return it in Target 2
    event order. The persisted frozen dataset-card file
    ``data/ml/auctions.csv`` is never written by this function."""
    out = subprocess.run(
        ["curl", "-g", "-s", "--max-time", "90", AUCTIONS_QUERY_URL],
        capture_output=True, text=True, check=True,
    ).stdout
    payload = json.loads(out)
    recs = payload.get("data", [])
    if not recs:
        raise RuntimeError("FiscalData auctions pull returned no records")
    df = pd.DataFrame(recs)
    for c in ("auction_date", "issue_date"):
        df[c] = pd.to_datetime(df[c])
    df["auction_date"] = df["auction_date"].dt.normalize()
    df["issue_date"] = df["issue_date"].dt.normalize()
    for c in ("offering_amt", "bid_to_cover_ratio"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["auction_date", "cusip"]).reset_index(drop=True)
    df["event_pos"] = np.arange(len(df))
    return df


def fetch_upcoming_raw() -> list[dict]:
    """TreasuryDirect published upcoming-auctions announcements."""
    out = subprocess.run(
        ["curl", "-s", "--max-time", "90", TD_UPCOMING_URL],
        capture_output=True, text=True, check=True,
    ).stdout
    records = json.loads(out)
    if not isinstance(records, list):
        raise RuntimeError("TreasuryDirect upcoming payload is not a list")
    return records


def parse_upcoming(records: list[dict], today: date | None = None) -> list[dict]:
    """The issuable announced bill auctions in ``records`` (see the
    scorecard header's frozen issuance definition)."""
    today = today or _today_et()
    out: list[dict] = []
    for r in records:
        if r.get("securityType") != "Bill":
            continue
        term = r.get("securityTerm")
        if term not in T2.BILL_REGULAR_TERMS:
            continue
        if r.get("cashManagementBill") == "Yes":
            continue
        amt = r.get("offeringAmount")
        if not amt:
            continue  # tentative schedule entry, not yet announced with an amount
        ann = pd.Timestamp(r["announcementDate"]).date()
        if ann > today:
            continue
        auc = pd.Timestamp(r["competitiveAuctionCloseDate"]).date()
        if auc < today:
            continue  # auction already held; never issue retroactively
        out.append({
            "auction_date": pd.Timestamp(auc),
            "announcement_date": pd.Timestamp(ann),
            "issue_date": pd.Timestamp(r["issueDate"]),
            "security_type": "Bill",
            "security_term": term,
            "offering_amt": float(amt),
            "cusip": str(r["cusip"]),
        })
    out.sort(key=lambda d: (d["auction_date"], d["cusip"]))
    return out


def parse_upcoming_calendar(records: list[dict], today: date | None = None) -> list[dict]:
    """All announced upcoming securities (bills AND coupons) with a
    published offering amount — the announced-schedule frame the
    frozen same-day context features are computed on. Only
    regular-term bill auctions from this frame are ever issued
    (see parse_upcoming); coupon rows enter solely so that
    coupon_day_context / coupon_amt_day for an issued bill
    auction reflect the announced coupon schedule of its date,
    exactly as the backtest's calendar reflected the full file."""
    today = today or _today_et()
    out: list[dict] = []
    for r in records:
        if r.get("securityType") not in ("Bill", "Note", "Bond"):
            continue
        amt = r.get("offeringAmount")
        if not amt:
            continue
        ann = pd.Timestamp(r["announcementDate"]).date()
        if ann > today:
            continue
        auc = pd.Timestamp(r["competitiveAuctionCloseDate"]).date()
        if auc < today:
            continue
        out.append({
            "auction_date": pd.Timestamp(auc),
            "announcement_date": pd.Timestamp(ann),
            "issue_date": pd.Timestamp(r["issueDate"]),
            "security_type": str(r["securityType"]),
            "security_term": str(r["securityTerm"]),
            "offering_amt": float(amt),
            "cusip": str(r["cusip"]),
        })
    out.sort(key=lambda d: (d["auction_date"], d["cusip"]))
    return out


def _combined_auctions(settled: pd.DataFrame, upcoming: list[dict]) -> pd.DataFrame:
    """Settled auctions plus announced upcoming auctions, in event
    order — the frame features are built on. Upcoming rows carry
    no bid-to-cover ratio (label side does not exist yet). Rows
    already present in the settled record (same date and CUSIP)
    are not duplicated."""
    if not upcoming:
        return settled
    rows = pd.DataFrame([{
        "auction_date": u["auction_date"],
        "issue_date": u["issue_date"],
        "security_type": u["security_type"],
        "security_term": u["security_term"],
        "offering_amt": u["offering_amt"],
        "bid_to_cover_ratio": np.nan,
        "cusip": u["cusip"],
    } for u in upcoming])
    combined = pd.concat(
        [settled.drop(columns=["event_pos"], errors="ignore"), rows],
        ignore_index=True,
    )
    combined = combined.drop_duplicates(
        subset=["auction_date", "cusip"], keep="first")
    combined = combined.sort_values(["auction_date", "cusip"]).reset_index(drop=True)
    combined["event_pos"] = np.arange(len(combined))
    return combined


# ---------------------------------------------------------------------------
# Issuance
# ---------------------------------------------------------------------------
def _fmt_num(v) -> str:
    return repr(float(v))


def _fmt_snapshot_val(v: float) -> str:
    return format(float(v), ".6g")


def format_entry_t2(entry: dict) -> str:
    snap = entry["inputs_snapshot"]
    feats = ", ".join(
        f"{k}={_fmt_snapshot_val(v)}" for k, v in snap["features"].items()
    )
    dv = snap["data_vintage"]
    vintage = (
        f"auction record through {dv['auctions_through']} "
        f"({dv['auction_records']} records), training sample "
        f"{dv['sample_rows']} rows through {dv['sample_through']}, "
        f"announcement: TreasuryDirect upcoming auctions as of "
        f"{dv['announced_asof']}"
    )
    return (
        f"## {entry['prediction_id']}\n"
        f"- issued_at: {entry['issued_at']}\n"
        f"- target: {entry['target']}\n"
        f"- horizon: {entry['horizon']}\n"
        f"- model_version: {entry['model_version']}\n"
        f"- inputs_snapshot: asof={snap['asof']}; auction_date={snap['asof']}; "
        f"cusip={snap['cusip']}; term={snap['term']}; "
        f"announcement_date={snap['announcement_date']}; features: {feats}; "
        f"vintage: {vintage}\n"
        f"- prediction: {SC.format_prediction(entry['prediction'])}\n"
        f"- outcome: pending\n"
        f"- error: pending"
    )


def append_entries_t2(entries: list[dict], path: Path = SCORECARD_T2_PATH) -> int:
    """Append new entries; refuse if any prediction_id already exists."""
    path = Path(path)
    text = path.read_text()
    existing = {e["prediction_id"] for e in SC.parse_entries(text)}
    dupes = [e["prediction_id"] for e in entries if e["prediction_id"] in existing]
    if dupes:
        raise ValueError(f"duplicate prediction_id(s), already issued: {dupes}")
    blocks = "\n\n".join(format_entry_t2(e) for e in entries)
    path.write_text(text.rstrip("\n") + "\n\n" + blocks + "\n")
    return len(entries)


def build_t2_entries(settled: pd.DataFrame, upcoming: list[dict],
                       calendar: list[dict] | None = None) -> tuple[list[dict], dict]:
    """Two entries (logistic + climatology) per issuable upcoming
    bill auction. Auctions whose frozen feature row is incomplete
    are skipped and named in the returned meta — never issued on
    partial inputs. ``calendar`` is the announced-schedule frame
    (all announced securities) the same-day context features are
    computed on; it defaults to ``upcoming`` itself."""
    from engine import ml_target2_backtest as BT

    panel_settled = T2.bills_panel(settled)
    sample = T2.build_sample(panel_settled, settled, "bills")
    X_train = sample[T2.FEATURE_NAMES + ["group"]]
    y_train = sample["weak"].to_numpy(dtype=float)
    clim_p = float(np.mean(y_train))

    combined = _combined_auctions(settled, calendar if calendar is not None else upcoming)
    panel_comb = T2.bills_panel(combined)
    X_all = T2.build_features(panel_comb, combined, "bills")

    # Locate each upcoming auction's row in the combined bills panel.
    targets: list[tuple[dict, int]] = []
    skipped: list[dict] = []
    for u in upcoming:
        match = panel_comb.index[
            (panel_comb["auction_date"] == u["auction_date"])
            & (panel_comb["cusip"] == u["cusip"])
        ]
        if len(match) != 1:
            skipped.append({**{k: str(v) for k, v in u.items()},
                            "reason": "upcoming auction not found exactly once in combined panel"})
            continue
        idx = int(match[0])
        if not np.isfinite(X_all.loc[idx]).all():
            skipped.append({**{k: str(v) for k, v in u.items()},
                            "reason": "incomplete frozen feature row"})
            continue
        targets.append((u, idx))

    entries: list[dict] = []
    meta_auctions: list[dict] = []
    if targets:
        Xte = pd.DataFrame(
            [{**{f: float(X_all.loc[idx, f]) for f in T2.FEATURE_NAMES},
              "group": u["security_term"]} for u, idx in targets],
            columns=T2.FEATURE_NAMES + ["group"],
        )
        probs, hp = BT.logistic_fit_predict_df(
            X_train, y_train, Xte, T2.BILL_REGULAR_TERMS)
        issued_at = _now_et()
        vintage = {
            "auctions_through": str(settled["auction_date"].max().date()),
            "auction_records": int(len(settled)),
            "sample_rows": int(len(sample)),
            "sample_through": str(sample["auction_date"].max().date()),
            "announced_asof": str(_today_et()),
        }
        for (u, idx), prob in zip(targets, probs):
            thr = T2.trailing_quantile(panel_comb, panel_comb.loc[idx])
            if not np.isfinite(thr):
                skipped.append({**{k: str(v) for k, v in u.items()},
                                "reason": "weak-auction threshold not defined"})
                continue
            asof_str = str(u["auction_date"].date())
            term_slug = u["security_term"].replace("-", "")
            snapshot_feats = {f: float(X_all.loc[idx, f]) for f in T2.FEATURE_NAMES}
            base = {
                "issued_at": issued_at,
                "target": TARGET_DESCR,
                "horizon": HORIZON_DESCR,
                "inputs_snapshot": {
                    "asof": asof_str,
                    "cusip": u["cusip"],
                    "term": u["security_term"],
                    "announcement_date": str(u["announcement_date"].date()),
                    "features": dict(snapshot_feats),
                    "data_vintage": dict(vintage),
                },
            }
            entries.append(dict(
                base,
                prediction_id=f"t2s-{asof_str}-{term_slug}-{u['cusip']}-01",
                model_version=LOGISTIC_VERSION,
                prediction={"probability": float(prob),
                            "threshold_btc": float(thr), "C": hp["C"]},
                outcome="pending", error="pending",
            ))
            entries.append(dict(
                base,
                prediction_id=f"t2s-{asof_str}-{term_slug}-{u['cusip']}-02",
                model_version=CLIM_VERSION,
                prediction={"probability": clim_p,
                            "threshold_btc": float(thr)},
                outcome="pending", error="pending",
            ))
            meta_auctions.append({
                "auction_date": asof_str,
                "term": u["security_term"],
                "cusip": u["cusip"],
                "announcement_date": str(u["announcement_date"].date()),
                "offering_amt": float(u["offering_amt"]),
                "logistic_probability": float(prob),
                "climatology_probability": clim_p,
                "threshold_btc": float(thr),
                "logistic_C": hp["C"],
            })
    meta = {
        "auctions": meta_auctions,
        "skipped": skipped,
        "sample_rows": int(len(sample)),
        "sample_through": str(sample["auction_date"].max().date()),
        "climatology_probability": clim_p,
    }
    return entries, meta


# ---------------------------------------------------------------------------
# Scoring + running record
# ---------------------------------------------------------------------------
def _pred_prob_thr(prediction_text: str) -> tuple[float, float]:
    p = re.search(r"probability=([-\d.eE+]+)", prediction_text)
    t = re.search(r"threshold_btc=([-\d.eE+]+)", prediction_text)
    if not p or not t:
        raise ValueError(f"cannot parse probability/threshold from {prediction_text!r}")
    return float(p.group(1)), float(t.group(1))


def score_pending_t2(settled: pd.DataFrame, path: Path = SCORECARD_T2_PATH) -> list[dict]:
    """Fill outcome/error for every pending entry whose auction's
    realised bid-to-cover ratio is now published in the refreshed
    auction record. Auctions not yet published stay pending."""
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text()
    entries = SC.parse_entries(text)
    lines = text.splitlines()
    scored: list[dict] = []
    for e in entries:
        if e["outcome"] != "pending":
            continue
        snap = e["fields"].get("inputs_snapshot", "")
        m = re.search(r"cusip=([0-9A-Z]+)", snap)
        if not m:
            continue
        cusip = m.group(1)
        asof = pd.Timestamp(e["asof"])
        rows = settled[(settled["auction_date"] == asof)
                       & (settled["cusip"] == cusip)]
        if len(rows) != 1:
            continue
        btc = rows.iloc[0]["bid_to_cover_ratio"]
        if pd.isna(btc):
            continue
        prob, thr = _pred_prob_thr(e["fields"].get("prediction", ""))
        weak = 1 if float(btc) <= thr else 0
        outcome = f"btc={float(btc)!r}, weak={weak}"
        error = f"brier={(prob - weak) ** 2!r}"
        for i in range(e["start_line"], e["end_line"]):
            if lines[i].startswith("- outcome:"):
                lines[i] = f"- outcome: {outcome}"
            elif lines[i].startswith("- error:"):
                lines[i] = f"- error: {error}"
        scored.append({
            "prediction_id": e["prediction_id"],
            "target": e["target_key"],
            "model_version": e["model_version"],
            "outcome": outcome,
            "error": error,
        })
    if scored:
        path.write_text("\n".join(lines) + "\n")
    return scored


def running_record_t2(path: Path = SCORECARD_T2_PATH) -> list[dict]:
    """Per-model mean Brier component over scored entries."""
    path = Path(path)
    if not path.exists():
        return []
    entries = SC.parse_entries(path.read_text())
    agg: dict[str, list[float]] = {}
    for e in entries:
        if e["outcome"] == "pending":
            continue
        m = re.search(r"brier=([-\d.eE+]+)", e["error"])
        if not m:
            continue
        agg.setdefault(e["model_version"], []).append(float(m.group(1)))
    return [
        {"model_version": mv, "metric": "mean_brier",
         "value": float(np.mean(vals)), "n_scored": len(vals)}
        for mv, vals in sorted(agg.items())
    ]


# ---------------------------------------------------------------------------
# The daily step (called by run_ml_daily.py and the CLI dry-run)
# ---------------------------------------------------------------------------
def run_t2(path: Path = SCORECARD_T2_PATH) -> dict:
    """Score settled entries, then issue the newly announced
    upcoming bill auctions. Raises on failure, after best-effort
    recording the error into the scorecard header state (the
    caller isolates the failure)."""
    path = Path(path)
    ensure_t2_file(path)
    try:
        settled = refresh_auctions()
        scored = score_pending_t2(settled, path=path)
        records = fetch_upcoming_raw()
        upcoming = parse_upcoming(records)
        calendar = parse_upcoming_calendar(records)
        entries, meta = build_t2_entries(settled, upcoming, calendar=calendar)
        existing = {e["prediction_id"] for e in SC.parse_entries(path.read_text())}
        new_entries = [e for e in entries if e["prediction_id"] not in existing]
        already = len(entries) - len(new_entries)
        issued = append_entries_t2(new_entries, path=path) if new_entries else 0
        issued_ids = [e["prediction_id"] for e in new_entries] if issued else []
        _set_grading_window(path=path)
        _set_header_state(None, path=path)
        summary = {
            "status": "ok",
            "scored_count": len(scored),
            "scored": scored,
            "issued_count": issued,
            "issued_ids": issued_ids,
            "already_issued_count": already,
            "auctions": meta["auctions"],
            "skipped": meta["skipped"],
            "issued_note": (
                "no announced upcoming regular-term bill auction with a "
                "published offering amount was available at this pass"
                if not upcoming else None
            ),
            "running_record": running_record_t2(path=path),
        }
        return summary
    except Exception as exc:
        try:
            record_t2_error(str(exc), path=path)
            _set_grading_window(path=path)
        except Exception:
            pass
        raise


def dry_run_entries() -> tuple[list[dict], dict]:
    """Would-be entries from the STORED auction record plus the
    live announcements (no refresh, nothing written)."""
    settled = T2.load_auctions()
    records = fetch_upcoming_raw()
    upcoming = parse_upcoming(records)
    calendar = parse_upcoming_calendar(records)
    return build_t2_entries(settled, upcoming, calendar=calendar)


__all__ = [
    "SCORECARD_T2_PATH", "LOGISTIC_VERSION", "CLIM_VERSION",
    "ensure_t2_file", "record_t2_error", "refresh_auctions",
    "fetch_upcoming_raw", "parse_upcoming", "build_t2_entries",
    "format_entry_t2", "append_entries_t2", "score_pending_t2",
    "running_record_t2", "run_t2", "dry_run_entries",
]
