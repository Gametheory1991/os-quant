"""Phase 4 daily runner — standing issuance + scoring pass.

Modes
-----
``--dry-run``
    Print the would-be scorecard entries for the full frozen ladder as
    of the last panel date. Writes nothing. Always allowed.
``--live``
    The standing daily pass (refresh -> score matured -> issue new ->
    summary), writing to the KB prediction scorecard. **Gated by the
    issuance unlock below.**
(no flag)
    Refuses (exit 2), exactly as the gated build did.

Issuance unlock (deliberate, approved by Harry 2026-10-13)
----------------------------------------------------------
Live issuance requires the environment variable ``ML_DAILY_ISSUANCE=1``.
That variable is the unlock: it is set by the scheduled job
``ml-daily-issuance`` (owner ``goal:market-knowledge-agent-ml-prediction-
engine``; Harry approved the standing schedule 2026-10-13) and by a human
running a supervised pass. An ad-hoc ``--live`` invocation without the
variable still refuses with exit 2 and writes nothing — the refusal the
verify gate audited (reports/ml_verify_gate.md, Check 6) remains the
default posture. Dry-run behaviour is unchanged.

Live pass steps (all incremental; nothing is re-pulled from scratch):
  1. FINRA TRACE daily updater (fetches only dates newer than saved).
  2. Yahoo cache refresh via engine.data.load_or_fetch (only if stale).
  3. Panel + feature store rebuild via engine.ml_panel.main()
     (FRED series merge incrementally; auctions not re-pulled).
  4. Score every pending scorecard entry whose horizon has closed
     against realized labels (engine.ml_scorecard.score_pending);
     immature entries stay pending — never scored early.
  5. Issue the new day's entries (engine.ml_predict) as of the latest
     panel date with a complete frozen feature row, appended to the
     scorecard by engine.ml_scorecard. A date is issued at most once:
     duplicates are refused by the scorecard writer and reported.
     NaN GUARD (added 2026-10-25): before issuing, the last panel
     date's inputs are checked (see nan_guard). If illiq_agg or any
     frozen feature is NaN on the last panel date, issuing is WITHHELD
     entirely — the runner does NOT fall back to an older as-of date —
     and the summary carries status WITHHELD_NAN_GUARD plus a
     failure_banner string for the daily report to display. Before
     this guard, a NaN illiq_agg on 2026-10-20 (stale Yahoo cache)
     silently pinned issuance at the 2026-10-16 as-of date for four
     consecutive runs (Oct 19-21 passes issued 0, exit 0, one flag
     line in the JSON summary).
  5b. Shadow step (v2, shadow mode approved by Harry 2026-10-24):
     after the v1 summary is built, run engine.ml_shadow.run_shadow
     — score matured shadow entries and issue the new 8-entry
     trained-models-only v2-frozen32 shadow batch into the separate
     shadow scorecard (knowledge/prediction_scorecard_v2_shadow.md).
     This step is failure-isolated: any exception is caught, reported
     only in the summary's "shadow" key, and best-effort recorded in
     the shadow file header; it never fails the v1 pass. v1 remains
     the only version in Harry's daily report.
  5c. Target 4a step (weather -> natural gas; forward issuance
     approved by Harry 2026-10-24): after the shadow step, run
     engine.ml_t4a_daily.run_t4a — incrementally refresh the T4a
     dataset, score matured entries and issue the new 3-entry batch
     (graduated REDUCED ridge + seasonal climatology + zero-change)
     into the separate T4a scorecard
     (knowledge/prediction_scorecard_t4a.md). Failure-isolated
     exactly like 5b: any exception is caught, reported only in the
     summary's "t4a" key, and best-effort recorded in the T4a file
     header; it never fails the v1 pass. T4a has no daily message
     of its own.
  5e. Target 2 shadow step (T2c bills weak-auction classification;
     shadow forward log approved by Harry 2026-10-27): after the
     Target 4a step, run engine.ml_t2_shadow.run_t2 — refresh the
     auction record, score settled entries and issue entries for
     newly announced upcoming regular-term bill auctions (two
     entries per auction: the T2c BILLS logistic configuration,
     which passed within the experiment, plus climatology) into
     the separate T2 shadow scorecard
     (knowledge/prediction_scorecard_t2_shadow.md). An issuance is
     one announced auction, not one calendar day: a pass with no
     newly announced upcoming bill auction issues nothing. The
     T2c coupon configuration FAILED and issues nothing.
     Failure-isolated exactly like 5b/5c: any exception is caught,
     reported only in the summary's "t2" key, and best-effort
     recorded in the T2 shadow file header; it never fails the
     v1 pass. The T2 shadow has no daily message of its own.
  5f. Stage C-3 shadow step (leg V-b natural-gas move-size ridge;
     shadow forward log approved by Harry 2026-10-27): after the
     Target 2 step, run engine.ml_c3_shadow.run_c3 — refresh the
     Target 4a dataset through that module's own refresh, score
     matured entries and issue the new 3-entry batch (leg V-b
     REDUCED ridge on |label_ret5| + trailing-mean and
     seasonal-climatology absolute-return baselines) into the
     separate C-3 shadow scorecard
     (knowledge/prediction_scorecard_c3_shadow.md).
     Failure-isolated exactly like 5b/5c: any exception is caught,
     reported only in the summary's "c3" key, and best-effort
     recorded in the C-3 shadow file header; it never fails the
     v1 pass. The C-3 shadow has no daily message of its own.
  5d. Stage A reporting (approved 2026-10-27): refresh the
     standing power-accounting (E2) and calibration (E3) block in
     each scorecard's header from its stored predictions
     (engine.ml_reporting.refresh_scorecard_reporting; the shadow
     and Target 4a steps refresh their own files). Reporting only
     — zero trials, never a graduation leg. Failure-isolated
     exactly like 5b/5c: a reporting failure is recorded as a note
     in the affected file's header and in the summary, and never
     fails the v1 pass or blocks issuance.
  6. Print a JSON summary (data-as-of, issued/scored counts, running
     record vs baselines, data-integrity flags).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from engine import ml_harness as H
from engine.ml_predict import build_scorecard_entries

np.random.seed(H.SEED)

UNLOCK_ENV = "ML_DAILY_ISSUANCE"
ROOT = Path(__file__).resolve().parent
UST_DIR = Path.home() / "workspace" / "your_files" / "ust_data"
COLLECTOR_SRC = Path.home() / "workspace" / "os-bloom-merged" / "collector" / "src"
YAHOO_TICKERS = ["SHY", "IEF", "TLT", "TIP", "^TNX", "^VIX"]


def issuance_unlocked() -> bool:
    """The unlock: ML_DAILY_ISSUANCE=1 in the environment (see docstring)."""
    return os.environ.get(UNLOCK_ENV) == "1"


def _refresh_trace_updater() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(COLLECTOR_SRC) + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(
        [sys.executable, "update_trace_data.py"],
        cwd=UST_DIR, env=env, capture_output=True, text=True, timeout=1200,
    )
    tail = (r.stdout or "").strip().splitlines()[-1:] or [""]
    print(f"# trace updater: {tail[0]}", file=sys.stderr)
    if r.returncode != 0:
        raise RuntimeError(
            f"TRACE updater failed (exit {r.returncode}): "
            f"{(r.stderr or '').strip()[-500:]}"
        )


def _refresh_yahoo_cache() -> None:
    from engine.data import load_or_fetch

    for t in YAHOO_TICKERS:
        load_or_fetch(t)


def nan_guard(panel, X) -> dict:
    """Withholding guard: is the last panel date's input row trustworthy?

    Checks the LAST panel date only (historical rows legitimately carry
    NaNs — e.g. holiday-stub days, whose ILLIQ is NaN by design). The
    guard trips when, on the last panel date:

    - ``illiq_agg`` (the aggregate Amihud illiquidity the panel's main
      target predicts) is NaN, or
    - any of the 14 frozen features is NaN (an incomplete feature row).

    Returns {"tripped": bool, "banner": str | None, "nan_features":
    [...], "illiq_agg_nan": bool, "last_date": str}. The banner is the
    visible failure text the daily report shows when issuing is
    withheld; it names the date and every offending input."""
    last_date = str(panel.index.max().date())
    illiq_nan = bool(panel["illiq_agg"].isna().iloc[-1])
    last_row = X.iloc[-1]
    nan_features = [str(c) for c in X.columns if bool(last_row.isna()[c])]
    tripped = illiq_nan or bool(nan_features)
    banner = None
    if tripped:
        parts = []
        if illiq_nan:
            parts.append("illiq_agg is NaN")
        if nan_features:
            parts.append("incomplete frozen features: " + ", ".join(nan_features))
        banner = (
            f"ISSUANCE WITHHELD — NaN GUARD: on the last panel date "
            f"{last_date}, " + "; ".join(parts)
            + ". No predictions were issued this pass (no fallback to an "
              "older as-of date). Repair the input data and re-run; do "
              "not treat the zero issued count as a normal run."
        )
    return {
        "tripped": tripped,
        "banner": banner,
        "nan_features": nan_features,
        "illiq_agg_nan": illiq_nan,
        "last_date": last_date,
    }


def _run_reporting_step(summary: dict) -> None:
    """Stage A reporting step (E2 power + E3 calibration) for the
    v1 scorecard. Failure-isolated exactly like steps 5b/5c:
    refresh_scorecard_reporting never raises by contract (a failure
    is recorded in the scorecard header and returned as a failed
    status); the try/except here is a second layer so that even an
    unexpected reporting error cannot fail the v1 pass or block
    issuance."""
    try:
        from engine import ml_reporting, ml_scorecard

        summary["reporting"] = ml_reporting.refresh_scorecard_reporting(
            ml_scorecard.SCORECARD_PATH
        )
    except Exception as exc:  # noqa: BLE001 - reporting must never break v1
        summary["reporting"] = {"status": "failed", "error": str(exc)[-500:]}


def run_live() -> int:
    from engine import ml_panel, ml_scorecard

    prev_through = None
    panel_path = H.ML_DIR / "feature_store.parquet"
    if panel_path.exists():
        prev_through = str(H.load_panel().index.max().date())

    _refresh_trace_updater()
    _refresh_yahoo_cache()
    ml_panel.main()  # rebuild + persist panel, feature store, integrity.json

    panel = H.load_panel()
    oas = H.load_oas()
    through = str(panel.index.max().date())

    scored = ml_scorecard.score_pending(panel, oas)

    integrity = json.loads((H.ML_DIR / "integrity.json").read_text())
    flags: list[str] = []
    if integrity.get("recon_failures"):
        flags.append(f"recon failures: {integrity['recon_failures']}")
    for d in integrity.get("stub_days", []):
        if prev_through is None or d > prev_through:
            flags.append(f"new holiday-stub day in panel: {d} (ILLIQ NaN, kept as reported)")
    if through == prev_through:
        flags.append("panel did not advance on this pass (no new TRACE dates)")
    if bool(panel["illiq_agg"].isna().iloc[-1]):
        flags.append(f"illiq_agg is NaN on the last panel date {through}")

    X = H.build_feature_matrix(panel)
    complete = X.notna().all(axis=1)
    last_complete = X.index[complete].max() if complete.any() else None

    # NaN GUARD: a NaN-tainted last panel date withholds issuing
    # entirely. No fallback to an older as-of date — that fallback is
    # what silently pinned issuance at 2026-10-16 for four runs.
    guard = nan_guard(panel, X)
    if guard["tripped"]:
        flags.append(guard["banner"])
        print(guard["banner"], file=sys.stderr)
        summary = {
            "status": "WITHHELD_NAN_GUARD",
            "failure_banner": guard["banner"],
            "issuance_status": "withheld",
            "nan_guard": guard,
            "data_asof": str(last_complete.date()) if last_complete is not None else None,
            "panel_through": through,
            "panel_rows": int(len(panel)),
            "oas_through": str(max(s.index.max() for s in oas.values()).date()),
            "scored_count": len(scored),
            "scored": scored,
            "issued_count": 0,
            "issued_by_target": {},
            "issued_note": "withheld by the NaN guard (see failure_banner)",
            "running_record": ml_scorecard.running_record(),
            "integrity_flags": flags,
            "shadow": {"status": "withheld_nan_guard"},
        }
        _run_reporting_step(summary)
        _run_t4a_step(summary)
        _run_t2_step(summary)
        _run_c3_step(summary)
        print(json.dumps(summary, indent=2, default=float))
        return 0

    if not complete.any():
        raise RuntimeError("no panel date with a complete frozen feature row")
    asof = last_complete
    entries = build_scorecard_entries(panel, oas, asof=asof)
    issued_note = None
    try:
        issued = ml_scorecard.append_entries(entries)
    except ValueError as exc:  # already issued for this as-of date
        issued = 0
        issued_note = str(exc)

    by_target: dict[str, int] = {}
    if issued:
        for e in entries:
            tk = e["target"].split(":")[0]
            by_target[tk] = by_target.get(tk, 0) + 1
    summary = {
        "status": "OK",
        "issuance_status": "issued" if issued else "nothing_new",
        "data_asof": str(asof.date()),
        "panel_through": through,
        "panel_rows": int(len(panel)),
        "oas_through": str(max(s.index.max() for s in oas.values()).date()),
        "scored_count": len(scored),
        "scored": scored,
        "issued_count": issued,
        "issued_by_target": by_target,
        "issued_note": issued_note,
        "running_record": ml_scorecard.running_record(),
        "integrity_flags": flags,
    }
    # 5b. Shadow step (v2-frozen32, failure-isolated; see docstring).
    try:
        from engine import ml_shadow

        shadow_result = ml_shadow.run_shadow(panel, oas)
        summary["shadow"] = {
            "status": "ok",
            "scored_count": shadow_result["scored_count"],
            "issued_count": shadow_result["issued_count"],
            "issued_ids": shadow_result["issued_ids"],
            "asof": shadow_result["asof"],
            "reporting": shadow_result.get("reporting"),
        }
    except Exception as exc:  # noqa: BLE001 - shadow must never break v1
        summary["shadow"] = {"status": "failed", "error": str(exc)[-500:]}
        try:
            from engine import ml_shadow

            ml_shadow.record_shadow_error(str(exc))
        except Exception:
            pass
    # 5c. Target 4a step (failure-isolated; see docstring).
    _run_t4a_step(summary)
    # 5e. Target 2 shadow step (failure-isolated; see docstring).
    _run_t2_step(summary)
    # 5f. Stage C-3 shadow step (failure-isolated; see docstring).
    _run_c3_step(summary)
    # Stage A reporting (E2/E3) for the v1 scorecard
    # (failure-isolated; see _run_reporting_step).
    _run_reporting_step(summary)
    print(json.dumps(summary, indent=2, default=float))
    return 0


def _run_t4a_step(summary: dict) -> None:
    """Step 5c (Target 4a), failure-isolated exactly like 5b. Runs on
    both the normal and the NaN-guard-withheld paths: T4a uses its own
    weather/natural-gas dataset, not the Treasury panel, so a Treasury
    panel NaN does not taint it."""
    try:
        from engine import ml_t4a_daily

        t4a_result = ml_t4a_daily.run_t4a()
        summary["t4a"] = {
            "status": "ok",
            "scored_count": t4a_result["scored_count"],
            "issued_count": t4a_result["issued_count"],
            "issued_ids": t4a_result["issued_ids"],
            "asof": t4a_result["asof"],
            "reporting": t4a_result.get("reporting"),
        }
    except Exception as exc:  # noqa: BLE001 - T4a must never break v1
        summary["t4a"] = {"status": "failed", "error": str(exc)[-500:]}
        try:
            from engine import ml_t4a_daily

            ml_t4a_daily.record_t4a_error(str(exc))
        except Exception:
            pass


def _run_t2_step(summary: dict) -> None:
    """Step 5e (Target 2 T2c bills shadow), failure-isolated exactly
    like 5b/5c. Runs on both the normal and the NaN-guard-withheld
    paths: the T2 log uses the auction record and TreasuryDirect
    announcements, not the Treasury liquidity panel, so a panel NaN
    does not taint it."""
    try:
        from engine import ml_t2_shadow

        t2_result = ml_t2_shadow.run_t2()
        summary["t2"] = {
            "status": "ok",
            "scored_count": t2_result["scored_count"],
            "issued_count": t2_result["issued_count"],
            "issued_ids": t2_result["issued_ids"],
            "auctions": t2_result["auctions"],
        }
    except Exception as exc:  # noqa: BLE001 - T2 must never break v1
        summary["t2"] = {"status": "failed", "error": str(exc)[-500:]}
        try:
            from engine import ml_t2_shadow

            ml_t2_shadow.record_t2_error(str(exc))
        except Exception:
            pass


def _run_c3_step(summary: dict) -> None:
    """Step 5f (Stage C-3 leg V-b shadow), failure-isolated exactly
    like 5b/5c. Runs on both the normal and the NaN-guard-withheld
    paths: the C-3 log uses the weather/natural-gas dataset, not
    the Treasury liquidity panel, so a panel NaN does not taint it."""
    try:
        from engine import ml_c3_shadow

        c3_result = ml_c3_shadow.run_c3()
        summary["c3"] = {
            "status": "ok",
            "scored_count": c3_result["scored_count"],
            "issued_count": c3_result["issued_count"],
            "issued_ids": c3_result["issued_ids"],
            "asof": c3_result["asof"],
        }
    except Exception as exc:  # noqa: BLE001 - C-3 must never break v1
        summary["c3"] = {"status": "failed", "error": str(exc)[-500:]}
        try:
            from engine import ml_c3_shadow

            ml_c3_shadow.record_c3_error(str(exc))
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="print would-be scorecard entries; write nothing")
    mode.add_argument("--live", action="store_true",
                      help="standing daily pass (requires the issuance unlock: "
                           f"{UNLOCK_ENV}=1; see module docstring)")
    args = ap.parse_args()
    if args.live:
        if not issuance_unlocked():
            print(
                f"REFUSED: live issuance is unlocked only with {UNLOCK_ENV}=1 "
                "in the environment (standing approval: Harry, 2026-10-13; "
                "set by the ml-daily-issuance scheduled job or a supervised "
                "human run). Nothing was written. Re-run with --dry-run to "
                "preview the scorecard entries.",
                file=sys.stderr,
            )
            return 2
        return run_live()
    if not args.dry_run:
        print(
            "REFUSED: live prediction issuance is gated (see module "
            "docstring for the unlock). Re-run with --dry-run to preview "
            "the scorecard entries, or --live with the unlock set for the "
            "standing daily pass.",
            file=sys.stderr,
        )
        return 2
    entries = build_scorecard_entries()
    print(json.dumps(entries, indent=2, default=float))
    print(f"# dry-run: {len(entries)} would-be scorecard entries; nothing written.",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
