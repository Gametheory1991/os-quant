"""ML Target 1 v2 shadow-mode operation.

Shadow mode per Harry's 2026-10-24 approval: v2-frozen32 trained
models are issued and scored forward in a SEPARATE shadow scorecard
(``knowledge/prediction_scorecard_v2_shadow.md``), alongside — never
instead of — the v1 standing issuance. Shadow entries are NOT part
of Harry's daily report. Promotion requires a full quarter of
forward evidence + Harry's decision.

This module reuses engine/ml_scorecard.py's path-parameterized
functions (append_entries, score_pending, running_record) for the
shadow file; it does not fork their logic.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from engine import ml_harness as H
from engine import ml_harness_v2 as H2
from engine import ml_scorecard
from engine.ml_predict_v2 import build_scorecard_entries_v2

SHADOW_PATH = (
    Path.home()
    / "workspace/goals/quant-research-backtest-engine/knowledge/prediction_scorecard_v2_shadow.md"
)

_HEADER_TEMPLATE = """# Prediction Scorecard — ML engine v2 SHADOW (Target 1: Treasury liquidity)

Shadow mode per Harry's 2026-10-24 approval. Model version: v2-frozen32
(32 features, engine/ml_harness_v2.py; spec:
reports/ml_target1_v2_experiment_spec.md). Trained models only — no
baselines (they are version-independent and already issued under v1).

This scorecard is NOT part of the daily report. v1 remains the only
version in Harry's daily report. Promotion of v2 requires a full
quarter of forward evidence + Harry's decision.

Schema and rules per the v1 scorecard (knowledge/prediction_scorecard.md,
§3c entry schema as in engine/ml_scorecard.py format_entry): every
prediction is logged at issue time, before its outcome is knowable;
no retroactive edits; outcomes are filled only after the horizon has
closed. Shadow prediction IDs carry the literal prefix "v2s-".

- last_shadow_run: none
- last_shadow_error: none
"""


def _now_et() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")


def ensure_shadow_file(path: Path = SHADOW_PATH) -> Path:
    """Create the shadow scorecard with its required header if absent."""
    path = Path(path)
    if not path.exists():
        path.write_text(_HEADER_TEMPLATE)
    return path


def _set_header_state(error: str | None, path: Path = SHADOW_PATH) -> None:
    """Update the last_shadow_run / last_shadow_error lines in the
    header block (before the first entry). Bookkeeping only — this
    never edits an issued entry."""
    path = Path(path)
    ensure_shadow_file(path)
    text = path.read_text()
    lines = text.splitlines()
    # Header block = lines before the first entry heading.
    first_entry = next(
        (i for i, ln in enumerate(lines) if ln.startswith("## ")), len(lines)
    )
    header = lines[:first_entry]
    rest = lines[first_entry:]
    run_line = f"- last_shadow_run: {_now_et()}"
    err_line = f"- last_shadow_error: {error if error else 'none'}"
    seen_run = seen_err = False
    for i, ln in enumerate(header):
        if ln.startswith("- last_shadow_run:"):
            header[i] = run_line
            seen_run = True
        elif ln.startswith("- last_shadow_error:"):
            header[i] = err_line
            seen_err = True
    if not seen_run:
        header.append(run_line)
    if not seen_err:
        header.append(err_line)
    path.write_text("\n".join(header + rest).rstrip("\n") + "\n")


def record_shadow_error(error: str, path: Path = SHADOW_PATH) -> None:
    """Best-effort recording of a shadow failure into the header state."""
    _set_header_state(str(error), path=path)


# Alias with a descriptive name for external callers.
update_shadow_header_state = _set_header_state


def run_shadow(panel, oas, path: Path = SHADOW_PATH) -> dict:
    """Score matured shadow entries, then issue the new shadow batch.

    Returns {status, scored_count, issued_count, issued_ids, asof,
    running_record} (plus issued_note when a duplicate batch was
    refused). Raises on failure, after best-effort recording the
    error into the shadow file header state.
    """
    path = Path(path)
    ensure_shadow_file(path)
    try:
        scored = ml_scorecard.score_pending(panel, oas, path=path)

        X2 = H2.build_feature_matrix_v2(panel)
        complete = X2.notna().all(axis=1)
        if not complete.any():
            raise RuntimeError("no panel date with a complete v2 feature row")
        asof = X2.index[complete].max()
        entries = build_scorecard_entries_v2(panel, oas, asof=asof)
        issued_note = None
        try:
            issued = ml_scorecard.append_entries(entries, path=path)
            issued_ids = [e["prediction_id"] for e in entries] if issued else []
        except ValueError as exc:  # already issued for this as-of date
            issued = 0
            issued_ids = []
            issued_note = str(exc)

        _set_header_state(None, path=path)
        # Stage A reporting (E2/E3): failure-isolated inside
        # refresh_scorecard_reporting (it never raises; a failure is
        # recorded in this file's header and reported here only).
        from engine import ml_reporting

        reporting = ml_reporting.refresh_scorecard_reporting(path)
        summary = {
            "status": "ok",
            "scored_count": len(scored),
            "issued_count": issued,
            "issued_ids": issued_ids,
            "asof": str(asof.date()),
            "running_record": ml_scorecard.running_record(path=path),
            "reporting": reporting,
        }
        if issued_note is not None:
            summary["issued_note"] = issued_note
        return summary
    except Exception as exc:
        try:
            _set_header_state(str(exc), path=path)
        except Exception:
            pass
        raise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print would-be shadow scorecard entries; write nothing")
    args = ap.parse_args()
    if not args.dry_run:
        print(
            "REFUSED: ml_shadow has no live CLI mode; the daily pass "
            "(run_ml_daily.py --live) calls run_shadow(). Re-run with "
            "--dry-run to preview the shadow entries.",
            file=sys.stderr,
        )
        return 2
    panel = H.load_panel()
    oas = H.load_oas()
    X2 = H2.build_feature_matrix_v2(panel)
    complete = X2.notna().all(axis=1)
    if not complete.any():
        raise RuntimeError("no panel date with a complete v2 feature row")
    asof = X2.index[complete].max()
    entries = build_scorecard_entries_v2(panel, oas, asof=asof)
    print(json.dumps(entries, indent=2, default=float))
    print(f"# dry-run: {len(entries)} would-be shadow scorecard entries; nothing written.",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
