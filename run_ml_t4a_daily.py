"""ML Target 4a daily runner — forward issuance + scoring pass.

Modes
-----
``--dry-run``
    Print the would-be 3-entry batch (REDUCED ridge, seasonal
    climatology, zero-change) as of the latest feature-complete
    date in the STORED panel — no dataset refresh, writes nothing.
    Always allowed.
``--live``
    The Target 4a daily step (incremental dataset refresh -> score
    matured -> issue new), writing to the T4a scorecard
    (knowledge/prediction_scorecard_t4a.md). **Gated by the same
    issuance unlock as Target 1.**
(no flag)
    Refuses (exit 2).

Issuance unlock (deliberate, approved by Harry 2026-10-24 — the
Target 4a forward-scoring approval; Target 1's standing issuance
approval is Harry 2026-10-13)
-----------------------------------------------------------------
Live issuance requires the environment variable
``ML_DAILY_ISSUANCE=1`` — set by the scheduled job
``ml-daily-issuance`` (whose runner, run_ml_daily.py step 5c, calls
engine.ml_t4a_daily.run_t4a inside the daily pass; this CLI is the
supervised/manual entry point to the same step) or by a human
running a supervised pass. An ad-hoc ``--live`` invocation without
the variable refuses with exit 2 and writes nothing.
"""
from __future__ import annotations

import argparse
import json
import sys

from engine import ml_t4a_daily

UNLOCK_ENV = "ML_DAILY_ISSUANCE"


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="print would-be scorecard entries from the stored "
                           "panel (no refresh); write nothing")
    mode.add_argument("--live", action="store_true",
                      help="Target 4a daily step (requires the issuance "
                           f"unlock: {UNLOCK_ENV}=1; see module docstring)")
    args = ap.parse_args()
    if args.live:
        import os

        if os.environ.get(UNLOCK_ENV) != "1":
            print(
                f"REFUSED: live issuance is unlocked only with {UNLOCK_ENV}=1 "
                "in the environment (standing approval: Harry, 2026-10-24; "
                "set by the ml-daily-issuance scheduled job or a supervised "
                "human run). Nothing was written. Re-run with --dry-run to "
                "preview the scorecard entries.",
                file=sys.stderr,
            )
            return 2
        summary = ml_t4a_daily.run_t4a()
        print(json.dumps(summary, indent=2, default=float))
        return 0
    if not args.dry_run:
        print(
            "REFUSED: live prediction issuance is gated (see module "
            "docstring for the unlock). Re-run with --dry-run to preview "
            "the scorecard entries, or --live with the unlock set for "
            "the daily step.",
            file=sys.stderr,
        )
        return 2
    entries = ml_t4a_daily.dry_run_entries()
    print(json.dumps(entries, indent=2, default=float))
    print(f"# dry-run: {len(entries)} would-be scorecard entries from the "
          "stored panel (no refresh); nothing written.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
