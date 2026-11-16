"""ML Stage C-3 leg V-b — shadow forward scorecard (approved by Harry 2026-10-27).

Wired configuration ONLY: the leg V-b RIDGE configuration from the
frozen specification
``reports/ml_stage_c3_volatility_move_spec_2026-10-27.md``, which
passed the frozen pooled-mean-absolute-error rule within the
experiment (pooled MAE 4.0082% against the trailing-mean baseline
4.1433% and the seasonal-climatology absolute-return baseline
4.0708%; non-roll MAE 3.6737% against 3.8083% and 3.8044%).
The leg V-b gradient-boosting configuration FAILED (pooled MAE
4.3054%) and issues nothing here. Leg V-a FAILED for both of its
configurations and issues nothing anywhere.

Daily issuability, as designed: leg V-b is a regression on the
existing Target 4a panel — the same panel the graduated Target 4a
model already issues from daily — with the frozen REDUCED
18-feature set and a 5-trading-day horizon. Every trading day
therefore has exactly one issuable prediction date (the latest
panel date with a complete REDUCED feature row), exactly as for
Target 4a. Each pass:

  1. Refreshes the Target 4a dataset INCREMENTALLY through the
     Target 4a daily module's own refresh
     (``engine.ml_t4a_daily.refresh_dataset`` — the point-in-time
     panel constructor whose truncation check passed at maximum
     absolute difference 0.0) and rebuilds the panel.
  2. Scores every pending entry whose 5-trading-day horizon has
     matured — the realised outcome is the absolute value of the
     panel's ``label_ret5`` at the entry's as-of date.
  3. Issues the new day's batch of THREE predictions as of the
     latest feature-complete panel date: the frozen REDUCED ridge
     retrained on ALL labeled sample rows with the target
     ``|label_ret5|`` (``engine.ml_models.ridge_fit_predict``,
     alpha by inner TimeSeriesSplit(3) on the training data only,
     exactly the backtest's construction), the trailing-mean
     absolute-return baseline, and the seasonal-climatology
     absolute-return baseline — the two baselines of the frozen
     leg V-b bar, computed under the backtest's expanding rule
     (labels fully realised at least 6 trading positions before
     the prediction day). A date is issued at most once:
     duplicates are refused by the scorecard writer.

Scorecard: ``knowledge/prediction_scorecard_c3_shadow.md`` — a NEW
append-only file (schema mirrors engine/ml_scorecard.py section
3c; parsing and duplicate detection reuse ml_scorecard.parse_entries
via the broadened ID regex: prefix ``c3s-``). The Target 1, v2
shadow, Target 4a, and Target 2 shadow scorecards are never
written by this module.

Failure isolation: run_ml_daily.py calls run_c3() inside a
try/except exactly like the shadow (5b) and Target 4a (5c) steps;
a failure is recorded in this scorecard's header state and the
summary's "c3" key and never breaks Target 1 issuance or the
daily report. This module has no daily message of its own.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from engine import ml_models as M
from engine import ml_scorecard as SC
from engine import ml_t4a as T
from engine import ml_t4a_daily as D

ROOT = Path(__file__).resolve().parent.parent
T4A_DIR = T.T4A_DIR
SCORECARD_C3_PATH = (
    Path.home()
    / "workspace/goals/quant-research-backtest-engine/knowledge/prediction_scorecard_c3_shadow.md"
)

FEATURES = list(T.REDUCED_FEATURES)  # the frozen 18, in spec order
TARGET_DESCR = (
    "C3-Vb: absolute 5-trading-day-ahead log return of the NG=F "
    "front-month close (|label_ret5|, raw log units)"
)
HORIZON_DESCR = "5 trading days"
RIDGE_VERSION = "ridge:c3-vb-reduced18"
TRAIL_VERSION = "baseline:trailing_mean"
SEAS_VERSION = "baseline:seasonal_climatology_abs"
GRADING_ISSUANCE_DAYS = 63

_HEADER_TEMPLATE = """# Prediction Scorecard — ML Stage C-3 SHADOW (leg V-b: natural gas move size)

Shadow forward log per Harry's 2026-10-27 approval. Configuration
wired ONLY: the leg V-b ridge configuration from the frozen
specification `quant-engine/reports/ml_stage_c3_volatility_move_spec_2026-10-27.md`;
results: `quant-engine/reports/ml_stage_c3_results_2026-10-27.md`.
Within-experiment result for this configuration: pooled mean
absolute error (MAE — the average size of the forecast miss, in
percent) 4.0082% against the trailing-mean baseline 4.1433% and
the seasonal-climatology absolute-return baseline 4.0708%;
non-roll MAE 3.6737% against 3.8083% and 3.8044% (2,700 pooled
test rows, approximately 540 effective independent outcomes).
The leg V-b gradient-boosting configuration FAILED (pooled MAE
4.3054%, non-roll MAE 4.0045%) and issues NOTHING in this log.
Leg V-a failed for both of its configurations and issues nothing
anywhere. Target: the absolute value of the 5-trading-day-ahead
log return of the NG=F front-month close — this scorecard grades
how accurately the SIZE of the move is predicted, in the same
three-way contest form the backtest set up (ridge against its
two frozen baselines).

This scorecard is NOT part of the daily report. v1 remains the
only version in Harry's daily report. This log is a shadow
forward log, not a live signal: no alerting and no external
distribution.

Schema and rules mirror the Target 1 scorecard (section 3c,
engine/ml_scorecard.py): every prediction is logged at issue
time, before its outcome is knowable; no retroactive edits;
outcome/error are filled only after the 5-trading-day horizon
has closed. Prediction IDs carry the literal prefix "c3s-".

Grading criteria (single grading, Fix-4 style — frozen in this
header before the first batch was issued, not after results):
{grading_bullet}
- Promotion-eligible ONLY if, at that single grading, the
  ridge model's forward MAE beats BOTH baselines on BOTH legs
  of the frozen leg V-b bar applied forward: pooled MAE
  strictly below both baselines' pooled MAE, AND non-roll-slice
  MAE strictly below both baselines' non-roll MAE (the
  roll-window flag stored in each entry's snapshot at issue
  time defines the slices).
- Kill: if the ridge model's cumulative forward MAE is worse
  than the seasonal-climatology baseline's at grading, the
  configuration is killed and this log is retired.
- Extend: at most one extension, only by Harry's explicit
  decision. Promotion itself is Harry's decision.

Qualifications that travel with every reading of this file:
- The within-experiment pass is provisional under the Stage A
  settling-stage standard; this forward scorecard record is
  the deciding evidence for this target, per the frozen
  specification.
- The backtest margins are small (1.5% to 3.3% relative) and
  the selected ridge penalty was at the maximum of the frozen
  grid (alpha = 1000) in 39 of 43 folds: the margin is a small,
  heavily-shrunk tilt.
- Labels on roll-window days contain contract-stitch
  mechanics; they are scored as they are, exactly as the
  frozen backtest scored them, and the non-roll slice is the
  clean read.
- Storage values are as-revised-at-download, not
  as-first-published (frozen Target 4a limit, inherited).

- last_c3_run: none
- last_c3_error: none
"""


# ---------------------------------------------------------------------------
# Header / grading window (grading arithmetic reused from Target 4a)
# ---------------------------------------------------------------------------
def _now_et() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")


def _grading_bullet(w: dict | None) -> str:
    if w is None:
        return (
            "- Grading happens once, after 63 trading days of issuance. "
            "First issuance: pending; the expected 63rd issuance day, the "
            "final batch's maturity, and the grading window are filled in "
            "from the first issuance's as-of date."
        )
    return (
        f"- Grading happens once, after 63 trading days of issuance. "
        f"First issuance: {w['first_issuance']}. Expected 63rd issuance "
        f"day: {w['issuance_day_63']}. The final batch's horizon matures "
        f"{w['last_batch_matures']}; the grading window is "
        f"{w['window_start']} → {w['window_end']} (expected dates on a "
        f"weekdays-minus-US-federal-holidays calendar; they shift by "
        f"however many issuance days are skipped)."
    )


def ensure_c3_file(first_asof=None, path: Path = SCORECARD_C3_PATH) -> Path:
    """Create the C-3 shadow scorecard with its frozen header if absent."""
    path = Path(path)
    if not path.exists():
        w = D.grading_window(first_asof) if first_asof is not None else None
        path.write_text(_HEADER_TEMPLATE.format(grading_bullet=_grading_bullet(w)))
    return path


def _header_block(lines: list[str]) -> tuple[list[str], list[str]]:
    first_entry = next(
        (i for i, ln in enumerate(lines) if ln.startswith("## ")), len(lines)
    )
    return lines[:first_entry], lines[first_entry:]


def _set_grading_window(path: Path = SCORECARD_C3_PATH) -> None:
    path = Path(path)
    if not path.exists():
        return
    entries = SC.parse_entries(path.read_text())
    if not entries:
        return
    first_asof = min(e["asof"] for e in entries)
    bullet = _grading_bullet(D.grading_window(first_asof))
    lines = path.read_text().splitlines()
    header, rest = _header_block(lines)
    for i, ln in enumerate(header):
        if ln.startswith("- Grading happens once,"):
            header[i] = bullet
            break
    path.write_text("\n".join(header + rest).rstrip("\n") + "\n")


def _set_header_state(error: str | None, path: Path = SCORECARD_C3_PATH) -> None:
    path = Path(path)
    ensure_c3_file(path=path)
    lines = path.read_text().splitlines()
    header, rest = _header_block(lines)
    run_line = f"- last_c3_run: {_now_et()}"
    err_line = f"- last_c3_error: {error if error else 'none'}"
    seen_run = seen_err = False
    for i, ln in enumerate(header):
        if ln.startswith("- last_c3_run:"):
            header[i] = run_line
            seen_run = True
        elif ln.startswith("- last_c3_error:"):
            header[i] = err_line
            seen_err = True
    if not seen_run:
        header.append(run_line)
    if not seen_err:
        header.append(err_line)
    path.write_text("\n".join(header + rest).rstrip("\n") + "\n")


def record_c3_error(error: str, path: Path = SCORECARD_C3_PATH) -> None:
    """Best-effort recording of a C-3 failure into the header state."""
    _set_header_state(str(error), path=path)


# ---------------------------------------------------------------------------
# Issuance
# ---------------------------------------------------------------------------
def _fmt_snapshot_val(v: float) -> str:
    return format(float(v), ".6g")


def format_entry_c3(entry: dict) -> str:
    snap = entry["inputs_snapshot"]
    feats = ", ".join(
        f"{k}={_fmt_snapshot_val(v)}" for k, v in snap["features"].items()
    )
    dv = snap["data_vintage"]
    vintage = (
        f"panel through {dv['panel_through']} ({dv['panel_rows']} rows), "
        f"EIA storage week ending {dv['storage_week_ending']} "
        f"(available {dv['storage_available']}), training sample "
        f"{dv['sample_rows']} rows through {dv['sample_through']}"
    )
    return (
        f"## {entry['prediction_id']}\n"
        f"- issued_at: {entry['issued_at']}\n"
        f"- target: {entry['target']}\n"
        f"- horizon: {entry['horizon']}\n"
        f"- model_version: {entry['model_version']}\n"
        f"- inputs_snapshot: asof={snap['asof']}; features: {feats}; "
        f"roll_window={snap['roll_window']}; season={snap['season']}; "
        f"vintage: {vintage}\n"
        f"- prediction: {SC.format_prediction(entry['prediction'])}\n"
        f"- outcome: pending\n"
        f"- error: pending"
    )


def append_entries_c3(entries: list[dict], path: Path = SCORECARD_C3_PATH) -> int:
    """Append new entries; refuse if any prediction_id already exists."""
    path = Path(path)
    text = path.read_text()
    existing = {e["prediction_id"] for e in SC.parse_entries(text)}
    dupes = [e["prediction_id"] for e in entries if e["prediction_id"] in existing]
    if dupes:
        raise ValueError(f"duplicate prediction_id(s), already issued: {dupes}")
    blocks = "\n\n".join(format_entry_c3(e) for e in entries)
    path.write_text(text.rstrip("\n") + "\n\n" + blocks + "\n")
    return len(entries)


def _forward_baselines(ext: pd.DataFrame, i: int) -> tuple[float, float]:
    """The two frozen leg V-b baselines at extended-frame position
    ``i``, under the backtest's expanding rule (ml_stage_c3
    ``_expanding_baseline``): labels fully realised at least 6
    positions before the prediction day; the seasonal baseline
    additionally restricts to circular day-of-year distance
    <= 15. ``ext`` must carry ``y_abs`` and ``doy``."""
    y = ext["y_abs"].to_numpy(dtype=float)
    doy = ext["doy"].to_numpy(dtype=float)
    elig = (np.arange(len(y)) <= i - 6) & ~np.isnan(y)
    assert elig.any(), "V-b trailing baseline: no eligible history"
    trailing = float(np.mean(y[elig]))
    dist = np.abs(doy - doy[i])
    dist = np.minimum(dist, 366.0 - dist)
    elig_s = elig & (dist <= 15)
    assert elig_s.any(), "V-b seasonal baseline: no eligible history"
    seasonal = float(np.mean(y[elig_s]))
    return trailing, seasonal


def build_c3_entries(panel: pd.DataFrame) -> tuple[list[dict], dict]:
    """The new day's 3-entry batch (ridge |label_ret5|, trailing
    mean, seasonal climatology for absolute returns) as of the
    latest feature-complete panel date."""
    sample = T.build_sample(panel, FEATURES)
    ext = D.build_extended(panel)
    if not len(ext):
        raise RuntimeError("no panel date with a complete REDUCED feature row")
    ext = ext.copy()
    ext["y_abs"] = ext["label_ret5"].abs()
    asof = pd.Timestamp(ext["date"].iloc[-1])
    asof_row = ext.iloc[[-1]][FEATURES]
    i_last = len(ext) - 1

    X_train = sample[FEATURES]
    y_train = sample[T.TARGET].abs().to_numpy(dtype=float)
    pred_ridge, hp = M.ridge_fit_predict(X_train, y_train, asof_row)
    pred_trail, pred_seas = _forward_baselines(ext, i_last)

    stor = pd.read_parquet(T4A_DIR / "eia_storage.parquet")
    vintage = {
        "panel_through": str(panel["date"].max().date()),
        "panel_rows": int(len(panel)),
        "storage_week_ending": str(stor["week_ending"].max().date()),
        "storage_available": str(stor["available_date"].max().date()),
        "sample_rows": int(len(sample)),
        "sample_through": str(sample.index.max().date()),
    }
    snapshot = {f: float(asof_row[f].iloc[0]) for f in FEATURES}
    issued_at = _now_et()
    asof_str = str(asof.date())
    roll_window = bool(ext.iloc[-1]["roll_window"])
    season = T.season_for_month(asof.month)

    entries: list[dict] = []
    for seq, (model_version, prediction) in enumerate(
        [
            (RIDGE_VERSION, {"point": float(pred_ridge[0]), "alpha": hp["alpha"]}),
            (TRAIL_VERSION, float(pred_trail)),
            (SEAS_VERSION, float(pred_seas)),
        ],
        start=1,
    ):
        entries.append(dict(
            prediction_id=f"c3s-{asof_str}-C3Vb-{seq:02d}",
            issued_at=issued_at,
            target=TARGET_DESCR,
            horizon=HORIZON_DESCR,
            model_version=model_version,
            inputs_snapshot={
                "asof": asof_str,
                "features": dict(snapshot),
                "roll_window": roll_window,
                "season": season,
                "data_vintage": dict(vintage),
            },
            prediction=prediction,
            outcome="pending",
            error="pending",
        ))
    meta = {
        "asof": asof_str,
        "predictions": {
            "ridge_abs": float(pred_ridge[0]),
            "ridge_abs_pct": float(pred_ridge[0]) * 100.0,
            "ridge_alpha": hp["alpha"],
            "trailing_mean_abs": float(pred_trail),
            "trailing_mean_abs_pct": float(pred_trail) * 100.0,
            "seasonal_climatology_abs": float(pred_seas),
            "seasonal_climatology_abs_pct": float(pred_seas) * 100.0,
        },
        "roll_window": roll_window,
        "season": season,
        "sample_rows": int(len(sample)),
        "sample_through": str(sample.index.max().date()),
    }
    return entries, meta


# ---------------------------------------------------------------------------
# Scoring + running record
# ---------------------------------------------------------------------------
def score_pending_c3(panel: pd.DataFrame, path: Path = SCORECARD_C3_PATH) -> list[dict]:
    """Fill outcome/error for every pending entry whose 5-day
    horizon has closed. The outcome is |label_ret5| at the
    entry's as-of date. Immature entries stay pending."""
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text()
    entries = SC.parse_entries(text)
    labels = panel.set_index("date")["label_ret5"]
    lines = text.splitlines()
    scored: list[dict] = []
    for e in entries:
        if e["outcome"] != "pending":
            continue
        asof = pd.Timestamp(e["asof"])
        if asof not in labels.index:
            continue
        realized = labels.loc[asof]
        if pd.isna(realized):
            continue
        point = SC._prediction_point(e["fields"].get("prediction", ""))
        outcome = repr(float(abs(realized)))
        error = f"abs_error={abs(point - float(abs(realized)))!r}"
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


def running_record_c3(path: Path = SCORECARD_C3_PATH) -> list[dict]:
    """Per-model mean absolute error over scored entries."""
    path = Path(path)
    if not path.exists():
        return []
    entries = SC.parse_entries(path.read_text())
    agg: dict[str, list[float]] = {}
    for e in entries:
        if e["outcome"] == "pending":
            continue
        m = re.search(r"abs_error=([-\d.eE+]+)", e["error"])
        if not m:
            continue
        agg.setdefault(e["model_version"], []).append(float(m.group(1)))
    return [
        {"model_version": mv, "metric": "mean_abs_error",
         "value": float(np.mean(vals)), "n_scored": len(vals)}
        for mv, vals in sorted(agg.items())
    ]


# ---------------------------------------------------------------------------
# The daily step (called by run_ml_daily.py and the CLI dry-run)
# ---------------------------------------------------------------------------
def run_c3(path: Path = SCORECARD_C3_PATH) -> dict:
    """Score matured entries, then issue the new batch. Raises on
    failure, after best-effort recording the error into the
    scorecard header state (the caller isolates the failure)."""
    path = Path(path)
    try:
        refresh = D.refresh_dataset()
        panel = T.load_panel()
        scored = score_pending_c3(panel, path=path)
        entries, meta = build_c3_entries(panel)
        ensure_c3_file(first_asof=meta["asof"], path=path)
        issued_note = None
        try:
            issued = append_entries_c3(entries, path=path)
            issued_ids = [e["prediction_id"] for e in entries] if issued else []
        except ValueError as exc:  # already issued for this as-of date
            issued = 0
            issued_ids = []
            issued_note = str(exc)
        _set_grading_window(path=path)
        _set_header_state(None, path=path)
        summary = {
            "status": "ok",
            "asof": meta["asof"],
            "predictions": meta["predictions"],
            "roll_window": meta["roll_window"],
            "season": meta["season"],
            "scored_count": len(scored),
            "scored": scored,
            "issued_count": issued,
            "issued_ids": issued_ids,
            "running_record": running_record_c3(path=path),
            "refresh": refresh,
        }
        if issued_note is not None:
            summary["issued_note"] = issued_note
        return summary
    except Exception as exc:
        try:
            record_c3_error(str(exc), path=path)
            _set_grading_window(path=path)
        except Exception:
            pass
        raise


def dry_run_entries() -> list[dict]:
    """Would-be entries from the STORED panel (no refresh, nothing
    written)."""
    panel = T.load_panel()
    entries, _meta = build_c3_entries(panel)
    return entries


__all__ = [
    "SCORECARD_C3_PATH", "FEATURES", "RIDGE_VERSION", "TRAIL_VERSION",
    "SEAS_VERSION", "ensure_c3_file", "record_c3_error",
    "build_c3_entries", "format_entry_c3", "append_entries_c3",
    "score_pending_c3", "running_record_c3", "run_c3", "dry_run_entries",
]
