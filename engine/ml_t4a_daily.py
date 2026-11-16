"""ML Target 4a — daily forward issuance (Phase 4).

Forward operation approved by Harry 2026-10-24 ("Start daily forward
scoring for the gas model"). The graduated configuration ONLY: the
REDUCED ridge (18 features, forecast block excluded) from the frozen
spec ``reports/ml_target4a_experiment_spec.md``, run against the two
graded baselines issued alongside it — seasonal climatology and
zero-change. (Zero-change earned its place: it beat climatology on
every leg of the frozen backtest bar and beat the graduating model on
the clean non-roll slice; the forward contest is three-way.)

Each pass:

  1. Refreshes the T4a dataset INCREMENTALLY and rebuilds
     ``panel_t4a.parquet`` from the persisted sources via the dataset
     build's own panel constructor (``build_t4a.build_panel`` — the
     point-in-time construction whose truncation check passed at max
     abs diff 0.0). Refresh rules per source, never a from-scratch
     re-pull: NG=F via the engine cache (refreshes only when stale);
     CPC degree-day actuals — only the current-year files are
     re-downloaded (CPC updates them in place), all other years are
     the persisted raw files; EIA storage XLS — a fresh snapshot is
     fetched only when a new week has posted (its Thursday release
     date has passed and the stored parquet does not yet carry it);
     FRED Henry Hub spot — refetched only when the stored series is
     stale. CPC *forecasts* are not refreshed: the REDUCED feature set
     does not use them, and the panel's forecast columns simply carry
     their last archived values forward (the dataset build remains
     the route that extends the forecast archive).
  2. Scores every pending entry in the T4a scorecard whose
     5-trading-day horizon has matured — the panel's ``label_ret5``
     at the entry's as-of date is realized (the frozen label:
     log(close[t+5]/close[t]) on the unadjusted stitch, roll
     treatment identical to the backtest — roll days are scored as
     they are, and the scorecard's running record is over all days).
  3. Issues the new day's batch of THREE predictions as of the
     latest panel date with a complete REDUCED feature row: the
     frozen REDUCED ridge retrained on ALL labeled sample rows
     (ml_models.ridge_fit_predict — Pipeline + StandardScaler +
     Ridge, alpha by inner TimeSeriesSplit(3) on the training data
     only, exactly the backtest's construction), seasonal
     climatology under the spec's frozen rule (mean of labels over
     sample days with circular day-of-year distance <= 15 whose
     labels were fully realised at least 6 trading positions before
     the prediction day), and zero-change. A date is issued at most
     once: duplicates are refused by the scorecard writer.

Scorecard: ``knowledge/prediction_scorecard_t4a.md`` — a NEW
append-only file (schema mirrors engine/ml_scorecard.py §3c; parsing
and duplicate detection reuse ml_scorecard.parse_entries via the
broadened ID regex: prefix ``t4a-``). The Target 1 scorecard and the
v2 shadow scorecard are never written by this module.

Failure isolation: run_ml_daily.py step 5c calls run_t4a() inside a
try/except exactly like the shadow step (5b); a failure is recorded
in this scorecard's header state and the summary's "t4a" key and
never breaks Target 1 issuance or the daily report. This module has
no daily message of its own.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from engine import ml_models as M
from engine import ml_scorecard as SC
from engine import ml_t4a as T

ROOT = Path(__file__).resolve().parent.parent
T4A_DIR = T.T4A_DIR
PANEL_PATH = T.PANEL_PATH
RAW_DIR = T4A_DIR / "raw"
SCORECARD_T4A_PATH = (
    Path.home()
    / "workspace/goals/quant-research-backtest-engine/knowledge/prediction_scorecard_t4a.md"
)

FEATURES = list(T.REDUCED_FEATURES)  # the frozen 18, in spec order
TARGET_DESCR = (
    "T4a-NG-5d: 5-trading-day-ahead log return of the NG=F "
    "front-month close (raw log units)"
)
HORIZON_DESCR = "5 trading days"
RIDGE_VERSION = "ridge:t4a-reduced18"
CLIM_VERSION = "baseline:climatology"
ZERO_VERSION = "baseline:zero-change"
GRADING_ISSUANCE_DAYS = 63

_HEADER_TEMPLATE = """# Prediction Scorecard — ML Target 4a (weather → natural gas futures)

Forward issuance per Harry's 2026-10-24 approval. Graduated
configuration ONLY: the REDUCED ridge (18 features, forecast block
excluded) from the frozen spec
`quant-engine/reports/ml_target4a_experiment_spec.md`; results:
`goals/market-knowledge-agent-ml-prediction-engine/files/ml_target4a_results_2026-10-24.md`.
Issued alongside it, every day, the two graded baselines: seasonal
climatology (the calendar) and zero-change (predict no move). Target:
the 5-trading-day-ahead log return of the NG=F front-month close —
this scorecard grades how big the move is (mean absolute error of
the predicted return), the three-way contest the backtest set up.

Schema and rules mirror the Target 1 scorecard (§3c, engine/
ml_scorecard.py): every prediction is logged at issue time, before
its outcome is knowable; no retroactive edits; outcome/error are
filled only after the 5-trading-day horizon has closed. Prediction
IDs carry the literal prefix "t4a-".

Grading criteria (single grading, Fix-4 style — set before the
first batch, not after results):
{grading_bullet}
- Promotion-eligible ONLY if the model's forward mean absolute
  error beats BOTH climatology and zero-change on this scorecard.
- Kill: if the model's cumulative mean absolute error is worse
  than climatology's at grading, the configuration is killed.
- Extend: at most one extension, only by Harry's explicit
  decision. Promotion itself is Harry's decision.

Qualifications that travel with every reading of this file:
- On the backtest (2,700 test days, 2016-01-15 → 2026-10-16) the
  model's margin over zero-change was 0.0048 MAE points (model
  6.4042% vs zero-change 6.4090%; climatology 6.5175%) — a whisker.
- The backtest rests on roughly 540 effective independent
  outcomes (overlapping 5-day labels), not 2,700.
- Direction is a coin flip in every model tested (up/down
  accuracy 49.6%–51.7%): this scorecard grades move-SIZE
  accuracy, not direction calls.
- Labels on roll-window days (38.3% of backtest days) contain
  contract-stitch mechanics; they are scored as they are,
  exactly as the frozen backtest scored them.
- Storage values are as-revised-at-download, not
  as-first-published (frozen spec §5, disclosed limit).

- last_t4a_run: none
- last_t4a_error: none
"""


# ---------------------------------------------------------------------------
# Trading-day arithmetic (grading window)
# ---------------------------------------------------------------------------
def _trading_days_from(start: pd.Timestamp, count: int) -> list[pd.Timestamp]:
    """Weekdays minus US federal holidays, starting at `start`
    (rolled forward to the next trading day if it is not one)."""
    start = pd.Timestamp(start).normalize()
    span_end = start + pd.Timedelta(days=int(count * 1.6) + 30)
    days = pd.bdate_range(start, span_end)
    holidays = USFederalHolidayCalendar().holidays(days[0], days[-1])
    return [d for d in days if d not in holidays]


def nth_trading_day(start: pd.Timestamp, n: int) -> pd.Timestamp:
    """The n-th trading day counting `start` (or the next trading
    day, if it is not one) as the 1st."""
    return _trading_days_from(start, n)[n - 1]


def grading_window(first_asof) -> dict:
    """Expected grading dates for a first issuance on `first_asof`:
    the 63rd issuance day, the day the final batch's 5-trading-day
    horizon matures (5 trading days after the 63rd issuance day),
    and a one-week grading window from maturity."""
    first = pd.Timestamp(first_asof).normalize()
    day63 = nth_trading_day(first, GRADING_ISSUANCE_DAYS)
    matures = nth_trading_day(day63, 6)  # day63 counts as day 1 → +5 trading days
    return {
        "first_issuance": str(first.date()),
        "issuance_day_63": str(day63.date()),
        "last_batch_matures": str(matures.date()),
        "window_start": str(matures.date()),
        "window_end": str((matures + pd.Timedelta(days=7)).date()),
    }


# ---------------------------------------------------------------------------
# Scorecard file: header, state, formatting, append
# ---------------------------------------------------------------------------
def _now_et() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")


def _grading_bullet(w: dict | None) -> str:
    """The header's single-line grading bullet (see grading_window)."""
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


def ensure_t4a_file(first_asof=None, path: Path = SCORECARD_T4A_PATH) -> Path:
    """Create the T4a scorecard with its criteria header if absent.
    The grading-window dates are filled from the first issuance's
    as-of date (here, or later by _set_grading_window once the
    first entries exist)."""
    path = Path(path)
    if not path.exists():
        w = grading_window(first_asof) if first_asof is not None else None
        path.write_text(_HEADER_TEMPLATE.format(grading_bullet=_grading_bullet(w)))
    return path


def _header_block(lines: list[str]) -> tuple[list[str], list[str]]:
    first_entry = next(
        (i for i, ln in enumerate(lines) if ln.startswith("## ")), len(lines)
    )
    return lines[:first_entry], lines[first_entry:]


def _set_grading_window(path: Path = SCORECARD_T4A_PATH) -> None:
    """(Re)write the header's grading lines from the earliest entry
    in the file. Deterministic in the first issuance's as-of date;
    a no-op until at least one entry exists. Header bookkeeping
    only — never edits an issued entry."""
    path = Path(path)
    if not path.exists():
        return
    entries = SC.parse_entries(path.read_text())
    if not entries:
        return
    first_asof = min(e["asof"] for e in entries)
    bullet = _grading_bullet(grading_window(first_asof))
    text = path.read_text()
    lines = text.splitlines()
    header, rest = _header_block(lines)
    for i, ln in enumerate(header):
        if ln.startswith("- Grading happens once,"):
            header[i] = bullet
            break
    path.write_text("\n".join(header + rest).rstrip("\n") + "\n")


def _set_header_state(error: str | None, path: Path = SCORECARD_T4A_PATH) -> None:
    """Update the last_t4a_run / last_t4a_error header lines.
    Bookkeeping only — never edits an issued entry."""
    path = Path(path)
    ensure_t4a_file(path=path)
    text = path.read_text()
    lines = text.splitlines()
    header, rest = _header_block(lines)
    run_line = f"- last_t4a_run: {_now_et()}"
    err_line = f"- last_t4a_error: {error if error else 'none'}"
    seen_run = seen_err = False
    for i, ln in enumerate(header):
        if ln.startswith("- last_t4a_run:"):
            header[i] = run_line
            seen_run = True
        elif ln.startswith("- last_t4a_error:"):
            header[i] = err_line
            seen_err = True
    if not seen_run:
        header.append(run_line)
    if not seen_err:
        header.append(err_line)
    path.write_text("\n".join(header + rest).rstrip("\n") + "\n")


def record_t4a_error(error: str, path: Path = SCORECARD_T4A_PATH) -> None:
    """Best-effort recording of a T4a failure into the header state."""
    _set_header_state(str(error), path=path)


def _fmt_num(v) -> str:
    return repr(float(v))


def _fmt_snapshot_val(v: float) -> str:
    return format(float(v), ".6g")


def format_entry_t4a(entry: dict) -> str:
    """The §3c entry schema (same fields as ml_scorecard.format_entry)
    with a T4a data-vintage string."""
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
        f"vintage: {vintage}\n"
        f"- prediction: {SC.format_prediction(entry['prediction'])}\n"
        f"- outcome: pending\n"
        f"- error: pending"
    )


def append_entries_t4a(entries: list[dict], path: Path = SCORECARD_T4A_PATH) -> int:
    """Append new entries; refuse if any prediction_id already exists
    (ml_scorecard.append_entries' rule, T4a formatter)."""
    path = Path(path)
    text = path.read_text()
    existing = {e["prediction_id"] for e in SC.parse_entries(text)}
    dupes = [e["prediction_id"] for e in entries if e["prediction_id"] in existing]
    if dupes:
        raise ValueError(f"duplicate prediction_id(s), already issued: {dupes}")
    blocks = "\n\n".join(format_entry_t4a(e) for e in entries)
    path.write_text(text.rstrip("\n") + "\n\n" + blocks + "\n")
    return len(entries)


# ---------------------------------------------------------------------------
# Incremental dataset refresh (see module docstring for the rules)
# ---------------------------------------------------------------------------
def _refresh_cpc_current_year(B) -> list[str]:
    """Re-download only the current-year CPC actuals files (CPC
    updates them in place). On any fetch failure the persisted file
    is kept and the staleness surfaces as panel feature lag."""
    refreshed: list[str] = []
    yr = date.today().year
    for kind in ("Heating", "Cooling"):
        dest = B.RAW / "cpc_actuals" / f"{yr}_Population.{kind}.txt"
        if not dest.exists():
            continue  # build_actuals' fetch_if_missing does the first fetch
        tmp = dest.parent / (dest.name + ".tmp")
        try:
            B.curl(f"{B.CPC}/daily_data/{yr}/Population.{kind}.txt", tmp)
            tmp.rename(dest)
            refreshed.append(f"cpc_actuals_{yr}_{kind}")
        except Exception:  # noqa: BLE001 - keep persisted file
            if tmp.exists():
                tmp.unlink()
    return refreshed


def refresh_dataset() -> dict:
    """Incrementally refresh the persisted T4a sources and rebuild
    the panel. Returns a refresh-info dict for the run summary."""
    from data.t4a_weather_futures import build_t4a as B

    info: dict = {"sources_refreshed": [], "panel_before": None,
                  "panel_after": None, "newly_realized_labels": 0,
                  "overlap_max_abs_diff": None}
    old_panel = T.load_panel() if PANEL_PATH.exists() else None
    if old_panel is not None:
        info["panel_before"] = str(old_panel["date"].max().date())

    # A. NG=F — engine cache refreshes only when stale.
    ng_cache_max_before = None
    cache_path = ROOT / "data" / "cache" / "NG=F.parquet"
    if cache_path.exists():
        ng_cache_max_before = str(pd.read_parquet(cache_path).index.max().date())
    ng = B.build_ng()  # load_or_fetch inside; dated snapshot per build convention
    ng_cache_max_after = str(ng["date"].max().date())
    if ng_cache_max_after != ng_cache_max_before:
        info["sources_refreshed"].append(f"ng_f_cache→{ng_cache_max_after}")

    # B. CPC actuals — current-year files refreshed, then the
    #    persisted raw set is re-parsed (all other years skip-fetch).
    info["sources_refreshed"].extend(_refresh_cpc_current_year(B))
    dd = B.build_actuals()
    norms = B.compute_norms(dd)
    norms.to_csv(T4A_DIR / "cpc_norms_1981-2010.csv")
    dd.reset_index().to_parquet(T4A_DIR / "cpc_actuals_conus.parquet", index=False)
    info["cpc_actuals_through"] = str(dd.index.max().date())

    # C. CPC forecasts — NOT refreshed (REDUCED does not use them);
    #    the stored summary is carried into the panel unchanged.
    fc_sum = pd.read_parquet(T4A_DIR / "cpc_forecasts_conus_summary.parquet")

    # D. EIA storage — fresh XLS snapshot only when a new week has
    #    posted (next available_date <= today and not yet stored).
    stor_path = T4A_DIR / "eia_storage.parquet"
    stor = pd.read_parquet(stor_path)
    next_available = stor["available_date"].max() + pd.Timedelta(days=7)
    if pd.Timestamp(date.today()) >= next_available:
        stor = B.build_storage()
        stor.to_parquet(stor_path, index=False)
        info["sources_refreshed"].append(
            f"eia_storage→week_ending {stor['week_ending'].max().date()}"
        )
    info["storage_week_ending"] = str(stor["week_ending"].max().date())
    info["storage_available"] = str(stor["available_date"].max().date())

    # E. FRED Henry Hub spot — refetch only when the stored series
    #    is more than a day behind today.
    hh_path = T4A_DIR / "fred_hh_spot.parquet"
    hh = pd.read_parquet(hh_path)
    if (pd.Timestamp(date.today()) - hh["date"].max()).days > 1:
        hh = B.build_hh()
        hh.to_parquet(hh_path, index=False)
        info["sources_refreshed"].append(f"fred_hh_spot→{hh['date'].max().date()}")
    info["hh_spot_through"] = str(hh["date"].max().date())

    # Panel rebuild from the persisted sources (point-in-time
    # construction; the dataset truncation check passed at 0.0).
    panel = B.build_panel(ng, dd, norms, fc_sum, stor, hh)
    panel.to_parquet(PANEL_PATH, index=False)
    info["panel_after"] = str(panel["date"].max().date())
    info["panel_rows"] = int(len(panel))

    # Overlap integrity: rows that existed before must not change
    # (features/closes); labels may only newly realize at the tail.
    if old_panel is not None:
        old = old_panel.set_index("date")
        new = panel.set_index("date")
        common = old.index.intersection(new.index)
        cols = ["ng_close"] + FEATURES
        a = old.loc[common, cols].to_numpy(dtype=float)
        b = new.loc[common, cols].to_numpy(dtype=float)
        both = ~np.isnan(a) & ~np.isnan(b)
        info["overlap_max_abs_diff"] = (
            float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
        )
        info["overlap_nan_pattern_changes"] = int((np.isnan(a) != np.isnan(b)).sum())
        lab_old = old.loc[common, "label_ret5"].to_numpy(dtype=float)
        lab_new = new.loc[common, "label_ret5"].to_numpy(dtype=float)
        info["newly_realized_labels"] = int((np.isnan(lab_old) & ~np.isnan(lab_new)).sum())
    return info


# ---------------------------------------------------------------------------
# Issuance
# ---------------------------------------------------------------------------
def build_extended(panel: pd.DataFrame) -> pd.DataFrame:
    """Panel rows with a complete REDUCED feature row (label may be
    NaN — the tail rows are the issuable ones), in panel order."""
    keep = panel[FEATURES].notna().all(axis=1)
    ext = panel.loc[keep].copy().reset_index(drop=True)
    ext["doy"] = pd.to_datetime(ext["date"]).dt.dayofyear.to_numpy(dtype=float)
    return ext


def climatology_forward(ext: pd.DataFrame, asof) -> float:
    """The spec's frozen climatology rule applied forward: mean of
    label_ret5 over days s with circular day-of-year distance <= 15
    from the prediction day AND s at least 6 positions earlier in
    the feature-complete ordering AND s's label realized. For any
    in-sample date this coincides exactly with
    ml_t4a.climatology_predict (feature-complete unlabeled rows
    exist only in the panel's last 5 rows, after every sample row),
    which the test suite asserts."""
    y = ext["label_ret5"].to_numpy(dtype=float)
    doy = ext["doy"].to_numpy(dtype=float)
    hits = np.where(pd.to_datetime(ext["date"]) == pd.Timestamp(asof))[0]
    assert len(hits) == 1, f"asof {asof} not found exactly once in extended frame"
    i = int(hits[0])
    dist = np.abs(doy - doy[i])
    dist = np.minimum(dist, 366.0 - dist)
    elig = (dist <= 15) & (np.arange(len(y)) <= i - 6) & ~np.isnan(y)
    assert elig.any(), f"climatology: no eligible history for asof {asof}"
    return float(np.mean(y[elig]))


def build_t4a_entries(panel: pd.DataFrame) -> tuple[list[dict], dict]:
    """The new day's 3-entry batch (ridge REDUCED, climatology,
    zero-change) as of the latest feature-complete panel date."""
    sample = T.build_sample(panel, FEATURES)
    ext = build_extended(panel)
    if not len(ext):
        raise RuntimeError("no panel date with a complete REDUCED feature row")
    asof = pd.Timestamp(ext["date"].iloc[-1])
    asof_row = ext.iloc[[-1]][FEATURES]

    X_train = sample[FEATURES]
    y_train = sample[T.TARGET].to_numpy(dtype=float)
    pred_ridge, hp = M.ridge_fit_predict(X_train, y_train, asof_row)
    pred_clim = climatology_forward(ext, asof)
    pred_zero = 0.0

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

    entries: list[dict] = []
    for seq, (model_version, prediction) in enumerate(
        [
            (RIDGE_VERSION, {"point": float(pred_ridge[0]), "alpha": hp["alpha"]}),
            (CLIM_VERSION, float(pred_clim)),
            (ZERO_VERSION, float(pred_zero)),
        ],
        start=1,
    ):
        entries.append(
            dict(
                prediction_id=f"t4a-{asof_str}-T4aNG5d-{seq:02d}",
                issued_at=issued_at,
                target=TARGET_DESCR,
                horizon=HORIZON_DESCR,
                model_version=model_version,
                inputs_snapshot={
                    "asof": asof_str,
                    "features": dict(snapshot),
                    "data_vintage": dict(vintage),
                },
                prediction=prediction,
                outcome="pending",
                error="pending",
            )
        )
    meta = {
        "asof": asof_str,
        "predictions": {
            "ridge_reduced": float(pred_ridge[0]),
            "ridge_reduced_pct": float(pred_ridge[0]) * 100.0,
            "ridge_alpha": hp["alpha"],
            "climatology": float(pred_clim),
            "climatology_pct": float(pred_clim) * 100.0,
            "zero_change": 0.0,
        },
        "sample_rows": int(len(sample)),
        "sample_through": str(sample.index.max().date()),
    }
    return entries, meta


# ---------------------------------------------------------------------------
# Scoring + running record
# ---------------------------------------------------------------------------
def score_pending_t4a(panel: pd.DataFrame, path: Path = SCORECARD_T4A_PATH) -> list[dict]:
    """Fill outcome/error for every pending entry whose 5-day
    horizon has closed (panel label_ret5 realized at its as-of
    date). Immature entries stay pending — never scored early."""
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
        outcome = repr(float(realized))
        error = f"abs_error={abs(point - float(realized))!r}"
        for i in range(e["start_line"], e["end_line"]):
            if lines[i].startswith("- outcome:"):
                lines[i] = f"- outcome: {outcome}"
            elif lines[i].startswith("- error:"):
                lines[i] = f"- error: {error}"
        scored.append(
            {
                "prediction_id": e["prediction_id"],
                "target": e["target_key"],
                "model_version": e["model_version"],
                "outcome": outcome,
                "error": error,
            }
        )
    if scored:
        path.write_text("\n".join(lines) + "\n")
    return scored


def running_record_t4a(path: Path = SCORECARD_T4A_PATH) -> list[dict]:
    """Per-model mean absolute error over scored entries (the single
    T4a target; all three issued forecasts side by side)."""
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
        {
            "model_version": mv,
            "metric": "mean_abs_error",
            "value": float(np.mean(vals)),
            "n_scored": len(vals),
        }
        for mv, vals in sorted(agg.items())
    ]


# ---------------------------------------------------------------------------
# The daily step (called by run_ml_daily.py step 5c and the CLI)
# ---------------------------------------------------------------------------
def run_t4a(path: Path = SCORECARD_T4A_PATH) -> dict:
    """Score matured entries, then issue the new batch. Raises on
    failure, after best-effort recording the error into the
    scorecard header state (the caller isolates the failure)."""
    path = Path(path)
    try:
        refresh = refresh_dataset()
        panel = T.load_panel()
        scored = score_pending_t4a(panel, path=path)
        entries, meta = build_t4a_entries(panel)
        ensure_t4a_file(first_asof=meta["asof"], path=path)
        issued_note = None
        try:
            issued = append_entries_t4a(entries, path=path)
            issued_ids = [e["prediction_id"] for e in entries] if issued else []
        except ValueError as exc:  # already issued for this as-of date
            issued = 0
            issued_ids = []
            issued_note = str(exc)
        _set_grading_window(path=path)
        _set_header_state(None, path=path)
        # Stage A reporting (E2 power block; E3 does not apply to
        # this regression-only scorecard): failure-isolated inside
        # refresh_scorecard_reporting (it never raises; a failure is
        # recorded in this file's header and reported here only).
        from engine import ml_reporting

        reporting = ml_reporting.refresh_scorecard_reporting(path)
        summary = {
            "status": "ok",
            "reporting": reporting,
            "asof": meta["asof"],
            "predictions": meta["predictions"],
            "scored_count": len(scored),
            "scored": scored,
            "issued_count": issued,
            "issued_ids": issued_ids,
            "running_record": running_record_t4a(path=path),
            "refresh": refresh,
        }
        if issued_note is not None:
            summary["issued_note"] = issued_note
        return summary
    except Exception as exc:
        try:
            record_t4a_error(str(exc), path=path)
            _set_grading_window(path=path)
        except Exception:
            pass
        raise


def dry_run_entries() -> list[dict]:
    """Would-be entries from the STORED panel (no refresh, nothing
    written) — the CLI's --dry-run."""
    panel = T.load_panel()
    entries, _meta = build_t4a_entries(panel)
    return entries


__all__ = [
    "SCORECARD_T4A_PATH", "FEATURES", "grading_window", "nth_trading_day",
    "ensure_t4a_file", "record_t4a_error", "refresh_dataset",
    "build_extended", "climatology_forward", "build_t4a_entries",
    "format_entry_t4a", "append_entries_t4a", "score_pending_t4a",
    "running_record_t4a", "run_t4a", "dry_run_entries",
]
