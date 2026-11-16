"""Scorecard persistence for ML Target 1 (Phase 4 standing operation).

Owns the KB prediction scorecard file
(``knowledge/prediction_scorecard.md``, schema per
``knowledge_agent_design.md`` §3c):

- **Issuance** — append freshly built entries (from
  ``engine.ml_predict.build_scorecard_entries``) in the scorecard's
  markdown format. Duplicate prediction IDs are refused: a day is
  issued at most once, never silently twice.
- **Scoring** — fill ``outcome`` / ``error`` for pending entries whose
  horizon has closed against realized data. Realized values come from
  the harness's own label construction (``ml_harness.build_labels``) on
  the current panel + OAS — the same definitions the predictions were
  issued under. An entry whose label is still NaN stays pending;
  nothing is ever scored early.
- **Running record** — per (target, model) aggregates over scored
  entries: mean abs error for the T1a regression targets, mean Brier
  component for the T1b classification targets. Accuracy is only ever
  reported next to the baselines' numbers (design §3c).

Issued predictions themselves are never edited: scoring writes only
the ``outcome`` and ``error`` lines of an entry, which the schema
reserves for exactly this fill-in.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from engine import ml_harness as H

SCORECARD_PATH = (
    Path.home()
    / "workspace/goals/quant-research-backtest-engine/knowledge/prediction_scorecard.md"
)

_ID_RE = re.compile(r"^## ((?:v2s-|t4a-|t2s-|c3s-)?\d{4}-\d{2}-\d{2}-\S+)\s*$")
_FIELD_RE = re.compile(r"^- (\w+): (.*)$")
_ASOF_RE = re.compile(r"asof=(\d{4}-\d{2}-\d{2})")


# ---------------------------------------------------------------------------
# Parse / format
# ---------------------------------------------------------------------------
def parse_entries(text: str) -> list[dict]:
    """Parse scorecard markdown into entry dicts with line spans."""
    lines = text.splitlines()
    entries: list[dict] = []
    cur: dict | None = None
    for i, line in enumerate(lines):
        m = _ID_RE.match(line)
        if m:
            if cur is not None:
                cur["end_line"] = i
                entries.append(cur)
            cur = {"prediction_id": m.group(1), "start_line": i, "fields": {}}
            continue
        if cur is not None:
            fm = _FIELD_RE.match(line)
            if fm:
                cur["fields"][fm.group(1)] = fm.group(2)
    if cur is not None:
        cur["end_line"] = len(lines)
        entries.append(cur)
    for e in entries:
        f = e["fields"]
        e["target_key"] = f.get("target", "").split(":")[0]
        e["model_version"] = f.get("model_version", "")
        am = _ASOF_RE.search(f.get("inputs_snapshot", ""))
        e["asof"] = am.group(1) if am else e["prediction_id"][:10]
        e["outcome"] = f.get("outcome", "pending")
        e["error"] = f.get("error", "pending")
    return entries


def _fmt_num(v) -> str:
    return repr(float(v))


def _fmt_snapshot_val(v: float) -> str:
    return format(float(v), ".6g")


def format_prediction(prediction) -> str:
    if isinstance(prediction, (int, float)):
        return _fmt_num(prediction)
    return ", ".join(f"{k}={_fmt_num(v)}" for k, v in prediction.items())


def format_entry(entry: dict) -> str:
    snap = entry["inputs_snapshot"]
    feats = ", ".join(
        f"{k}={_fmt_snapshot_val(v)}" for k, v in snap["features"].items()
    )
    dv = snap["data_vintage"]
    vintage = (
        f"feature_store through {dv['feature_store_through']} "
        f"({dv['feature_store_rows']} rows), OAS through {dv['oas_through']}, "
        f"training sample {dv['sample_rows']} rows through {dv['sample_through']}"
    )
    return (
        f"## {entry['prediction_id']}\n"
        f"- issued_at: {entry['issued_at']}\n"
        f"- target: {entry['target']}\n"
        f"- horizon: {entry['horizon']}\n"
        f"- model_version: {entry['model_version']}\n"
        f"- inputs_snapshot: asof={snap['asof']}; features: {feats}; "
        f"vintage: {vintage}\n"
        f"- prediction: {format_prediction(entry['prediction'])}\n"
        f"- outcome: pending\n"
        f"- error: pending"
    )


def append_entries(entries: list[dict], path: Path = SCORECARD_PATH) -> int:
    """Append new entries; refuse if any prediction_id already exists."""
    text = Path(path).read_text()
    existing = {e["prediction_id"] for e in parse_entries(text)}
    dupes = [e["prediction_id"] for e in entries if e["prediction_id"] in existing]
    if dupes:
        raise ValueError(f"duplicate prediction_id(s), already issued: {dupes}")
    blocks = "\n\n".join(format_entry(e) for e in entries)
    Path(path).write_text(text.rstrip("\n") + "\n\n" + blocks + "\n")
    return len(entries)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _prediction_point(prediction_text: str) -> float:
    try:
        return float(prediction_text)
    except ValueError:
        pass
    m = re.search(r"point=([-\d.eE+]+)", prediction_text)
    if not m:
        raise ValueError(f"cannot parse prediction point from {prediction_text!r}")
    return float(m.group(1))


def _prediction_prob(prediction_text: str) -> tuple[float, float]:
    p = re.search(r"probability=([-\d.eE+]+)", prediction_text)
    t = re.search(r"threshold_pp=([-\d.eE+]+)", prediction_text)
    if not p or not t:
        raise ValueError(f"cannot parse probability/threshold from {prediction_text!r}")
    return float(p.group(1)), float(t.group(1))


def realized_outcome(entry: dict, labels: pd.DataFrame) -> tuple[str, str] | None:
    """(outcome, error) strings if the entry's horizon has closed, else None."""
    asof = pd.Timestamp(entry["asof"])
    if asof not in labels.index:
        return None
    row = labels.loc[asof]
    pred_text = entry["fields"].get("prediction", "")
    tk = entry["target_key"]
    if tk in ("T1a1", "T1a5"):
        realized = row[tk]
        if pd.isna(realized):
            return None
        point = _prediction_point(pred_text)
        return repr(float(realized)), f"abs_error={abs(point - float(realized))!r}"
    if tk in ("T1b_HY", "T1b_IG"):
        widen = row["oas_hy_widen5" if tk == "T1b_HY" else "oas_ig_widen5"]
        if pd.isna(widen):
            return None
        prob, thr = _prediction_prob(pred_text)
        y = 1 if float(widen) >= thr else 0
        brier = (prob - y) ** 2
        return f"widening_pp={float(widen)!r}, event={y}", f"brier={brier!r}"
    raise ValueError(f"unknown target key {tk!r}")


def score_pending(
    panel: pd.DataFrame, oas: dict[str, pd.Series], path: Path = SCORECARD_PATH
) -> list[dict]:
    """Fill outcome/error for every pending entry whose horizon closed.

    Returns the list of entries scored in this pass. Entries whose label
    is not yet realized are left pending, untouched."""
    path = Path(path)
    text = path.read_text()
    entries = parse_entries(text)
    labels = H.build_labels(panel, oas)
    lines = text.splitlines()
    scored: list[dict] = []
    for e in entries:
        if e["outcome"] != "pending":
            continue
        res = realized_outcome(e, labels)
        if res is None:
            continue
        outcome, error = res
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


# ---------------------------------------------------------------------------
# Running record
# ---------------------------------------------------------------------------
def running_record(path: Path = SCORECARD_PATH) -> list[dict]:
    """Per (target, model) aggregates over scored entries only."""
    entries = parse_entries(Path(path).read_text())
    agg: dict[tuple[str, str], list[float]] = {}
    for e in entries:
        if e["outcome"] == "pending":
            continue
        m = re.search(r"(?:abs_error|brier)=([-\d.eE+]+)", e["error"])
        if not m:
            continue
        agg.setdefault((e["target_key"], e["model_version"]), []).append(
            float(m.group(1))
        )
    out = []
    for (tk, mv), vals in sorted(agg.items()):
        metric = "mean_abs_error" if tk in ("T1a1", "T1a5") else "mean_brier"
        out.append(
            {
                "target": tk,
                "model_version": mv,
                "metric": metric,
                "value": float(np.mean(vals)),
                "n_scored": len(vals),
            }
        )
    return out
