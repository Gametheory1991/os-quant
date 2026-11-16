"""Target 5a forward issuance runner — standard-NAV stratum only.

Implements the frozen forward protocol
`~/workspace/goals/quant-research-backtest-engine/knowledge/target5a_forward_protocol_2026-11-03.md`
(frozen 2026-11-03, before the first issuance; trial 8331, cross-referenced
to the Target 5a experiment trials 7830-7837).

Each run:
  1. Rebuilds the live standard-NAV panel from the accruing caches
     (engine.ml_t5a.reconstruct_fund with no end date — the experiment's
     frozen 2026-10-27 panel end applies to the experiment only).
  2. Scores every previously issued prediction whose fund now has an
     aligned panel position five positions beyond the issued position.
     Issued values are never edited; scoring writes only the realised
     value and the absolute-error fields.
  3. If the latest aligned date D* is later than the most recent issued
     batch's feature date, fits the two graduated configurations once,
     pooled, on the training set defined by the experiment's purge and
     five-date embargo applied to D*, and issues four values per eligible
     fund (gradient boosting, ridge, persistence, trailing median).
  4. Persists the prediction store and re-renders the scorecard.

Prints one JSON summary line. Exit code 0 on success (including the
nothing-new state); non-zero on failure.
"""
from __future__ import annotations

import json
import os
import time
import random
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from engine import ml_t5a as T
from engine import ml_models as M

ROOT = Path(__file__).resolve().parent
STORE_DIR = ROOT / "data" / "t5a_premium" / "forward"
STORE_PATH = STORE_DIR / "t5a_forward_predictions.parquet"
KG_DIR = (Path.home() / "workspace" / "goals" /
          "quant-research-backtest-engine" / "knowledge")
SCORECARD_PATH = KG_DIR / "prediction_scorecard_t5a_forward.md"

STORE_COLUMNS = [
    "issue_date", "feature_date", "ticker", "aligned_pos", "premium_bp",
    "pred_gbm_bp", "pred_ridge_bp", "pred_persistence_bp",
    "pred_trailmed_bp", "ridge_alpha", "vix_level", "vix_stress",
    "credit", "scored", "realised_bp", "err_gbm_bp", "err_ridge_bp",
    "err_persistence_bp", "err_trailmed_bp",
]


def write_with_retry(path: Path, write_fn, verify_fn, attempts: int = 30):
    """Bounded-retry write with per-step receipt verification (standing
    remedy for the host's intermittent EPERM write fault)."""
    last_exc = None
    for k in range(attempts):
        try:
            write_fn()
            verify_fn()
            return
        except (PermissionError, OSError) as exc:
            last_exc = exc
            time.sleep(5 + random.uniform(0, 20))
    raise RuntimeError(f"write to {path} failed after {attempts} attempts: "
                       f"{last_exc}")


def save_store(store: pd.DataFrame):
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STORE_PATH.with_suffix(".tmp.parquet")

    def _write():
        store.to_parquet(tmp, index=False)
        os.replace(tmp, STORE_PATH)

    def _verify():
        back = pd.read_parquet(STORE_PATH)
        assert len(back) == len(store), "store row count mismatch after write"

    write_with_retry(STORE_PATH, _write, _verify)


def load_store() -> pd.DataFrame:
    if STORE_PATH.exists():
        store = pd.read_parquet(STORE_PATH)
    else:
        store = pd.DataFrame(columns=STORE_COLUMNS)
    # The scored flag must be a true boolean: an integer 0/1 column makes
    # the ~ operator produce -1/-2 instead of True/False.
    store["scored"] = store["scored"].fillna(False).astype(bool)
    store["vix_stress"] = store["vix_stress"].fillna(False).astype(bool)
    store["credit"] = store["credit"].fillna(False).astype(bool)
    return store


def build_live_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = [T.reconstruct_fund(t, end=None) for t in T.STD_FUNDS]
    panel = pd.concat(parts, ignore_index=True)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    vix = T.load_vix()
    feats = pd.concat(
        [T.build_fund_features(panel[panel["ticker"] == t], vix)
         for t in T.STD_FUNDS],
        ignore_index=True)
    return panel, feats


def score_outstanding(store: pd.DataFrame, panel: pd.DataFrame) -> int:
    """Score issued rows whose horizon has closed. Returns the number of
    newly scored rows."""
    if store.empty:
        return 0
    prem = {(r.ticker, int(r.aligned_pos)): float(r.premium_bp)
            for r in panel.itertuples()}
    n_new = 0
    unscored = store.index[~store["scored"]]
    for idx in unscored:
        row = store.loc[idx]
        key = (row["ticker"], int(row["aligned_pos"]) + 5)
        if key not in prem:
            continue
        realised = prem[key] - float(row["premium_bp"])
        store.loc[idx, "realised_bp"] = realised
        for model in ["gbm", "ridge", "persistence", "trailmed"]:
            store.loc[idx, f"err_{model}_bp"] = abs(
                float(row[f"pred_{model}_bp"]) - realised)
        store.loc[idx, "scored"] = True
        n_new += 1
    return n_new


def issue_batch(store: pd.DataFrame, feats: pd.DataFrame,
                issue_date: str) -> dict | None:
    d_star = feats["date"].max()
    if not store.empty:
        last_feature = pd.to_datetime(store["feature_date"]).max()
        if d_star <= last_feature:
            return None
    sample = T.build_sample(feats, "T5a-C5", "STD")
    cols = T.design_columns("STD")
    axis = pd.DatetimeIndex(sorted(set(sample["date"].unique()) | {d_star}))
    pos_of = {d: i for i, d in enumerate(axis)}
    pos_star = pos_of[pd.Timestamp(d_star)]
    sample_pos = sample["date"].map(pos_of).to_numpy()
    keep = ((sample["label_end_date"] < pd.Timestamp(d_star)).to_numpy()
            & (sample_pos <= pos_star - 6))
    train = sample.loc[keep]
    assert len(train) > 0, "forward training set is empty"
    Xtr = train[cols].to_numpy(dtype=float)
    ytr = train["y"].to_numpy(dtype=float)

    day = feats[feats["date"] == d_star].copy()
    eligible = day[(day["aligned_pos"] >= 21)
                   & day[T.NUMERIC_FEATURES].notna().all(axis=1)].copy()
    for t in T.STD_FUNDS:
        eligible[f"fund_{t}"] = (eligible["ticker"] == t).astype(float)
    Xte = eligible[cols].to_numpy(dtype=float)

    pred_ridge, info = T.ridge_fit_predict_t5a(Xtr, ytr, Xte)
    pred_gbm, _ = M.gbm_reg_fit_predict(Xtr, ytr, Xte)
    alpha = float(info["alpha"])

    rows = []
    for j, (_, r) in enumerate(eligible.iterrows()):
        rows.append({
            "issue_date": issue_date,
            "feature_date": pd.Timestamp(d_star).date().isoformat(),
            "ticker": r["ticker"],
            "aligned_pos": int(r["aligned_pos"]),
            "premium_bp": float(r["premium_bp"]),
            "pred_gbm_bp": float(pred_gbm[j]),
            "pred_ridge_bp": float(pred_ridge[j]),
            "pred_persistence_bp": 0.0,
            "pred_trailmed_bp": float(r["prem_med_252_bp"]
                                      - r["prem_level_bp"]),
            "ridge_alpha": alpha,
            "vix_level": float(r["vix_level"]),
            "vix_stress": bool(r["vix_stress"] >= 0.5),
            "credit": r["ticker"] in T.CREDIT_FUNDS,
            "scored": False,
            "realised_bp": np.nan,
            "err_gbm_bp": np.nan,
            "err_ridge_bp": np.nan,
            "err_persistence_bp": np.nan,
            "err_trailmed_bp": np.nan,
        })
    issued = pd.DataFrame(rows, columns=STORE_COLUMNS)
    skipped = [t for t in T.STD_FUNDS if t not in set(issued["ticker"])]
    store = pd.concat([store, issued], ignore_index=True)
    store["scored"] = store["scored"].astype(bool)
    return {
        "feature_date": pd.Timestamp(d_star).date().isoformat(),
        "funds_issued": int(len(issued)),
        "predictions_issued": int(4 * len(issued)),
        "skipped_funds": skipped,
        "ridge_alpha": alpha,
        "n_train": int(len(train)),
        "_store": store,
    }


def _mae_block(d: pd.DataFrame) -> dict:
    out = {"n": int(len(d)), "eff": round(len(d) / 5.0, 1)}
    for model in ["gbm", "ridge", "persistence", "trailmed"]:
        out[model] = float(d[f"err_{model}_bp"].mean())
    if len(d) and out["persistence"] > 0:
        out["gbm_ratio"] = out["gbm"] / out["persistence"]
        out["ridge_ratio"] = out["ridge"] / out["persistence"]
    return out


def render_scorecard(store: pd.DataFrame, run: dict):
    batches = (store.groupby("feature_date")["ticker"].count()
               if not store.empty else pd.Series(dtype=int))
    scored = store[store["scored"]] if not store.empty else store
    lines = []
    lines.append("# Prediction scorecard — Target 5a forward test "
                 "(standard-NAV stratum)")
    lines.append("")
    lines.append("Protocol: `knowledge/target5a_forward_protocol_2026-11-03.md` "
                 "(frozen 2026-11-03, before the first issuance).")
    lines.append("Trial: 8331 (one trial for the whole forward test; "
                 "cross-reference Target 5a experiment trials 7830–7837, "
                 "in particular 7833 primary and 7832 secondary).")
    lines.append("Configuration: T5a-C5, the five-trading-day change in "
                 "premium (market close / issuer-published NAV − 1, basis "
                 "points), per fund, for the 57 standard-NAV funds. Primary: "
                 "gradient boosting (in-experiment pooled MAE 14.6919 bp, "
                 "ratio 0.8717 against persistence, 49 of 57 funds). "
                 "Secondary: ridge (16.5235 bp, ratio 0.9803, 37 of 57 "
                 "funds). Benchmarks issued beside the models: persistence "
                 "(zero change) and the trailing-median change.")
    lines.append("Verdict standard: no verdict before 63 trading days of "
                 "issuance, graded once in the week after the final batch "
                 "matures. Kill condition at grading: primary forward MAE "
                 "ratio against persistence at or above 1.000, or fewer "
                 "than 29 of 57 funds beating persistence.")
    lines.append("")
    lines.append("## State")
    lines.append("")
    lines.append(f"- Batches issued: {len(batches)}")
    lines.append(f"- Predictions issued: {len(store)} fund predictions "
                 f"(four issued values each: gradient boosting, ridge, "
                 f"persistence, trailing median)")
    lines.append(f"- Scored: {int(store['scored'].sum()) if not store.empty else 0}; "
                 f"outstanding: {int((~store['scored']).sum()) if not store.empty else 0}")
    if not store.empty:
        lines.append("- Batches (feature date: funds issued): "
                     + "; ".join(f"{d}: {n}" for d, n in batches.items()))
    lines.append(f"- Latest run: {run['run_date']} — scored "
                 f"{run['scored_new']} newly matured predictions; "
                 + (f"issued a batch at feature date "
                    f"{run['batch']['feature_date']} for "
                    f"{run['batch']['funds_issued']} funds"
                    if run.get("batch") else "no new batch (panel did not "
                    "advance beyond the last issued feature date)")
                 + ".")
    if run.get("batch") and run["batch"]["skipped_funds"]:
        lines.append(f"- Funds skipped in the latest batch (no eligible "
                     f"aligned row on the feature date): "
                     f"{', '.join(run['batch']['skipped_funds'])}")
    lines.append("")
    lines.append("## Running record (record only — no verdict before the "
                 "63-day grading)")
    lines.append("")
    if len(scored) == 0:
        lines.append("None yet — no issued batch has matured. The first "
                     "batch matures five trading days after its feature "
                     "date.")
    else:
        pooled = _mae_block(scored)
        lines.append(f"Pooled over {pooled['n']} scored predictions "
                     f"({pooled['eff']} effective independent outcomes): "
                     f"gradient boosting MAE {pooled['gbm']:.4f} bp; ridge "
                     f"{pooled['ridge']:.4f} bp; persistence "
                     f"{pooled['persistence']:.4f} bp; trailing median "
                     f"{pooled['trailmed']:.4f} bp; gradient boosting ratio "
                     f"against persistence {pooled['gbm_ratio']:.4f}.")
        lines.append("")
        lines.append("| Slice | Scored | Eff. outcomes | GBM MAE (bp) | "
                     "Ridge MAE (bp) | Persistence MAE (bp) | Trail-med "
                     "MAE (bp) | GBM ratio |")
        lines.append("|---|---|---|---|---|---|---|---|")
        slices = [("Pooled", scored),
                  ("Stress (VIX ≥ 30)", scored[scored["vix_stress"]]),
                  ("Calm (VIX < 30)", scored[~scored["vix_stress"]]),
                  ("Credit (7 funds)", scored[scored["credit"]]),
                  ("Non-credit", scored[~scored["credit"]])]
        for name, sub in slices:
            if len(sub) == 0:
                lines.append(f"| {name} | 0 | 0.0 | — | — | — | — | — |")
                continue
            b = _mae_block(sub)
            lines.append(f"| {name} | {b['n']} | {b['eff']} | "
                         f"{b['gbm']:.4f} | {b['ridge']:.4f} | "
                         f"{b['persistence']:.4f} | {b['trailmed']:.4f} | "
                         f"{b.get('gbm_ratio', float('nan')):.4f} |")
        lines.append("")
        lines.append("Per-fund scored record (funds with at least one scored "
                     "prediction):")
        lines.append("")
        lines.append("| Fund | Scored | GBM MAE (bp) | Persistence MAE (bp) "
                     "| Beats persistence |")
        lines.append("|---|---|---|---|---|")
        for t, sub in scored.groupby("ticker"):
            g = float(sub["err_gbm_bp"].mean())
            p = float(sub["err_persistence_bp"].mean())
            lines.append(f"| {t} | {len(sub)} | {g:.4f} | {p:.4f} | "
                         f"{'yes' if g < p else 'no'} |")
    lines.append("")
    text = "\n".join(lines) + "\n"

    def _write():
        SCORECARD_PATH.write_text(text, encoding="utf-8")

    def _verify():
        assert SCORECARD_PATH.read_text(encoding="utf-8") == text

    write_with_retry(SCORECARD_PATH, _write, _verify)


def main() -> int:
    issue_date = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    store = load_store()
    panel, feats = build_live_features()
    scored_new = score_outstanding(store, panel)
    batch = issue_batch(store, feats, issue_date)
    if batch is not None:
        store = batch.pop("_store")
    save_store(store)
    run = {"run_date": issue_date, "scored_new": int(scored_new),
           "batch": batch}
    render_scorecard(store, run)
    summary = {
        "status": "ok",
        "run_date": issue_date,
        "as_of_feature_date": str(feats["date"].max().date()),
        "scored_new": int(scored_new),
        "batch": ({k: v for k, v in batch.items()} if batch else None),
        "cumulative": {
            "fund_predictions": int(len(store)),
            "scored": int(store["scored"].sum()) if len(store) else 0,
            "outstanding": int((~store["scored"]).sum()) if len(store) else 0,
            "batches": int(store["feature_date"].nunique()) if len(store) else 0,
        },
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
