"""Driver for ML Target 5c (per-fund ETF return prediction).

Frozen specification:
``~/workspace/goals/quant-research-backtest-engine/files/ml_target5c_experiment_spec_frozen_2026-11-04.md``
(frozen 2026-11-04, before any Target 5c fit; trials booked in the
programme trials ledger on 2026-11-04, rows 8451-8454 — this driver
books nothing). Order of operations is part of the specification:
build the panel (section 3) -> run the section 10.1 entry gate as a
distinct phase that fits nothing -> only if the gate passes, fit
the walk-forward ladder of sections 8-9 with per-fold checkpoints
-> write the results by the streaming writer of section 10.3. Any
entry-gate failure halts the build before any fitting (kill
criterion section 12.1). The compute-kill projection of section
12.2 is enforced inside the run phase after the first
gradient-boosting fold completes. The independent verification of
section 10.3 is a separate stage and is not part of this driver.

Phases (``--phase``):
  build     Build data/t5c_ret/panel_t5c.parquet, one fund at a
            time, appended incrementally (section 3, section 10.3).
  gate      Run the section 10.1 entry gate (fits nothing) and
            write reports/ml_target5c_entry_gate_2026-11-04.md and
            reports/ml_target5c_entry_gate_2026-11-04.json.
  run       Walk-forward run (sections 8-9): expanding folds,
            per-fold per-target prediction part files under
            data/t5c_ret/pred_parts/, per-fold checkpoints and the
            run state file, so a host restart resumes at the last
            completed fold. Optional ``--only T5c-R1`` or
            ``--only T5c-R5`` restricts the run to one target; a
            fold is the atomic checkpoint unit and always computes
            all five prediction series (both learned models and the
            three deterministic comparators).
  finalize  Streaming results writer (section 10.3): recomputes
            every section 11 figure from the part files in batches
            of at most 100,000 rows, holding only scalar
            accumulators, under a 2 GB resident-set guard that
            fails the phase with a written record if exceeded.
            Writes reports/ml_target5c_results.json and
            reports/ml_target5c_results_2026-11-04.md.
  all       build -> gate -> run -> finalize, in that order.

Outputs:
  data/t5c_ret/panel_t5c.parquet
  data/t5c_ret/pred_parts/<target>_fold_<fold>.parquet
  data/t5c_ret/run_state.json
  reports/ml_target5c_entry_gate_2026-11-04.{md,json}
  reports/ml_target5c_compute_projection_2026-11-04.md (run phase)
  reports/ml_target5c_results.json
  reports/ml_target5c_results_2026-11-04.md

Offline evaluation only. No forward issuance, no scorecard writes.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine import ml_t5c as T  # noqa: E402

np.random.seed(T.SEED)

RUN_DATE = T.RUN_DATE
REPORTS = ROOT / "reports"
GATE_JSON = REPORTS / f"ml_target5c_entry_gate_{RUN_DATE}.json"
GATE_MD = REPORTS / f"ml_target5c_entry_gate_{RUN_DATE}.md"
PROJECTION_MD = REPORTS / f"ml_target5c_compute_projection_{RUN_DATE}.md"
MEMFAIL_MD = REPORTS / f"ml_target5c_results_memory_failure_{RUN_DATE}.md"
RESULTS_PATH = REPORTS / "ml_target5c_results.json"
RESULTS_DATED = REPORTS / f"ml_target5c_results_{RUN_DATE}.md"
STATE_PATH = T.STATE_PATH
RSS_LIMIT_KB = 2 * 1024 * 1024  # section 10.3: 2 GB resident bound


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_state(state: dict):
    T.robust_write_json(STATE_PATH, state)


# --- Panel build (section 3) ------------------------------------------------------

def build_panel(state: dict, force: bool = False) -> int:
    """Build the panel one fund at a time, appended incrementally
    to a single parquet file via a ParquetWriter; no more than one
    fund's frame is held during construction. Returns the panel row
    count."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if T.PANEL_PATH.exists() and state.get("panel_built") and not force:
        n = pq.ParquetFile(T.PANEL_PATH).metadata.num_rows
        log(f"panel: existing {T.PANEL_PATH} rows={n}")
        return n
    app = T.load_appendix()
    funds = T.eligible_funds(app)
    market = T.load_market_series()
    T.T5C_DIR.mkdir(parents=True, exist_ok=True)
    tmp = T.PANEL_PATH.with_name(T.PANEL_PATH.name + f".tmp{int(time.time())}")
    writer = None
    total = 0
    per_fund = {}
    try:
        for t in funds:
            raw = T.load_fund_raw(t)
            raw["ticker"] = t
            frame = T.build_fund_panel(raw, market)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema)
            writer.write_table(table)
            per_fund[t] = int(len(frame))
            total += int(len(frame))
            del frame, table, raw
        if writer is not None:
            writer.close()
            writer = None
    except Exception:
        if writer is not None:
            writer.close()
        tmp.unlink(missing_ok=True)
        raise

    def _publish():
        import os
        os.replace(tmp, T.PANEL_PATH)
        n = pq.ParquetFile(T.PANEL_PATH).metadata.num_rows
        if n != total:
            raise OSError(f"panel read-back rows {n} != {total}")

    T._retry(_publish, what="publish panel")
    state["panel_built"] = True
    state["panel_rows"] = total
    state["panel_per_fund_rows"] = per_fund
    save_state(state)
    log(f"panel: wrote {T.PANEL_PATH} rows={total} funds={len(funds)}")
    return total


def read_panel_fund(ticker: str) -> pd.DataFrame:
    df = pd.read_parquet(T.PANEL_PATH, filters=[("ticker", "=", ticker)])
    return df.sort_values("pos").reset_index(drop=True)


# --- Entry gate (section 10.1) -----------------------------------------------------

def _independent_fund_values(ticker: str) -> dict[str, np.ndarray]:
    """Independent recomputation of one fund's return series, both
    labels, all 14 features, and the 5-day trailing-median baseline
    input, written as scalar loops directly from the frozen
    specification's formulas and the raw cache files. This code
    path shares no construction code with engine/ml_t5c."""
    raw = pd.read_parquet(T.PRICE_CACHE / f"{ticker}.parquet")
    raw = raw.reset_index().rename(columns={"index": "date"})
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    raw = raw.sort_values("date").reset_index(drop=True)
    raw = raw.loc[raw["date"] <= T.PANEL_END].reset_index(drop=True)
    adj = raw["adjclose"].to_numpy(dtype=float)
    close = raw["close"].to_numpy(dtype=float)
    vol = raw["volume"].to_numpy(dtype=float)
    dates = list(raw["date"])
    n = len(adj)

    def _market(name):
        m = pd.read_parquet(T.PRICE_CACHE / f"{name}.parquet")
        m = m.reset_index().rename(columns={"index": "date"})
        m["date"] = pd.to_datetime(m["date"]).dt.normalize()
        m = m[["date", "close"]].dropna().sort_values("date").reset_index(drop=True)
        return list(m["date"]), m["close"].to_numpy(dtype=float)

    vix_dates, vix_close = _market("^VIX")
    tnx_dates, tnx_close = _market("^TNX")
    spy_dates, spy_close = _market("SPY")

    r = [math.nan] * n
    for i in range(1, n):
        r[i] = (adj[i] / adj[i - 1] - 1.0) * 10_000.0
    ret5cum = [math.nan] * n
    for i in range(5, n):
        ret5cum[i] = (adj[i] / adj[i - 5] - 1.0) * 10_000.0
    logvol = [math.nan] * n
    for i in range(n):
        dv = close[i] * vol[i]
        logvol[i] = math.log(dv) if dv > 0 else math.nan

    def _median(vals):
        return statistics.median(vals) if vals else math.nan

    def _window(vals, i, width, minimum):
        w = [vals[j] for j in range(max(0, i - width + 1), i + 1)
             if math.isfinite(vals[j])]
        return w if len(w) >= minimum else None

    cols: dict[str, list] = {c: [math.nan] * n for c in
                             T.RETURN_AND_FEATURE_COLS + ["y_1", "y_5"]}
    for i in range(n):
        cols["ret_bp"][i] = r[i]
        cols["ret1_bp"][i] = r[i]
        if i >= 5 and math.isfinite(r[i]) and math.isfinite(r[i - 5]):
            cols["ret_chg5_bp"][i] = r[i] - r[i - 5]
        if i >= 21 and math.isfinite(r[i]) and math.isfinite(r[i - 21]):
            cols["ret_chg21_bp"][i] = r[i] - r[i - 21]
        w21 = _window(r, i, 21, 10)
        if w21 is not None:
            cols["ret_med21_bp"][i] = _median(w21)
            cols["absret21_bp"][i] = sum(abs(x) for x in w21) / len(w21)
        w252 = _window(r, i, 252, 20)
        if w252 is not None:
            med = _median(w252)
            cols["ret_med252_bp"][i] = med
            cols["ret_mad252_bp"][i] = _median([abs(x - med) for x in w252])
        cols["ret5cum_bp"][i] = ret5cum[i]
        if math.isfinite(r[i]):
            cols["absret1_bp"][i] = abs(r[i])
        cols["logvol"][i] = logvol[i]
        if i >= 5 and math.isfinite(logvol[i]) and math.isfinite(logvol[i - 5]):
            cols["logvol_chg5"][i] = logvol[i] - logvol[i - 5]
        j = bisect.bisect_right(vix_dates, dates[i]) - 1
        if j >= 0:
            cols["vix_level"][i] = float(vix_close[j])
        j = bisect.bisect_right(tnx_dates, dates[i]) - 1
        if j >= 5:
            cols["tnx_chg5"][i] = float(tnx_close[j] - tnx_close[j - 5])
        j = bisect.bisect_right(spy_dates, dates[i]) - 1
        if j >= 5:
            cols["spy_ret5_bp"][i] = (spy_close[j] / spy_close[j - 5]
                                      - 1.0) * 10_000.0
        w = _window(ret5cum, i, 252, 20)
        if w is not None:
            cols["ret5cum_med252_bp"][i] = _median(w)
        if i + 1 < n:
            cols["y_1"][i] = (adj[i + 1] / adj[i] - 1.0) * 10_000.0
        if i + 5 < n:
            cols["y_5"][i] = (adj[i + 5] / adj[i] - 1.0) * 10_000.0
    return {c: np.asarray(v, dtype=float) for c, v in cols.items()}


def _compare_columns(stored: pd.DataFrame, hand: dict[str, np.ndarray],
                     cols: list[str]) -> dict:
    worst = 0.0
    violations = 0
    nan_mismatch = 0
    for c in cols:
        a = stored[c].to_numpy(dtype=float)
        b = hand[c]
        if len(a) != len(b):
            return {"length_mismatch": [int(len(a)), int(len(b))],
                    "max_abs_diff": float("inf"), "violations": 1,
                    "nan_pattern_mismatches": 1}
        na, nb = np.isnan(a), np.isnan(b)
        nan_mismatch += int((na != nb).sum())
        both = ~(na | nb)
        if both.any():
            diff = np.abs(a[both] - b[both])
            tol = np.maximum(1e-12, 1e-12 * np.abs(a[both]))
            violations += int((diff > tol).sum())
            worst = max(worst, float(diff.max()))
    return {"max_abs_diff": worst, "violations": violations,
            "nan_pattern_mismatches": nan_mismatch}


def run_entry_gate(state: dict) -> dict:
    gate: dict = {"experiment": "ML Target 5c — per-fund ETF return prediction",
                  "gate": "Entry gate (frozen specification section 10.1)",
                  "run_date": RUN_DATE, "checks": {}, "passed": True}

    def record(name, ok, detail):
        gate["checks"][name] = {"pass": bool(ok), "detail": detail}
        gate["passed"] = gate["passed"] and bool(ok)
        log(f"gate: {name}: {'PASS' if ok else 'FAIL'}")

    app = T.load_appendix()

    # Item 0 — frozen specification receipt (section 17 procedure).
    receipt = T.verify_spec_receipt()
    record("0_frozen_specification_receipt",
           receipt["bytes_match"] and receipt["hash_match"]
           and receipt["hash_field_occurrences"] == 1
           and receipt["byte_field_occurrences"] == 1,
           receipt)

    # Item 1 — eligibility assertion (section 10.1 item 1).
    fund_rows = app[app["series_kind"] == "FUND"]
    mismatches = []
    for _, row in app.iterrows():
        t = row["ticker"]
        raw = T.load_fund_raw(t, end=None)
        at_end = raw[raw["date"] <= T.PANEL_END]
        if int(len(at_end)) != int(row["n_rows_at_panel_end_2026-11-03"]):
            mismatches.append({"ticker": t, "reconstructed": int(len(at_end)),
                               "appendix": int(row["n_rows_at_panel_end_2026-11-03"])})
        if len(at_end):
            first_ok = str(at_end["date"].iloc[0].date()) == str(row["first_date"])
            last_ok = str(at_end["date"].iloc[-1].date()) == str(row["last_date_at_panel_end"])
            if not (first_ok and last_ok):
                mismatches.append({"ticker": t, "span_mismatch": {
                    "reconstructed_first": str(at_end["date"].iloc[0].date()),
                    "reconstructed_last": str(at_end["date"].iloc[-1].date()),
                    "appendix_first": str(row["first_date"]),
                    "appendix_last": str(row["last_date_at_panel_end"])}})
    elig1 = T.eligible_funds(app)
    elig5 = T.eligible_funds_r5(app)
    excluded = sorted(set(T.appendix_funds(app)) - set(elig1))
    record("1_eligibility_assertion",
           not mismatches and len(fund_rows) == 145 and len(app) == 147
           and len(elig1) == T.EXPECTED_UNIVERSE
           and len(elig5) == T.EXPECTED_UNIVERSE and elig1 == elig5
           and excluded == sorted(T.EXCLUDED_HISTORY),
           {"appendix_rows": int(len(app)), "fund_rows": int(len(fund_rows)),
            "market_state_rows": int(len(app) - len(fund_rows)),
            "row_count_or_span_mismatches": mismatches,
            "selected_universe_r1_min_399": len(elig1),
            "selected_universe_r5_min_404": len(elig5),
            "excluded_on_history": excluded})

    # Item 2 — hand checks on SPY, SOXS, PRIV (section 10.1 item 2).
    hand_detail = {}
    ok2 = True
    for t in ["SPY", "SOXS", "PRIV"]:
        stored = read_panel_fund(t)
        hand = _independent_fund_values(t)
        cmp_ = _compare_columns(stored, hand,
                                T.RETURN_AND_FEATURE_COLS + ["y_1", "y_5"])
        hand_detail[t] = cmp_
        ok2 = ok2 and cmp_["violations"] == 0 and cmp_["nan_pattern_mismatches"] == 0
    record("2_hand_checks_SPY_SOXS_PRIV", ok2,
           {"tolerance": "max(1e-12, 1e-12 * |stored value|), elementwise, "
                         "with identical NaN patterns",
            "columns_compared": T.RETURN_AND_FEATURE_COLS + ["y_1", "y_5"],
            "positions_compared": "all panel positions of each fund",
            "per_fund": hand_detail})

    # Item 3 — truncation check at 2022-12-31 (section 10.1 item 3).
    market_full = T.load_market_series()
    market_cut = T.load_market_series(end=T.TRUNC_END)
    trunc_detail = {"max_abs_diff": 0.0, "nan_pattern_identical": True,
                    "funds_compared": 0, "rows_compared": 0}
    ok3 = True
    for t in elig1:
        raw_cut = T.load_fund_raw(t, end=T.TRUNC_END)
        raw_cut["ticker"] = t
        cut_frame = T.build_fund_panel(raw_cut, market_cut)
        full_frame = read_panel_fund(t)
        full_cut = full_frame[full_frame["date"] <= T.TRUNC_END].reset_index(drop=True)
        if len(full_cut) != len(cut_frame):
            ok3 = False
            trunc_detail[f"length_mismatch_{t}"] = [int(len(full_cut)),
                                                    int(len(cut_frame))]
            continue
        for c in T.RETURN_AND_FEATURE_COLS:
            a = full_cut[c].to_numpy(dtype=float)
            b = cut_frame[c].to_numpy(dtype=float)
            if not bool((np.isnan(a) == np.isnan(b)).all()):
                trunc_detail["nan_pattern_identical"] = False
                ok3 = False
            both = np.isfinite(a) & np.isfinite(b)
            if both.any():
                d = float(np.max(np.abs(a[both] - b[both])))
                trunc_detail["max_abs_diff"] = max(trunc_detail["max_abs_diff"], d)
        trunc_detail["funds_compared"] += 1
        trunc_detail["rows_compared"] += int(len(cut_frame))
        del cut_frame, full_frame, full_cut, raw_cut
    ok3 = ok3 and trunc_detail["max_abs_diff"] == 0.0 \
        and trunc_detail["nan_pattern_identical"]
    trunc_detail["note"] = ("Labels are excluded from this comparison: a "
                            "label's forward window crosses the truncation "
                            "date by construction. All return, feature, and "
                            "baseline-input columns are compared.")
    record("3_truncation_check_2022_12_31", ok3, trunc_detail)

    # Item 4 — input hashes (section 10.1 item 4).
    hashes = T.input_hashes(app)
    gate["input_hashes"] = hashes
    record("4_input_hashes_recorded", len(hashes) == 148,
           {"n_files_hashed": len(hashes),
            "composition": "145 fund cache files (SPY among them, in its "
                           "dual role as a universe fund and as the "
                           "market-state SPY series), the ^VIX and ^TNX "
                           "market-state cache files, and the frozen "
                           "specification file",
            "hashes_recorded_in": "the input_hashes field of this gate "
                                  "report (JSON and Markdown forms)"})
    return gate


def write_gate_report(gate: dict):
    T.robust_write_json(GATE_JSON, gate)
    lines = []
    lines.append("# ML Target 5c — Entry Gate Report")
    lines.append("")
    lines.append(f"**Date:** {RUN_DATE}. **Specification:** the frozen "
                 "Target 5c experiment specification of 2026-11-04 "
                 "(section 10.1 entry gate). **Function of this gate:** "
                 "every assertion below passes exactly, or no sample is "
                 "constructed and no model is fitted (kill criterion "
                 "section 12.1). This gate fits nothing.")
    lines.append("")
    lines.append(f"**Overall entry gate: "
                 f"{'PASS' if gate['passed'] else 'FAIL'}**")
    lines.append("")
    for name, chk in gate["checks"].items():
        lines.append(f"## {name}: {'PASS' if chk['pass'] else 'FAIL'}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(chk["detail"], indent=1, default=str))
        lines.append("```")
        lines.append("")
    lines.append("## Input hashes (section 10.1 item 4)")
    lines.append("")
    lines.append("SHA-256 of every input file, recorded at the gate:")
    lines.append("")
    for k, v in gate["input_hashes"].items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    T.robust_write_text(GATE_MD, "\n".join(lines) + "\n")
    log(f"gate: wrote {GATE_MD} and {GATE_JSON}")


# --- Walk-forward run (sections 8-9, 10.2, 12.2) --------------------------------------

def part_path(target: str, fold_id: int) -> Path:
    return T.PARTS_DIR / f"{target}_fold_{fold_id:04d}.parquet"


def build_sample_for(target: str, state: dict):
    panel = pd.read_parquet(T.PANEL_PATH)
    funds = T.eligible_funds()
    sample = T.build_sample(panel, target, funds)
    del panel
    folds = T.make_pooled_folds(sample, T.HORIZON_OF[target])
    return sample, folds


def run_phase(state: dict, only: str | None) -> int:
    if not state.get("gate", {}).get("passed"):
        log("run: entry gate has not passed — stopping before any fitting")
        return 1
    if state.get("compute_kill", {}).get("triggered"):
        log("run: compute kill (section 12.2) already triggered — no further fitting")
        return 3
    targets = [only] if only else list(T.TARGETS)
    run_state = state.setdefault("run", {})

    # Plan pass: fold axes and training sizes for both targets (the
    # section 12.2 projection is stated over both gradient-boosting
    # configurations, so both fold axes are computed before fitting).
    plan = {}
    for target in T.TARGETS:
        if target in run_state.get("plan", {}):
            plan[target] = run_state["plan"][target]
            continue
        panel = pd.read_parquet(T.PANEL_PATH)
        sample = T.build_sample(panel, target, T.eligible_funds())
        del panel
        folds = T.make_pooled_folds(sample, T.HORIZON_OF[target])
        plan[target] = {
            "n_sample": int(len(sample)),
            "n_folds": int(len(folds)),
            "folds": [f["meta"] for f in folds],
            "sum_n_train": int(sum(f["meta"]["n_train"] for f in folds)),
        }
        del sample, folds
    run_state["plan"] = plan
    save_state(state)

    for target in targets:
        horizon = T.HORIZON_OF[target]
        sample, folds = build_sample_for(target, state)
        funds = T.eligible_funds()
        n_funds = len(funds)
        Xn = sample[T.NUMERIC_FEATURES].to_numpy(dtype=float)
        fund_idx = sample["fund_idx"].to_numpy(dtype=int)
        dow_idx = sample["dow_idx"].to_numpy(dtype=int)
        dow_cols = sample[["dow_tue", "dow_wed", "dow_thu", "dow_fri"]].to_numpy(dtype=float)
        y_all = sample["y"].to_numpy(dtype=float)
        tstate = run_state.setdefault(target, {"completed_folds": [], "alphas": {},
                                               "gbm_fold_times_sec": {}})
        completed = set(tstate["completed_folds"])
        for f in folds:
            fid = int(f["fold_id"])
            pp = part_path(target, fid)
            if pp.exists() and fid in completed:
                continue
            if pp.exists() and fid not in completed:
                # A part file written before a host loss, whose state
                # update did not persist: verify and adopt it rather
                # than refitting.
                import pyarrow.parquet as pq
                n_rows = pq.ParquetFile(pp).metadata.num_rows
                if n_rows == 5 * int(f["meta"]["n_test"]):
                    tstate["completed_folds"].append(fid)
                    completed.add(fid)
                    save_state(state)
                    log(f"run: {target} fold {fid} adopted from checkpoint file")
                    continue
            tr, te = f["train_idx"], f["test_idx"]
            test_rows = sample.iloc[te]
            pred_ridge, hp = T.ridge_fit_predict_gram(
                Xn, fund_idx, dow_idx, y_all, tr, te, n_funds)
            Xtr = T.design_matrix(Xn, fund_idx, dow_cols, tr, n_funds)
            Xte = T.design_matrix(Xn, fund_idx, dow_cols, te, n_funds)
            t0 = time.perf_counter()
            pred_gbm, _ = T.gbm_fit_predict(Xtr, y_all[tr], Xte)
            gbm_sec = time.perf_counter() - t0
            del Xtr, Xte
            if horizon == 1:
                persistence = test_rows["ret1_bp"].to_numpy(dtype=float)
                trailing_median = test_rows["ret_med252_bp"].to_numpy(dtype=float)
            else:
                persistence = test_rows["ret5cum_bp"].to_numpy(dtype=float)
                trailing_median = test_rows["ret5cum_med252_bp"].to_numpy(dtype=float)
            preds = {
                "persistence": persistence,
                "trailing_median": trailing_median,
                "zero": np.zeros(len(te), dtype=float),
                "ridge": pred_ridge,
                "gbm": pred_gbm,
            }
            parts = []
            for model in T.MODELS:
                parts.append(pd.DataFrame({
                    "ticker": test_rows["ticker"].to_numpy(),
                    "date": test_rows["date"].to_numpy(),
                    "fold": fid,
                    "target": target,
                    "horizon": horizon,
                    "model": model,
                    "y_true": y_all[te],
                    "y_pred": preds[model],
                    "vix_level": test_rows["vix_level"].to_numpy(dtype=float),
                    "alpha": hp["alpha"] if model == "ridge" else np.nan,
                }))
            fold_frame = pd.concat(parts, ignore_index=True)
            T.robust_write_parquet(pp, fold_frame, expected_rows=5 * len(te))
            del fold_frame, parts
            tstate["completed_folds"].append(fid)
            tstate["alphas"][str(fid)] = hp["alpha"]
            tstate["gbm_fold_times_sec"][str(fid)] = gbm_sec
            completed.add(fid)
            save_state(state)
            log(f"run: {target} fold {fid}/{len(folds) - 1} written "
                f"(gbm {gbm_sec:.1f}s, ridge alpha {hp['alpha']})")

            # Compute kill (section 12.2): after the first gradient-
            # boosting fold completes, project both configurations.
            if "projection" not in run_state:
                rate = gbm_sec / int(f["meta"]["n_train"])  # seconds per training row
                proj = {}
                total_sec = 0.0
                for tgt in T.TARGETS:
                    p_sec = rate * plan[tgt]["sum_n_train"]
                    proj[tgt] = {"projected_fit_hours": p_sec / 3600.0,
                                 "sum_n_train_over_folds": plan[tgt]["sum_n_train"],
                                 "n_folds": plan[tgt]["n_folds"]}
                    total_sec += p_sec
                projection = {
                    "basis_target": target, "basis_fold": fid,
                    "basis_fold_gbm_seconds": gbm_sec,
                    "basis_fold_n_train": int(f["meta"]["n_train"]),
                    "seconds_per_training_row": rate,
                    "per_configuration": proj,
                    "projected_total_hours_both_configurations": total_sec / 3600.0,
                    "threshold_hours": 72.0,
                    "exceeds_threshold": bool(total_sec / 3600.0 > 72.0),
                    "method": "The first completed gradient-boosting fold's "
                              "measured fit time gives a seconds-per-training-row "
                              "rate; each configuration's projected fit time is "
                              "that rate multiplied by the sum of its folds' "
                              "training-row counts on the frozen fold axis.",
                }
                run_state["projection"] = projection
                save_state(state)
                write_projection_report(projection)
                if projection["exceeds_threshold"]:
                    state["compute_kill"] = {
                        "triggered": True,
                        "criterion": "Frozen specification section 12.2",
                        "projected_total_hours": total_sec / 3600.0,
                        "threshold_hours": 72.0,
                        "report": str(PROJECTION_MD),
                    }
                    save_state(state)
                    log("run: COMPUTE KILL (section 12.2) — projected "
                        f"{total_sec / 3600.0:.1f}h exceeds 72h; halting, "
                        "no further fitting")
                    return 3
        del sample, Xn
        log(f"run: {target} complete — folds={len(folds)}")
    return 0


def write_projection_report(projection: dict):
    lines = []
    lines.append("# ML Target 5c — Gradient-Boosting Compute Projection")
    lines.append("")
    lines.append(f"**Date:** {RUN_DATE}. **Criterion:** frozen specification "
                 "section 12.2 (compute kill). The first gradient-boosting "
                 "fold's measured fit time projects both gradient-boosting "
                 "configurations' total fit time; if the projection exceeds "
                 "72 hours, the run stops and no further fitting takes place.")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(projection, indent=1, default=str))
    lines.append("```")
    lines.append("")
    lines.append(f"**Projected total for both configurations:** "
                 f"{projection['projected_total_hours_both_configurations']:.2f} "
                 f"hours. **Threshold:** 72 hours. **Outcome:** "
                 f"{'the projection exceeds the threshold; the run is halted.' if projection['exceeds_threshold'] else 'the projection does not exceed the threshold; the run proceeds.'}")
    lines.append("")
    T.robust_write_text(PROJECTION_MD, "\n".join(lines) + "\n")
    log(f"run: wrote {PROJECTION_MD}")


# --- Finalize: streaming results writer (sections 10.3, 11) ----------------------------

def _rss_exceeded() -> int:
    """Peak resident set size of this process, in kilobytes (Linux
    ru_maxrss). Returns the value so the caller can record it."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _new_model_acc():
    return {"sum_abs": 0.0, "sum_sq": 0.0, "n": 0, "dir": 0}


def finalize_phase(state: dict) -> int:
    import pyarrow.parquet as pq

    plan = state.get("run", {}).get("plan", {})
    files = sorted(T.PARTS_DIR.glob("*.parquet")) if T.PARTS_DIR.exists() else []
    if not files:
        log("finalize: no prediction part files — stopping")
        return 1
    if plan:
        missing = []
        for target, p in plan.items():
            for meta in p["folds"]:
                if not part_path(target, int(meta["fold"])).exists():
                    missing.append(f"{target} fold {meta['fold']}")
        if missing:
            log(f"finalize: missing part files {missing[:5]} "
                f"({len(missing)} total) — stopping")
            return 1

    # Scalar accumulators only, keyed by (target, model, fund, fold,
    # regime slice), per section 10.3.
    pooled: dict = {}
    regime: dict = {}
    per_fund: dict = {}
    per_fold: dict = {}
    cell_counts: dict = {}
    base_counts: dict = {}
    alphas: dict = {}
    peak_rss_kb = 0

    def acc_for(store, key):
        return store.setdefault(key, _new_model_acc())

    def _merge(store, key, sum_abs, sum_sq, n, dir_n):
        a = acc_for(store, key)
        a["sum_abs"] += float(sum_abs); a["sum_sq"] += float(sum_sq)
        a["n"] += int(n); a["dir"] += int(dir_n)

    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=100_000):
            # Batch-level group sums (at most 100,000 rows in memory),
            # merged immediately into the scalar accumulators; no
            # batch frame or group frame is retained.
            d = batch.to_pandas()
            d["_ae"] = (d["y_true"] - d["y_pred"]).abs()
            d["_se"] = (d["y_true"] - d["y_pred"]) ** 2
            d["_dir"] = (np.sign(d["y_pred"]) == np.sign(d["y_true"])).astype(int)
            d["_reg"] = np.where(d["vix_level"] >= 30.0, "vix_ge_30", "vix_lt_30")
            agg = ["_ae", "_se", "_dir"]
            for keys, store, keyfn in [
                (["target", "model"], pooled, lambda r: (r[0], r[1])),
                (["target", "model", "_reg"], regime, lambda r: (r[0], r[1], r[2])),
                (["target", "model", "ticker"], per_fund, lambda r: (r[0], r[1], r[2])),
                (["target", "model", "fold"], per_fold, lambda r: (r[0], r[1], int(r[2]))),
            ]:
                g = d.groupby(keys, sort=False)[agg].sum()
                cnt = d.groupby(keys, sort=False).size()
                for idx in g.index:
                    key = keyfn(idx if isinstance(idx, tuple) else (idx,))
                    row = g.loc[idx]
                    _merge(store, key, row["_ae"], row["_se"],
                           int(cnt.loc[idx]), row["_dir"])
            for idx, n in d.groupby(["target", "model", "ticker", "fold"],
                                    sort=False).size().items():
                key = (idx[0], idx[1], idx[2], int(idx[3]))
                cell_counts[key] = cell_counts.get(key, 0) + int(n)
            pers = d[d["model"] == "persistence"]
            for target, g in pers.groupby("target", sort=False):
                bc = base_counts.setdefault(
                    target, {"n": 0, "pos": 0, "zero": 0, "neg": 0})
                bc["n"] += int(len(g))
                bc["pos"] += int((g["y_true"] > 0).sum())
                bc["zero"] += int((g["y_true"] == 0).sum())
                bc["neg"] += int((g["y_true"] < 0).sum())
            ridge_rows = d[(d["model"] == "ridge") & np.isfinite(d["alpha"])]
            for (target, fold), g in ridge_rows.groupby(["target", "fold"],
                                                        sort=False):
                alphas.setdefault(str(target), {})[int(fold)] = float(g["alpha"].iloc[0])
            del d
            peak_rss_kb = max(peak_rss_kb, _rss_exceeded())
            if peak_rss_kb > RSS_LIMIT_KB:
                record = {
                    "phase": "finalize", "failure": "resident set size bound "
                    "of section 10.3 exceeded", "peak_rss_kb": peak_rss_kb,
                    "limit_kb": RSS_LIMIT_KB, "file_being_read": str(path),
                    "date": RUN_DATE,
                }
                T.robust_write_text(
                    MEMFAIL_MD,
                    "# ML Target 5c — Results Writer Memory Failure\n\n"
                    "The streaming results writer exceeded the 2 GB resident "
                    "bound of the frozen specification section 10.3. The "
                    "phase has failed with this written record; it is not "
                    "retried unchanged (kill criterion section 12.3).\n\n"
                    "```json\n" + json.dumps(record, indent=1) + "\n```\n")
                state["finalize_memory_failure"] = record
                save_state(state)
                log("finalize: RSS bound exceeded — phase failed with written record")
                return 4
    # Comparison-sample check from the cell counts: identical
    # (fund, fold) counts across all five models for each target.
    comparison_ok = True
    cells_by_target: dict = {}
    for (target, model, fund, fold), n in cell_counts.items():
        cells_by_target.setdefault(target, {}).setdefault((fund, fold), {})[model] = n
    for target, cells in cells_by_target.items():
        for cell, by_model in cells.items():
            if set(by_model) != set(T.MODELS) or len(set(by_model.values())) != 1:
                comparison_ok = False
    if not comparison_ok:
        log("finalize: comparison-sample check failed — models do not share "
            "identical test rows")
        state["finalize_comparison_sample_ok"] = False
        save_state(state)
        return 1

    results = _assemble_results(pooled, regime, per_fund, per_fold,
                                 base_counts, alphas, plan, peak_rss_kb,
                                 comparison_ok)
    T.robust_write_json(RESULTS_PATH, results)
    write_results_report(results)
    state["finalize"] = {"results": str(RESULTS_PATH), "report": str(RESULTS_DATED),
                         "peak_rss_kb": peak_rss_kb,
                         "comparison_sample_identical": comparison_ok}
    save_state(state)
    log(f"finalize: wrote {RESULTS_PATH} and {RESULTS_DATED} "
        f"(peak RSS {peak_rss_kb / 1024:.0f} MB)")
    return 0


def _metrics_from_acc(a: dict, pers_mae, med_mae, zero_mae) -> dict:
    if a["n"] == 0:
        return {"n": 0, "mae_bp": None, "rmse_bp": None,
                "mae_ratio_vs_persistence": None,
                "mae_ratio_vs_trailing_median": None,
                "mae_ratio_vs_zero_forecast": None,
                "directional_accuracy": None}
    mae = a["sum_abs"] / a["n"]
    return {
        "n": a["n"], "mae_bp": mae,
        "rmse_bp": math.sqrt(a["sum_sq"] / a["n"]),
        "mae_ratio_vs_persistence": mae / pers_mae if pers_mae else None,
        "mae_ratio_vs_trailing_median": mae / med_mae if med_mae else None,
        "mae_ratio_vs_zero_forecast": mae / zero_mae if zero_mae else None,
        "directional_accuracy": a["dir"] / a["n"],
    }


def _assemble_results(pooled, regime, per_fund, per_fold, base_counts,
                      alphas, plan, peak_rss_kb, comparison_ok) -> dict:
    funds = T.eligible_funds()
    out = {
        "experiment": "ML Target 5c — per-fund ETF return prediction",
        "run_date": RUN_DATE,
        "specification": str(T.SPEC_PATH),
        "trials": {"booked": 4, "ledger_rows": [8451, 8452, 8453, 8454],
                   "note": "Booked at the freeze on 2026-11-04, before any "
                           "fitting. The deterministic comparators book no "
                           "trials."},
        "comparison_sample_identical_across_models": comparison_ok,
        "results_writer": {"streaming_batch_rows": 100_000,
                           "peak_rss_kb": peak_rss_kb,
                           "rss_limit_kb": RSS_LIMIT_KB},
        "targets": {},
        "configurations": {},
    }
    for target, horizon in T.TARGETS.items():
        bc = base_counts.get(target, {"n": 0, "pos": 0, "zero": 0, "neg": 0})
        n_test = bc["n"]
        maes = {}
        for model in T.MODELS:
            a = pooled.get((target, model), _new_model_acc())
            maes[model] = a["sum_abs"] / a["n"] if a["n"] else None
        tgt = {
            "horizon": horizon, "n_test": n_test,
            "effective_independent_outcomes": n_test if horizon == 1 else n_test / 5.0,
            "base_rates": {
                "share_label_positive": bc["pos"] / n_test if n_test else None,
                "share_label_zero": bc["zero"] / n_test if n_test else None,
                "share_label_negative": bc["neg"] / n_test if n_test else None,
                "n_label_positive": bc["pos"], "n_label_zero": bc["zero"],
                "n_label_negative": bc["neg"],
            },
            "alphas_by_fold": {str(k): v for k, v in
                               sorted(alphas.get(target, {}).items())},
            "folds": plan.get(target, {}).get("folds", []),
            "models": {},
            "per_fund": {},
            "verdicts": {},
        }
        for model in T.MODELS:
            a = pooled.get((target, model), _new_model_acc())
            entry = _metrics_from_acc(a, maes["persistence"],
                                      maes["trailing_median"], maes["zero"])
            entry["regime_slice"] = {}
            for reg in ["vix_ge_30", "vix_lt_30"]:
                ra = regime.get((target, model, reg), _new_model_acc())
                entry["regime_slice"][reg] = _metrics_from_acc(
                    ra, maes["persistence"], maes["trailing_median"], maes["zero"])
            entry["per_fold"] = {}
            for (tg, md, fold), fa in sorted(per_fold.items()):
                if tg == target and md == model:
                    entry["per_fold"][str(fold)] = _metrics_from_acc(
                        fa, maes["persistence"], maes["trailing_median"],
                        maes["zero"])
            tgt["models"][model] = entry
        for fund in funds:
            fentry = {"n_test": per_fund.get((target, "persistence", fund),
                                             _new_model_acc())["n"]}
            for model in T.MODELS:
                fa = per_fund.get((target, model, fund), _new_model_acc())
                fentry[f"mae_{model}_bp"] = (fa["sum_abs"] / fa["n"]
                                             if fa["n"] else None)
            for model in T.LEARNED:
                fentry[f"{model}_beats_persistence"] = bool(
                    fentry[f"mae_{model}_bp"] is not None
                    and fentry["mae_persistence_bp"] is not None
                    and fentry[f"mae_{model}_bp"] < fentry["mae_persistence_bp"])
            tgt["per_fund"][fund] = fentry
        for model in T.LEARNED:
            m = tgt["models"][model]
            pooled_leg = bool(m["mae_bp"] is not None
                              and m["mae_bp"] < maes["persistence"]
                              and m["mae_bp"] < maes["trailing_median"]
                              and m["mae_bp"] < maes["zero"])
            n_beat = int(sum(1 for fund in funds
                             if tgt["per_fund"][fund][f"{model}_beats_persistence"]))
            per_fund_leg = bool(n_beat >= T.GRAD_MIN_FUNDS)
            verdict = {
                "verdict": "GRADUATES" if (pooled_leg and per_fund_leg)
                           else "DOES NOT GRADUATE",
                "pooled_leg": {
                    "pass": pooled_leg,
                    "model_mae_bp": m["mae_bp"],
                    "persistence_mae_bp": maes["persistence"],
                    "trailing_median_mae_bp": maes["trailing_median"],
                    "zero_forecast_mae_bp": maes["zero"],
                    "requirement": "Model pooled MAE strictly below the "
                                   "persistence MAE, strictly below the "
                                   "trailing-median MAE, and strictly below "
                                   "the zero-forecast MAE, on the identical "
                                   "comparison sample (section 11).",
                },
                "per_fund_leg": {
                    "pass": per_fund_leg,
                    "funds_beating_persistence": n_beat,
                    "denominator_funds": len(funds),
                    "required": T.GRAD_MIN_FUNDS,
                    "requirement": "The configuration beats persistence in "
                                   "at least 69 of 137 universe funds "
                                   "(section 11).",
                },
            }
            tgt["verdicts"][model] = verdict
            out["configurations"][f"{target}_{model}"] = {
                "target": target, "model": model, "n_test": n_test,
                "effective_independent_outcomes":
                    tgt["effective_independent_outcomes"],
                "mae_bp": m["mae_bp"], "rmse_bp": m["rmse_bp"],
                "mae_ratio_vs_persistence": m["mae_ratio_vs_persistence"],
                "mae_ratio_vs_trailing_median": m["mae_ratio_vs_trailing_median"],
                "mae_ratio_vs_zero_forecast": m["mae_ratio_vs_zero_forecast"],
                "directional_accuracy": m["directional_accuracy"],
                "base_rates": tgt["base_rates"],
                "regime_slice": m["regime_slice"],
                "verdict": verdict,
            }
        out["targets"][target] = tgt
    return out


def fmt(x, digits=4):
    if x is None:
        return "—"
    if isinstance(x, float) and (abs(x) >= 10000 or (0 < abs(x) < 1e-4)):
        return f"{x:.6g}"
    return f"{x:.{digits}f}"


def write_results_report(results: dict):
    lines = []
    lines.append("# ML Target 5c — Per-Fund ETF Return Prediction: Results")
    lines.append("")
    lines.append(f"**Run date:** {RUN_DATE}. **Specification:** the frozen "
                 "Target 5c experiment specification of 2026-11-04. "
                 "**Question (section 1):** for each fund in the 137-fund "
                 "universe and for each of the two return targets, can a "
                 "pooled walk-forward model predict the fund's return "
                 "better than the deterministic comparators — better in "
                 "the pooled test sample, and in a majority of the "
                 "universe funds — using only information knowable at "
                 "the prediction date?")
    lines.append("")
    lines.append("**Metric definitions (section 11, plain language).** "
                 "MAE (mean absolute error), in basis points, is the "
                 "average size of the forecast miss, ignoring direction; "
                 "lower is better and 0 is perfect. RMSE (root mean "
                 "squared error), in basis points, penalises large "
                 "misses more. An MAE ratio below 1 means the model "
                 "missed by less, on average, than the comparator it is "
                 "divided by. Directional accuracy is the share of test "
                 "rows on which the sign of the prediction equals the "
                 "sign of the realised return (the numpy sign tie rule); "
                 "it is reported beside the realised base rates, the "
                 "shares of test rows whose realised return was "
                 "positive, exactly zero, and negative. Every 5-day "
                 "figure is reported with its effective outcome count, "
                 "computed as n_test divided by 5, because five "
                 "consecutive overlapping 5-day labels are not five "
                 "independent outcomes. Graduation (section 11) "
                 "requires both legs: pooled MAE strictly below the "
                 "persistence MAE, strictly below the trailing-median "
                 "MAE, and strictly below the zero-forecast MAE on the "
                 "identical comparison sample; and a per-fund majority "
                 "against persistence, at least 69 of 137 funds. "
                 "Graduation is a verdict within this experiment only "
                 "and authorises no forward issuance and no trading "
                 "claim.")
    lines.append("")
    for target, horizon in T.TARGETS.items():
        tgt = results["targets"][target]
        lines.append(f"## {target} ({'next-day' if horizon == 1 else '5-trading-day'} return)")
        lines.append("")
        lines.append(f"Test rows: {tgt['n_test']}. Effective independent "
                     f"outcomes: {fmt(tgt['effective_independent_outcomes'], 1)}. "
                     f"Realised base rates: positive "
                     f"{fmt(tgt['base_rates']['share_label_positive'] * 100 if tgt['base_rates']['share_label_positive'] is not None else None, 2)}%, "
                     f"exactly zero "
                     f"{fmt(tgt['base_rates']['share_label_zero'] * 100 if tgt['base_rates']['share_label_zero'] is not None else None, 2)}%, "
                     f"negative "
                     f"{fmt(tgt['base_rates']['share_label_negative'] * 100 if tgt['base_rates']['share_label_negative'] is not None else None, 2)}%.")
        lines.append("")
        lines.append("| Model | n_test | MAE (bp) | RMSE (bp) | MAE vs persistence | MAE vs trailing median | MAE vs zero forecast | Directional accuracy |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for model in T.MODELS:
            m = tgt["models"][model]
            lines.append(
                f"| {model} | {m['n']} | {fmt(m['mae_bp'])} | {fmt(m['rmse_bp'])} | "
                f"{fmt(m['mae_ratio_vs_persistence'])} | "
                f"{fmt(m['mae_ratio_vs_trailing_median'])} | "
                f"{fmt(m['mae_ratio_vs_zero_forecast'])} | "
                f"{fmt(m['directional_accuracy'])} |")
        lines.append("")
        lines.append("### Regime slice (VIX level at the prediction date)")
        lines.append("")
        lines.append("| Model | Slice | n | MAE (bp) | RMSE (bp) | Directional accuracy |")
        lines.append("|---|---|---|---|---|---|")
        for model in T.MODELS:
            for reg, label in [("vix_ge_30", "VIX ≥ 30"), ("vix_lt_30", "VIX < 30")]:
                r = tgt["models"][model]["regime_slice"][reg]
                lines.append(f"| {model} | {label} | {r['n']} | {fmt(r['mae_bp'])} | "
                             f"{fmt(r['rmse_bp'])} | {fmt(r['directional_accuracy'])} |")
        lines.append("")
        lines.append("### Verdicts (section 11, both legs stated)")
        lines.append("")
        for model in T.LEARNED:
            v = tgt["verdicts"][model]
            lines.append(f"- **{target} {model}: {v['verdict']}.** Pooled leg: "
                         f"{'pass' if v['pooled_leg']['pass'] else 'fail'} "
                         f"(model MAE {fmt(v['pooled_leg']['model_mae_bp'])} bp; "
                         f"persistence {fmt(v['pooled_leg']['persistence_mae_bp'])} bp; "
                         f"trailing median {fmt(v['pooled_leg']['trailing_median_mae_bp'])} bp; "
                         f"zero forecast {fmt(v['pooled_leg']['zero_forecast_mae_bp'])} bp). "
                         f"Per-fund leg: "
                         f"{'pass' if v['per_fund_leg']['pass'] else 'fail'} "
                         f"({v['per_fund_leg']['funds_beating_persistence']} of "
                         f"{v['per_fund_leg']['denominator_funds']} funds beat "
                         f"persistence; {v['per_fund_leg']['required']} required).")
        lines.append("")
        lines.append("### Per-fund MAE table (basis points)")
        lines.append("")
        lines.append("| Fund | n_test | Persistence | Trailing median | Zero forecast | Ridge | GBM | Ridge beats persistence | GBM beats persistence |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        n_beat = {m: 0 for m in T.LEARNED}
        for fund in T.eligible_funds():
            fe = tgt["per_fund"][fund]
            for m in T.LEARNED:
                n_beat[m] += int(bool(fe[f"{m}_beats_persistence"]))
            lines.append(
                f"| {fund} | {fe['n_test']} | {fmt(fe['mae_persistence_bp'])} | "
                f"{fmt(fe['mae_trailing_median_bp'])} | {fmt(fe['mae_zero_bp'])} | "
                f"{fmt(fe['mae_ridge_bp'])} | {fmt(fe['mae_gbm_bp'])} | "
                f"{'yes' if fe['ridge_beats_persistence'] else 'no'} | "
                f"{'yes' if fe['gbm_beats_persistence'] else 'no'} |")
        lines.append("")
        lines.append(f"Funds beating persistence: ridge {n_beat['ridge']} of "
                     f"137; gradient boosting {n_beat['gbm']} of 137 "
                     f"(69 required for the per-fund leg).")
        lines.append("")
    lines.append("## Trials accounting (section 13)")
    lines.append("")
    lines.append("Exactly 4 learned configurations were booked in the "
                 "programme trials ledger on 2026-11-04, before any "
                 "fitting (rows 8451-8454: T5c-R1 ridge, T5c-R1 gradient "
                 "boosting, T5c-R5 ridge, T5c-R5 gradient boosting). The "
                 "persistence, trailing-median, and zero-forecast "
                 "comparators are deterministic and book no trials. All "
                 "four configurations are reported; there is no "
                 "selection among them.")
    lines.append("")
    lines.append("## Method and limits")
    lines.append("")
    lines.append("The results above were recomputed from the stored "
                 "per-fold prediction part files by the streaming "
                 "results writer of section 10.3 (batches of at most "
                 "100,000 rows; scalar accumulators only; peak resident "
                 f"set size {results['results_writer']['peak_rss_kb'] / 1024:.0f} "
                 "MB against the 2 GB bound). The comparison sample is "
                 "identical across models and comparators for each target "
                 f"({'verified' if results['comparison_sample_identical_across_models'] else 'NOT verified'} "
                 "from the stored part files). The independent "
                 "verification of section 10.3 is a separate stage and "
                 "is not reported here.")
    lines.append("")
    lines.append("Honest limits (section 14, restated): the prior is "
                 "negative on the programme's own evidence, and a "
                 "complete failure of all four configurations is a "
                 "complete and reportable outcome. The return basis is "
                 "a total-return basis: the adjusted close embeds "
                 "dividend reinvestment, so the figures describe the "
                 "return of an investor holding through distributions. "
                 "The 5-day labels overlap by construction. Multi-day "
                 "returns of leveraged and inverse funds are "
                 "path-dependent by construction of the products. A "
                 "material part of some funds' series is carried "
                 "prices (zero returns and zero volume), and the "
                 "comparators inherit the same property, so the "
                 "comparison is symmetric. The pooled figure weights "
                 "long-history funds more heavily, which is why the "
                 "per-fund leg and the per-fund table accompany every "
                 "pooled figure.")
    lines.append("")
    T.robust_write_text(RESULTS_DATED, "\n".join(lines) + "\n")


# --- Main ------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["build", "gate", "run", "finalize", "all"])
    ap.add_argument("--only", default=None, choices=list(T.TARGETS),
                    help="single target for the run phase")
    args = ap.parse_args()
    state = load_state()

    if args.phase in ("build", "all"):
        build_panel(state)
        if args.phase == "build":
            return 0

    if args.phase in ("gate", "all"):
        if not T.PANEL_PATH.exists() or not state.get("panel_built"):
            build_panel(state)
        gate = run_entry_gate(state)
        state["gate"] = {"passed": gate["passed"], "checks": gate["checks"],
                         "input_hashes": gate["input_hashes"],
                         "report_md": str(GATE_MD), "report_json": str(GATE_JSON)}
        save_state(state)
        write_gate_report(gate)
        log(f"gate: overall {'PASS' if gate['passed'] else 'FAIL'}")
        if not gate["passed"]:
            log("gate: FAIL — halting before any fitting (section 12.1)")
            return 1
        if args.phase == "gate":
            return 0

    if args.phase in ("run", "all"):
        if not state.get("gate", {}).get("passed"):
            log("run: entry gate has not passed — stopping")
            return 1
        rc = run_phase(state, args.only)
        if rc != 0:
            return rc
        if args.phase == "run":
            return 0

    if args.phase in ("finalize", "all"):
        return finalize_phase(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
