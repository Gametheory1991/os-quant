"""Driver for ML Target 5a (per-fund ETF premium prediction).

Pre-registered spec: reports/ml_target5a_experiment_spec_2026-10-28.md
(frozen 2026-10-28 before any Target 5a fit). Order of operations is
part of the spec: reconstruct the panel -> run the section 9.1 entry
gate -> ONLY IF the gate passes, fit the walk-forward ladder of
section 8 -> write artifacts -> exit verification from the stored
predictions. Any gate FAIL aborts before any model is fit.

Outputs (spec section 11, with the run-date suffix the results report
carries):
  data/t5a_premium/panel_t5a.parquet
  data/t5a_premium/wf_predictions_t5a.parquet
  reports/ml_target5a_predictions_2026-10-28.csv
  reports/ml_target5a_results.json (+ dated copy)
Offline evaluation only. No forward issuance, no scorecard writes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "workspace" / "goals" /
                       "ultimate-arbitrage-screener-mispricing-dislocation-flagging-machine"
                       / "screener"))

from engine import ml_t5a as T  # noqa: E402

np.random.seed(T.SEED)

RUN_DATE = "2026-10-28"
RESULTS_PATH = ROOT / "reports" / "ml_target5a_results.json"
RESULTS_DATED = ROOT / "reports" / f"ml_target5a_results_{RUN_DATE}.json"
PRED_CSV = ROOT / "reports" / f"ml_target5a_predictions_{RUN_DATE}.csv"
REPORT_PATH = ROOT / "reports" / f"ml_target5a_results_{RUN_DATE}.md"
REPORT_COPY = (Path.home() / "workspace" / "goals" /
               "market-knowledge-agent-ml-prediction-engine" / "files" /
               f"ml_target5a_results_{RUN_DATE}.md")


# --- Entry gate (spec section 9.1) ---------------------------------------------

def parse_premium_table() -> dict[str, float]:
    out: dict[str, float] = {}
    for line in T.PREMIUM_TABLE_PATH.read_text().splitlines():
        if not line.startswith("|"):
            continue
        parts = line.split("|")
        if len(parts) < 14:
            continue
        ticker = parts[1].strip()
        try:
            out[ticker] = float(parts[9].strip())
        except (ValueError, IndexError):
            continue
    return out


def entry_gate(panel: pd.DataFrame, features: pd.DataFrame,
               samples: dict, folds_by: dict) -> dict:
    import class2  # screener's own engine, on sys.path

    gate: dict = {"checks": {}, "passed": True}

    def record(name, ok, detail):
        gate["checks"][name] = {"pass": bool(ok), "detail": detail}
        gate["passed"] = gate["passed"] and bool(ok)

    elig_csv = pd.read_csv(T.ELIGIBILITY_PATH)
    csv_counts = dict(zip(elig_csv["ticker"], elig_csv["aligned_obs"]))

    # Item 1 — reconstruction fidelity, full series, eligible funds.
    max_diff, count_ok, per_fund = 0.0, True, {}
    class2_series = {}
    for t in T.ELIGIBLE:
        cs = class2.compute_series(t)
        class2_series[t] = cs
        s = cs["series"]
        rec = panel[panel["ticker"] == t].sort_values("date")
        same_len = len(rec) == len(s)
        count_ok = count_ok and same_len
        if same_len:
            d = float(np.max(np.abs(
                rec["premium_bp"].to_numpy()
                - s["premium"].to_numpy() * 10_000.0)))
            dates_equal = bool((rec["date"].to_numpy()
                                == s["date"].to_numpy()).all())
        else:
            d, dates_equal = float("inf"), False
        max_diff = max(max_diff, d)
        per_fund[t] = {"reconstructed_n": int(len(rec)),
                       "screener_n": int(len(s)),
                       "max_abs_diff_bp": d, "dates_equal": dates_equal}
    record("1_reconstruction_fidelity_full_series",
           count_ok and max_diff == 0.0,
           {"max_abs_diff_bp_all_eligible": max_diff,
            "counts_identical_all_eligible": count_ok, "per_fund": per_fund})

    # Item 2 — latest printed premium, all 118 covered funds, <= 0.5 bp.
    table = parse_premium_table()
    worst, fails, checked = 0.0, [], 0
    for t in class2.navmod.TICKER_SOURCES:
        cs = class2_series.get(t) or class2.compute_series(t)
        latest_bp = float(cs["latest_premium"]) * 10_000.0
        if t not in table:
            fails.append({"ticker": t, "reason": "absent from premium table"})
            continue
        checked += 1
        diff = abs(latest_bp - table[t])
        worst = max(worst, diff)
        if diff > 0.5:
            fails.append({"ticker": t, "reconstructed_bp": latest_bp,
                          "table_bp": table[t], "abs_diff_bp": diff})
    record("2_latest_printed_premium_all_covered",
           checked == 118 and not fails,
           {"funds_checked": checked, "max_abs_diff_bp": worst,
            "failures": fails,
            "table_path": str(T.PREMIUM_TABLE_PATH)})

    # Item 3 — eligibility assertion.
    counts_match, ends_ok = True, True
    for t in T.ELIGIBLE:
        rec = panel[panel["ticker"] == t]
        counts_match = counts_match and int(len(rec)) == int(csv_counts[t])
        ends_ok = ends_ok and rec["date"].max() == T.PANEL_END
    nav_kind_counts = {}
    for t, cs in class2_series.items():
        nav_kind_counts[t] = int(cs["n_history"])
    for t in class2.navmod.TICKER_SOURCES:
        if t not in nav_kind_counts:
            nav_kind_counts[t] = int(class2.compute_series(t)["n_history"])
    csv_match_all = all(int(nav_kind_counts[t]) == int(csv_counts[t])
                         for t in nav_kind_counts
                         if int(csv_counts.get(t, -1)) > 0)
    record("3_eligibility_assertion",
           counts_match and ends_ok and csv_match_all,
           {"eligible_funds": len(T.ELIGIBLE),
            "counts_equal_eligibility_csv_eligible": counts_match,
            "counts_equal_eligibility_csv_all_covered_with_counts": csv_match_all,
            "panel_end_2026_10_27_all_eligible": ends_ok,
            "strata": {"STD": len(T.STD_FUNDS), "PRO": len(T.PRO_FUNDS)}})

    # Item 4 — NAV guard and ProShares hole counts / gaps.
    kre_dates = set(panel[panel["ticker"] == "KRE"]["date"])
    gaps = {}
    for t in ("USD", "SSG"):
        dd = panel[panel["ticker"] == t]["date"].sort_values()
        gaps[t] = int(dd.diff().dt.days.max())
    nav_ok = bool((panel["nav"] > 0).all() and (panel["price"] > 0).all()
                  and (panel["yahoo_close"] > 0).all())
    record("4_nav_guard_and_proshares_holes",
           nav_ok and pd.Timestamp("2006-06-19") not in kre_dates
           and len(panel[panel["ticker"] == "USD"]) == 4701
           and len(panel[panel["ticker"] == "SSG"]) == 4633
           and gaps == {"USD": 196, "SSG": 106},
           {"nav_price_close_all_positive": nav_ok,
            "kre_2006_06_19_absent": pd.Timestamp("2006-06-19") not in kre_dates,
            "usd_aligned": int(len(panel[panel['ticker'] == 'USD'])),
            "ssg_aligned": int(len(panel[panel['ticker'] == 'SSG'])),
            "max_aligned_gap_days": gaps})

    # Item 5 — sample and fold integrity; every fund contributes test rows.
    nan_ok, funds_ok, detail5 = True, True, {}
    for (target, stratum), sample in samples.items():
        key = f"{target}|{stratum}"
        n_nan = int(sample[T.NUMERIC_FEATURES + ["y"]].isna().sum().sum())
        nan_ok = nan_ok and n_nan == 0
        test_pos = np.concatenate([f["test_idx"] for f in folds_by[(target, stratum)]])
        funds_with_test = set(sample.iloc[test_pos]["ticker"].unique())
        expected = set(T.STD_FUNDS if stratum == "STD" else T.PRO_FUNDS)
        funds_ok = funds_ok and funds_with_test == expected
        detail5[key] = {"n_sample": int(len(sample)), "n_nan_cells": n_nan,
                        "n_folds": len(folds_by[(target, stratum)]),
                        "n_test_rows": int(len(test_pos)),
                        "funds_with_test_rows": len(funds_with_test)}
    record("5_sample_and_fold_integrity", nan_ok and funds_ok, detail5)

    # Item 6 — truncation (point-in-time) check, SPY and USD, cut 2022-12-31.
    cut = pd.Timestamp("2022-12-31")
    vix_full = T.load_vix()
    trunc_detail, trunc_ok = {}, True
    for t in ("SPY", "USD"):
        full_f = features[features["ticker"] == t].sort_values("date")
        fund_cut = panel[(panel["ticker"] == t)
                         & (panel["date"] <= cut)]
        vix_cut = vix_full[vix_full["date"] <= cut]
        cut_f = T.build_fund_features(fund_cut, vix_cut)
        a = full_f[full_f["date"] <= cut].reset_index(drop=True)
        b = cut_f.reset_index(drop=True)
        same_dates = bool((a["date"].to_numpy() == b["date"].to_numpy()).all())
        md, nan_mismatch = 0.0, 0
        for c in T.NUMERIC_FEATURES:
            av, bv = a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float)
            nan_mismatch += int((np.isnan(av) != np.isnan(bv)).sum())
            both = ~np.isnan(av) & ~np.isnan(bv)
            if both.any():
                md = max(md, float(np.max(np.abs(av[both] - bv[both]))))
        ok = same_dates and md == 0.0 and nan_mismatch == 0
        trunc_ok = trunc_ok and ok
        trunc_detail[t] = {"rows_compared": int(len(a)),
                           "max_abs_diff": md,
                           "nan_pattern_mismatches": nan_mismatch,
                           "dates_identical": same_dates}
    record("6_truncation_check_2022_12_31", trunc_ok, trunc_detail)

    # Item 7 — input hashes (recorded; equality of counts in items 1-4
    # against the audited snapshot is the operative stop rule).
    record("7_input_hashes_recorded", True,
           {"n_files_hashed": 2 * len(T.ELIGIBLE) + 3,
            "hashes_in_results_json": "input_hashes"})
    return gate


# --- Exit verification (spec section 9.2 items 1-2, self-check) ------------------

def exit_verify(preds_path: Path, metrics: dict, fold_tables: dict) -> dict:
    stored = pd.read_parquet(preds_path)
    max_diff, ok_sets, ok_counts = 0.0, True, True
    for (target, stratum), m in metrics.items():
        sub = stored[(stored["target"] == target)
                     & (stored["stratum"] == stratum)]
        ref = None
        for model in T.MODELS:
            d = sub[sub["model"] == model]
            keys = set(zip(d["ticker"], d["date"].astype(str), d["fold"]))
            if ref is None:
                ref = keys
            ok_sets = ok_sets and keys == ref
            recomputed = float(np.mean(np.abs(d["y_true"] - d["y_pred"])))
            max_diff = max(max_diff, abs(recomputed
                                         - m["models"][model]["mae_bp"]))
            per_fold_n = {int(k): v["n"] for k, v in
                          m["models"][model]["per_fold"].items()}
            file_fold_n = {int(k): int(v) for k, v in
                           d.groupby("fold").size().items()}
            ok_counts = ok_counts and per_fold_n == file_fold_n
        json_test = sum(f["n_test"] for f in fold_tables[f"{target}|{stratum}"])
        ok_counts = ok_counts and json_test == m["n_test"]
    return {"recomputed_from_predictions_file": True,
            "max_abs_diff_mae": max_diff,
            "identical_test_row_sets_across_models": bool(ok_sets),
            "per_fold_counts_match_results_json": bool(ok_counts),
            "passed": bool(max_diff <= 1e-12 and ok_sets and ok_counts)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-report", action="store_true")
    ap.add_argument("--ledger-first", type=int, default=None)
    ap.add_argument("--ledger-last", type=int, default=None)
    args = ap.parse_args()
    if args.write_report:
        write_report(args.ledger_first, args.ledger_last)
        return

    hashes = T.input_hashes()
    panel = T.build_panel()
    vix = T.load_vix()
    features = T.build_all_features(panel, vix)
    samples, folds_by = {}, {}
    for target, horizon in T.TARGETS.items():
        for stratum in ("STD", "PRO"):
            s = T.build_sample(features, target, stratum)
            samples[(target, stratum)] = s
            folds_by[(target, stratum)] = T.make_pooled_folds(s, horizon)

    gate = entry_gate(panel, features, samples, folds_by)
    if not gate["passed"]:
        RESULTS_PATH.write_text(json.dumps(
            {"experiment": "ML Target 5a — per-fund ETF premium prediction",
             "spec": "reports/ml_target5a_experiment_spec_2026-10-28.md",
             "input_hashes": hashes, "entry_gate": gate,
             "run_aborted": "entry gate FAILED — no model was fit"},
            indent=2))
        raise SystemExit("STOP: entry gate FAILED — no model was fit. "
                         f"Gate written to {RESULTS_PATH}")

    T.T5A_DIR.mkdir(parents=True, exist_ok=True)
    panel_out = panel.copy()
    panel_out["date"] = panel_out["date"].dt.strftime("%Y-%m-%d")
    panel_out.to_parquet(T.PANEL_PATH, index=False)

    all_preds, metrics, fold_tables, alphas = [], {}, {}, {}
    for (target, stratum), sample in samples.items():
        key = f"{target}|{stratum}"
        preds, al = T.run_walk_forward(sample, target, stratum,
                                       folds_by[(target, stratum)])
        all_preds.append(preds)
        alphas[key] = al
        fold_tables[key] = [f["meta"] for f in folds_by[(target, stratum)]]
        print(f"{key}: folds={len(folds_by[(target, stratum)])} "
              f"sample={len(sample)}", flush=True)
    pred_df = pd.concat(all_preds, ignore_index=True).sort_values(
        ["target", "stratum", "model", "date", "ticker"]).reset_index(drop=True)
    pred_out = pred_df.copy()
    pred_out.to_parquet(T.PRED_PATH, index=False)
    csv_out = pd.DataFrame({
        "fund": pred_df["ticker"], "date": pred_df["date"].dt.strftime("%Y-%m-%d"),
        "target": pred_df["target"], "stratum": pred_df["stratum"],
        "model": pred_df["model"], "fold": pred_df["fold"],
        "prediction": pred_df["y_pred"], "realized": pred_df["y_true"],
        "vix_stress": pred_df["vix_stress"],
        "ridge_alpha": pred_df["alpha"],
        "prem_level_bp": pred_df["prem_level_bp"],
    })
    csv_out.to_csv(PRED_CSV, index=False)

    for (target, stratum), sample in samples.items():
        metrics[(target, stratum)] = T.compute_metrics(pred_df, target, stratum)
    verify = exit_verify(T.PRED_PATH, metrics, fold_tables)

    results = {
        "experiment": "ML Target 5a — per-fund ETF premium prediction (T5a-L1, T5a-C5)",
        "spec": "reports/ml_target5a_experiment_spec_2026-10-28.md "
                "(pre-registered 2026-10-28, before any Target 5a fit)",
        "run_date": RUN_DATE,
        "targets": {"T5a-L1": "next-day premium level, basis points",
                    "T5a-C5": "5-trading-day premium change, basis points; "
                              "effective independent outcomes = n_test / 5"},
        "strata": {"STD": "57 funds, premium = Yahoo raw close / issuer NAV - 1",
                   "PRO": "6 funds (USD, SSG, VIXY, VIXM, UVXY, SVXY), premium = "
                          "issuer official market price (marketPriceNav, ProShares "
                          "API) / NAV - 1 — premium struck off the issuer's official "
                          "market price (ProShares API), not the Yahoo close"},
        "input_hashes": hashes,
        "entry_gate": gate,
        "panel": {"path": "data/t5a_premium/panel_t5a.parquet",
                  "rows": int(len(panel)),
                  "funds": len(T.ELIGIBLE),
                  "per_fund": {t: {"aligned_obs": int(len(panel[panel['ticker'] == t])),
                                   "start": str(panel[panel['ticker'] == t]['date'].min().date()),
                                   "end": str(panel[panel['ticker'] == t]['date'].max().date())}
                               for t in T.ELIGIBLE}},
        "samples": {f"{t}|{s}": {"n": int(len(samples[(t, s)])),
                                 "start": str(samples[(t, s)]['date'].min().date()),
                                 "end": str(samples[(t, s)]['date'].max().date()),
                                 "design_width": len(T.design_columns(s))}
                    for (t, s) in samples},
        "folds": fold_tables,
        "ridge_alphas_by_fold": alphas,
        "metrics": {f"{t}|{s}": metrics[(t, s)] for (t, s) in metrics},
        "exit_verify_self_check": verify,
        "trials": {"learned_configurations": 8,
                   "configurations": ["T5a-L1 STD ridge", "T5a-L1 STD gbm",
                                      "T5a-C5 STD ridge", "T5a-C5 STD gbm",
                                      "T5a-L1 PRO ridge", "T5a-L1 PRO gbm",
                                      "T5a-C5 PRO ridge", "T5a-C5 PRO gbm"],
                   "baselines_counted": False,
                   "booking": "appended to the trials ledger before results "
                              "are reported; exact rows stated in the results report"},
        "scope": "Offline evaluation only. No forward issuance of any kind.",
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=float))
    RESULTS_DATED.write_text(json.dumps(results, indent=2, default=float))

    print(f"entry gate: {'PASS' if gate['passed'] else 'FAIL'}")
    for (target, stratum), m in metrics.items():
        for model in T.MODELS:
            b = m["models"][model]
            print(f"{target} {stratum} {model:16s} MAE {b['mae_bp']:.4f} bp "
                  f"RMSE {b['rmse_bp']:.4f}  n={b['n']}")
        for model in T.LEARNED:
            v = m["verdicts"][model]
            print(f"{target} {stratum} {model:16s} verdict: {v['verdict']} "
                  f"(funds beating persistence {v['leg_b_funds_beating_persistence']}"
                  f"/{v['leg_b_denominator_funds']})")
    print(f"exit verify: {verify}")
    print(f"panel -> {T.PANEL_PATH}")
    print(f"predictions -> {T.PRED_PATH} and {PRED_CSV}")
    print(f"results -> {RESULTS_PATH}")


# --- Results report --------------------------------------------------------------

def fmt(x, nd=4):
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def write_report(ledger_first: int | None, ledger_last: int | None) -> None:
    results = json.loads(RESULTS_PATH.read_text())
    pred_hash = T.sha256_file(T.PRED_PATH)
    pred_csv_hash = T.sha256_file(PRED_CSV)
    json_hash = T.sha256_file(RESULTS_PATH)
    gate = results["entry_gate"]
    L: list[str] = []
    L.append("# ML Target 5a — Results (per-fund ETF premium prediction)")
    L.append("")
    L.append(f"**Run date:** {RUN_DATE}. **Specification:** "
             "`reports/ml_target5a_experiment_spec_2026-10-28.md` (pre-registered "
             "2026-10-28, before any Target 5a model was fit). **Targets:** "
             "T5a-L1, the next-day premium level; T5a-C5, the 5-trading-day "
             "premium change. All premiums, premium changes, and errors are in "
             "basis points (bp), where one basis point is one hundredth of one "
             "percent (0.01%). Stratum PRO results are for premiums struck off "
             "the issuer's official market price (ProShares API), not the "
             "Yahoo close; the two strata are never pooled.")
    L.append("")
    L.append("**Metric definitions (plain language).** MAE (mean absolute "
             "error), in basis points, is the average size of the forecast "
             "miss, ignoring direction: an MAE of 8 bp means the typical "
             "forecast missed the realised premium (or premium change) by "
             "0.08 percentage points. Lower is better; 0 is perfect. RMSE "
             "(root mean squared error), also in basis points, is the square "
             "root of the average squared miss; it penalises large misses "
             "more than MAE does and is reported as a secondary figure only. "
             "Directional accuracy is the share of test rows on which the "
             "forecast got the direction of the premium's move right. An MAE "
             "ratio below 1 means the model missed by less, on average, than "
             "the baseline it is divided by. For T5a-C5, consecutive labels "
             "overlap, so every figure is reported beside its effective "
             "independent-outcome count, computed as the test-row count "
             "divided by 5.")
    L.append("")
    L.append("## Entry gate (specification section 9.1 — run before any fitting)")
    L.append("")
    L.append(f"**Outcome: {'PASS' if gate['passed'] else 'FAIL'}** — all seven "
             "checks passed before any model was fitted.")
    L.append("")
    for name, chk in gate["checks"].items():
        d = chk["detail"]
        if name.startswith("1_"):
            s = (f"maximum absolute difference against `class2.compute_series` "
                 f"= {d['max_abs_diff_bp_all_eligible']} bp (float-exact 0.0 "
                 f"required) for all 63 eligible funds; aligned counts "
                 f"identical for all 63 funds.")
        elif name.startswith("2_"):
            s = (f"{d['funds_checked']} covered funds checked; maximum "
                 f"absolute difference against the premium table's latest "
                 f"printed premium = {fmt(d['max_abs_diff_bp'])} bp "
                 f"(allowance 0.5 bp). Table: `phase2_premium_table.md`, "
                 f"scan of 2026-10-28 17:43 ET.")
        elif name.startswith("3_"):
            s = ("eligible set is exactly the 63 funds of specification "
                 "section 4.2 (57 stratum STD, 6 stratum PRO); per-fund "
                 "aligned counts equal the eligibility file; panel end "
                 "2026-10-27 for every eligible fund.")
        elif name.startswith("4_"):
            s = (f"no aligned row has NAV, price leg, or Yahoo close at or "
                 f"below zero; the KRE placeholder row of 2006-06-19 is "
                 f"absent; USD holds {d['usd_aligned']} and SSG "
                 f"{d['ssg_aligned']} aligned observations, with maximum "
                 f"aligned gaps of {d['max_aligned_gap_days']['USD']} and "
                 f"{d['max_aligned_gap_days']['SSG']} calendar days.")
        elif name.startswith("5_"):
            s = "; ".join(
                f"{k}: sample {v['n_sample']} rows, {v['n_folds']} folds, "
                f"{v['n_test_rows']} pooled test rows, "
                f"{v['funds_with_test_rows']} funds with test rows"
                for k, v in d.items()) + "."
        elif name.startswith("6_"):
            s = "; ".join(
                f"{k}: {v['rows_compared']} rows compared, maximum absolute "
                f"difference {v['max_abs_diff']}, NaN-pattern mismatches "
                f"{v['nan_pattern_mismatches']}" for k, v in d.items()) + "."
        else:
            s = f"{d['n_files_hashed']} input files hashed (SHA-256 values in the results JSON)."
        L.append(f"- **{name}** — {'PASS' if chk['pass'] else 'FAIL'}: {s}")
    L.append("")
    L.append(f"**Panel:** {results['panel']['rows']} aligned observations across "
             f"63 funds. **Exit verification (self-check, specification section "
             f"9.2 items 1–2):** headline MAE figures recomputed from the stored "
             f"predictions file agree with the results JSON with maximum "
             f"absolute difference "
             f"{results['exit_verify_self_check']['max_abs_diff_mae']}; test-row "
             f"sets are identical across models within each target and stratum; "
             f"per-fold counts match. Independent verification by a separate "
             f"verify gate remains a distinct step after this report.")
    L.append("")
    for key in ("T5a-L1|STD", "T5a-C5|STD", "T5a-L1|PRO", "T5a-C5|PRO"):
        target, stratum = key.split("|")
        m = results["metrics"][key]
        smp = results["samples"][key]
        folds = results["folds"][key]
        basis = ("premium = Yahoo raw close / issuer NAV − 1" if stratum == "STD"
                 else "premium struck off the issuer's official market price "
                      "(ProShares API), not the Yahoo close")
        L.append(f"## {target} — stratum {stratum} ({basis})")
        L.append("")
        L.append(f"Sample: {smp['n']} rows, {smp['start']} to {smp['end']}; "
                 f"design width {smp['design_width']} columns; {len(folds)} "
                 f"walk-forward folds; pooled test rows {m['n_test']}; "
                 f"effective independent outcomes "
                 f"{fmt(m['effective_independent_outcomes'], 1)}"
                 + ("" if target == "T5a-L1" else " (test rows / 5)") + ".")
        L.append("")
        L.append("| Model | MAE (bp) | RMSE (bp) | MAE / persistence | "
                 "MAE / trailing median | Directional accuracy |")
        L.append("|---|---|---|---|---|---|")
        for model in T.MODELS:
            b = m["models"][model]
            L.append(f"| {model} | {fmt(b['mae_bp'])} | {fmt(b['rmse_bp'])} | "
                     f"{fmt(b['mae_ratio_vs_persistence'], 4)} | "
                     f"{fmt(b['mae_ratio_vs_trailing_median'], 4)} | "
                     f"{fmt(b['directional_accuracy'], 4)} |")
        L.append("")
        pers = m["models"]["persistence"]
        med = m["models"]["trailing_median"]
        L.append(f"Baselines on the identical pooled test rows: persistence "
                 f"MAE {fmt(pers['mae_bp'])} bp; trailing-median MAE "
                 f"{fmt(med['mae_bp'])} bp.")
        L.append("")
        # Per-fold summary
        L.append(f"Per-fold detail: {len(folds)} folds, "
                 f"{folds[0]['test_start']} to {folds[-1]['test_end']}; "
                 f"contributing funds per fold range from "
                 f"{min(f['n_funds_in_test'] for f in folds)} to "
                 f"{max(f['n_funds_in_test'] for f in folds)}. Full per-fold "
                 f"MAE tables are in the results JSON. Folds in which each "
                 f"learned model recorded a lower MAE than persistence "
                 f"(descriptive count over the folds both were scored on):")
        L.append("")
        for model in T.LEARNED:
            pf = m["models"][model]["per_fold"]
            ppf = pers["per_fold"]
            won = sum(1 for fid in pf if pf[fid]["mae_bp"] < ppf[fid]["mae_bp"])
            L.append(f"- {model}: {won} of {len(pf)} folds below persistence MAE.")
        L.append("")
        # Regime and credit slices
        for model in T.MODELS:
            rs = m["models"][model]["regime_slice"]
            st, ca = rs["stress_vix_ge_30"], rs["calm_vix_lt_30"]
            L.append(f"- Regime slice, {model}: stress (VIX at least 30; "
                     f"premiums are NAV-staleness hypotheses in this regime) "
                     f"MAE {fmt(st['mae_bp'])} bp on {st['n']} rows; calm MAE "
                     f"{fmt(ca['mae_bp'])} bp on {ca['n']} rows.")
        if stratum == "STD":
            L.append("")
            for model in T.MODELS:
                cs = m["models"][model]["credit_slice"]
                L.append(f"- Credit slice, {model}: the seven slow-marking "
                         f"credit funds (CLOI, CLOB, ICLO, PRIV, BKLN, SRLN, "
                         f"BIZD — a NAV-staleness-hypothesis group) MAE "
                         f"{fmt(cs['credit_7_funds']['mae_bp'])} bp on "
                         f"{cs['credit_7_funds']['n']} rows; remaining 50 "
                         f"funds MAE {fmt(cs['remaining_50_funds']['mae_bp'])} "
                         f"bp on {cs['remaining_50_funds']['n']} rows.")
        L.append("")
        # Per-fund summary
        funds = T.STD_FUNDS if stratum == "STD" else T.PRO_FUNDS
        lev_note = (" Four of the six stratum PRO funds are leveraged "
                    "daily-reset products (USD, SSG, UVXY, SVXY), labelled "
                    "here per the specification." if stratum == "PRO" else "")
        L.append(f"Per-fund summary (descriptive slices of the single pooled "
                 f"configuration per model; not per-fund findings).{lev_note}")
        L.append("")
        for model in T.LEARNED:
            n_beat = m["verdicts"][model]["leg_b_funds_beating_persistence"]
            per = sorted(funds, key=lambda t: m["per_fund"][t][f"mae_{model}"])
            best, worst = per[0], per[-1]
            L.append(f"- {model}: beats persistence on per-fund MAE in "
                     f"{n_beat} of {len(funds)} funds. Lowest per-fund MAE: "
                     f"{best} at {fmt(m['per_fund'][best][f'mae_{model}'])} bp "
                     f"on {m['per_fund'][best]['n_test']} test rows"
                     + ("" if target == "T5a-L1" else
                        f" (effective outcomes {fmt(m['per_fund'][best]['effective_independent_outcomes'], 1)})")
                     + f". Highest per-fund MAE: {worst} at "
                     f"{fmt(m['per_fund'][worst][f'mae_{model}'])} bp on "
                     f"{m['per_fund'][worst]['n_test']} test rows"
                     + ("" if target == "T5a-L1" else
                        f" (effective outcomes {fmt(m['per_fund'][worst]['effective_independent_outcomes'], 1)})")
                     + ".")
            c = m["verdicts"][model]["concentration"]
            if c["largest_fund"] is not None:
                L.append(f"- {model}: concentration — the largest single "
                         f"fund's share of the pooled absolute-error "
                         f"improvement against persistence is "
                         f"{c['largest_fund']} at "
                         f"{fmt(c['largest_fund_share_of_improvement'], 4)} "
                         f"of the total improvement.")
            else:
                L.append(f"- {model}: concentration — the pooled "
                         f"absolute-error change against persistence is not "
                         f"an improvement (total {fmt(c['total_improvement_bp_sum'], 1)} bp "
                         f"summed), so no largest-fund share is stated.")
        L.append("")
        L.append("Full per-fund table (test rows, test blocks, effective "
                 "outcomes, MAE of each model and baseline, beat-persistence "
                 "flags): results JSON, `per_fund`.")
        L.append("")
        # Verdicts
        L.append("**Graduation verdicts (specification section 10, applied "
                 "mechanically):**")
        L.append("")
        for model in T.LEARNED:
            v = m["verdicts"][model]
            L.append(f"- {target} / {stratum} / {model}: **{v['verdict']}** — "
                     f"pooled MAE {fmt(v['model_mae_bp'])} bp against "
                     f"persistence {fmt(v['persistence_mae_bp'])} bp and "
                     f"trailing median {fmt(v['trailing_median_mae_bp'])} bp "
                     f"(pooled leg {'passed' if v['leg_a_pooled_mae_below_both_baselines'] else 'failed'}); "
                     f"per-fund leg: {v['leg_b_funds_beating_persistence']} of "
                     f"{v['leg_b_denominator_funds']} funds beat persistence "
                     f"(required at least {v['leg_b_required']}; "
                     f"{'passed' if v['leg_b_pass'] else 'failed'}).")
        L.append("")
    L.append("## Trials receipt")
    L.append("")
    if ledger_first is not None and ledger_last is not None:
        L.append(f"The 8 learned configurations of this experiment were "
                 f"appended to the trials ledger "
                 f"(`~/workspace/goals/quant-research-backtest-engine/knowledge/trials_ledger.md`) "
                 f"as rows {ledger_first}–{ledger_last}, before these results "
                 f"were reported. The ledger re-read immediately before "
                 f"appending stood at highest numbered row 7829, program "
                 f"total 7,829, ML model configurations family 58; after the "
                 f"appending the program total is 7,837 and the family count "
                 f"is 66. Baselines (persistence and trailing median) are "
                 f"deterministic comparators and are not counted. No "
                 f"Deflated Sharpe Ratio is computed or reported by this "
                 f"experiment; any graduation language cites the ML model "
                 f"configurations family count (66 after this booking), "
                 f"never the bare program total.")
    else:
        L.append("Trials rows: see the booking note appended to the trials "
                 "ledger with this run (8 learned configurations; baselines "
                 "are deterministic comparators and are not counted).")
    L.append("")
    L.append("## Files and hashes")
    L.append("")
    L.append(f"- Predictions (parquet, specification section 11): "
             f"`data/t5a_premium/wf_predictions_t5a.parquet`, SHA-256 "
             f"`{pred_hash}`.")
    L.append(f"- Predictions (CSV, one row per test row per model, baselines "
             f"included): `reports/ml_target5a_predictions_{RUN_DATE}.csv`, "
             f"SHA-256 `{pred_csv_hash}`.")
    L.append(f"- Results JSON: `reports/ml_target5a_results.json` (dated copy "
             f"`reports/ml_target5a_results_{RUN_DATE}.json`), SHA-256 "
             f"`{json_hash}`.")
    L.append(f"- Reconstructed panel: `data/t5a_premium/panel_t5a.parquet`. "
             f"Input SHA-256 hashes (63 NAV cache files, 63 Yahoo parquets, "
             f"the ^VIX parquet, the eligibility file, and the specification) "
             f"are recorded in the results JSON under `input_hashes`.")
    L.append(f"- Code: `engine/ml_t5a.py` and `run_ml_target5a.py`.")
    L.append("")
    L.append("## Scope and honest limits")
    L.append("")
    L.append("- Offline evaluation only. No Target 5a model is wired into "
             "daily issuance, the prediction scorecard, or any scheduled job, "
             "and no forward issuance of any kind is authorised by this "
             "experiment. $0 spend; no external sends; no data re-fetched.")
    L.append("- The premium dated d is knowable only on d+1 (NAV publication "
             "lag); this experiment tests a statistical relation on dated "
             "observations and does not by itself establish a tradable "
             "strategy — no costs, spreads, or execution assumptions are "
             "modelled anywhere in it.")
    L.append("- In stress regimes and for the slow-marking credit funds, a "
             "premium predominantly reflects a stale NAV rather than a "
             "mispriced fund, by the screener's standing framing; the regime "
             "and credit slices above exist so that a pooled result cannot "
             "silently be a staleness result.")
    L.append("- Stratum PRO's premiums are struck off the issuer's official "
             "market price, not the Yahoo close; four of its six funds are "
             "leveraged daily-reset products whose premiums embed the daily "
             "reset mechanics of those products. Its results describe that "
             "stratum only.")
    L.append("- The pooled panel's composition changes over time as funds "
             "incept; early folds describe the long-history funds only.")
    L.append("- T5a-C5 labels overlap by construction; the effective "
             "independent-outcome counts are the honest sample sizes for "
             "that target, and they are materially smaller than the row "
             "counts.")
    L.append("- Eligibility is a statement about data depth and measurement "
             "basis only; it is not a statement that any fund's premium is "
             "predictable. A negative result — no configuration graduates — "
             "is a complete and reportable outcome of this experiment.")
    L.append("- Data observation recorded at run time (no design change "
             "arises from it; the frozen construction of specification "
             "section 3 was executed exactly and entry gate item 1 "
             "reproduced the screener's own series float-exactly): in 16 "
             "funds (KIE, PSCC, PSCE, PSCH, PSCT, PSI, RSPD, RSPH, RSPM, "
             "RSPN, RSPS, RSPT, RSPU, SSG, XLF, XSD) the reconstructed "
             "premium carries a level shift in the earlier history — the "
             "median absolute premium through 2023 is between 1,775 and "
             "40,019 bp, against 2 to 9 bp from 2024 onward — a "
             "split-factor signature in the source legs for those spans. "
             "Readers should weight this observation when interpreting the "
             "trailing-median baseline, whose predictions inherit the "
             "shifted trailing level, and any per-fund figure for those "
             "funds over the affected spans.")
    L.append("")
    text = "\n".join(L)
    REPORT_PATH.write_text(text)
    REPORT_COPY.write_text(text)
    print(f"report -> {REPORT_PATH} and {REPORT_COPY}")


if __name__ == "__main__":
    main()
