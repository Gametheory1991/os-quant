"""ML proposal M4 — interaction and nonlinearity audit (diagnostic only).

Frozen plan: reports/ml_m4_interaction_audit_plan_2026-10-27.md.
Tabulates SAVED walk-forward prediction errors by three pre-named states
and block-bootstraps the paired differences (circular block, length 10,
5,000 resamples, seed 13 — the fix-pack convention, reusing
engine/ml_fixpack_stats machinery). Fits NO model, books NO trials,
changes NO scorecard or verdict.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from engine.ml_fixpack_stats import block_bootstrap, percentile_interval  # noqa: E402

ROOT = pathlib.Path(__file__).parent
ML = ROOT / "data" / "ml"
OUT = ROOT / "reports" / "ml_m4_interaction_audit_results.json"

BLOCK, N_BOOT, SEED, MIN_CELL_RESAMPLE = 10, 5000, 13, 5

FILES = {"v1": ML / "wf_predictions.parquet", "v2": ML / "wf_predictions_v2.parquet"}
TARGETS = ["T1a1", "T1a5", "T1b_HY", "T1b_IG"]
LINEAR = {"T1a1": "ridge", "T1a5": "ridge", "T1b_HY": "logistic", "T1b_IG": "logistic"}
REFS = {"T1a1": ["persistence", "climatology"], "T1a5": ["persistence", "climatology"],
        "T1b_HY": ["climatology"], "T1b_IG": ["climatology"]}
SPREAD_Z = {"T1a1": "oas_hy_z252", "T1a5": "oas_hy_z252",
            "T1b_HY": "oas_hy_z252", "T1b_IG": "oas_ig_z252"}
STATE_NAMES = {
    "S1_illiq_x_volregime": ["LowIlliq_Regime0", "LowIlliq_Regime1",
                             "HighIlliq_Regime0", "HighIlliq_Regime1"],
    "S2_illiq_x_auction": ["LowIlliq_NoAuction", "LowIlliq_Auction",
                           "HighIlliq_NoAuction", "HighIlliq_Auction"],
    "S3_spread_x_illiqchg": ["LowSpread_IlliqFalling", "LowSpread_IlliqRising",
                             "HighSpread_IlliqFalling", "HighSpread_IlliqRising"],
}
# (joint-reference cell, joint-elevated cell) per state, per frozen plan §6.3
CONTRAST = {"S1_illiq_x_volregime": (0, 3), "S2_illiq_x_auction": (0, 3),
            "S3_spread_x_illiqchg": (0, 3)}


def row_errors(sub: pd.DataFrame, target: str, model: str) -> np.ndarray:
    g = sub[sub["model"] == model]
    if target.startswith("T1a"):
        return np.abs(g["y_true"].to_numpy(float) - g["y_pred"].to_numpy(float))
    return (g["y_proba"].to_numpy(float) - g["y_true"].to_numpy(float)) ** 2


def boot_cell_stats(idx: np.ndarray, cells: np.ndarray,
                    series: dict[str, np.ndarray]) -> dict:
    """Per-resample cell means for every series, via one-hot aggregation."""
    n_res, n = idx.shape
    onehot = np.eye(4)[cells[idx]]                       # (R, n, 4)
    counts = onehot.sum(axis=1)                          # (R, 4)
    out = {"counts": counts}
    for name, s in series.items():
        sums = np.einsum("rn,rnc->rc", s[idx], onehot)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[name] = np.where(counts >= MIN_CELL_RESAMPLE, sums / np.where(counts == 0, 1, counts), np.nan)
    return out


def interval(vals: np.ndarray) -> dict:
    v = vals[~np.isnan(vals)]
    lo, hi = percentile_interval(v) if len(v) else (float("nan"), float("nan"))
    return {"lo": lo, "hi": hi, "n_valid_resamples": int(len(v)),
            "median": float(np.median(v)) if len(v) else float("nan")}


def main() -> None:
    fs = pd.read_parquet(ML / "feature_store.parquet")
    fs_cols = ["illiq_agg_z252", "illiq_chg5", "vix_regime_high",
               "auction_day", "oas_hy_z252", "oas_ig_z252"]
    results: dict = {"plan": "reports/ml_m4_interaction_audit_plan_2026-10-27.md",
                     "bootstrap": {"block": BLOCK, "n_resamples": N_BOOT, "seed": SEED,
                                   "method": "circular block bootstrap (fix-pack convention)"},
                     "versions": {}}
    for ver, path in FILES.items():
        preds = pd.read_parquet(path)
        results["versions"][ver] = {"file": str(path.relative_to(ROOT)), "targets": {}}
        for target in TARGETS:
            sub = preds[preds["target"] == target].copy()
            models = sorted(sub["model"].unique())
            date_sets = {m: tuple(sub[sub["model"] == m].sort_values("date")["date"])
                         for m in models}
            base_dates = date_sets[LINEAR[target]]
            for m, d in date_sets.items():
                assert d == base_dates, f"{ver}/{target}/{m}: date set differs from linear model"
            sub = sub.sort_values("date")
            dates = pd.to_datetime(base_dates)
            joined = fs.loc[dates, fs_cols]
            assert len(joined) == len(dates) and joined.notna().all().all(), \
                f"{ver}/{target}: state join incomplete or NaN"

            hi_illiq = (joined["illiq_agg_z252"].to_numpy(float) > 0).astype(int)
            regime = (joined["vix_regime_high"].to_numpy(float) == 1).astype(int)
            auction = (joined["auction_day"].to_numpy(float) == 1).astype(int)
            hi_spread = (joined[SPREAD_Z[target]].to_numpy(float) > 0).astype(int)
            rising = (joined["illiq_chg5"].to_numpy(float) > 0).astype(int)
            state_cells = {
                "S1_illiq_x_volregime": hi_illiq * 2 + regime,
                "S2_illiq_x_auction": hi_illiq * 2 + auction,
                "S3_spread_x_illiqchg": hi_spread * 2 + rising,
            }
            lin_name, gbm_name = LINEAR[target], "gbm"
            series = {"linear": row_errors(sub, target, lin_name),
                      "gbm": row_errors(sub, target, gbm_name)}
            for r in REFS[target]:
                series[f"ref_{r}"] = row_errors(sub, target, r)
            series["diff_gbm_minus_linear"] = series["gbm"] - series["linear"]

            n = len(dates)
            idx = block_bootstrap(n, BLOCK, N_BOOT, seed=SEED)
            t_out: dict = {"linear_model": lin_name, "n": int(n),
                           "effective_outcomes": float(n if target == "T1a1" else n / 5),
                           "date_min": str(dates.min().date()), "date_max": str(dates.max().date()),
                           "pooled": {}, "states": {}}
            for name, s in series.items():
                t_out["pooled"][name] = float(np.mean(s))
            if target.startswith("T1a"):
                t_out["pooled"]["gbm_mae_ratio_vs_persistence"] = float(
                    np.mean(series["gbm"]) / np.mean(series["ref_persistence"]))
                t_out["pooled"]["linear_mae_ratio_vs_persistence"] = float(
                    np.mean(series["linear"]) / np.mean(series["ref_persistence"]))
            # pooled GBM-linear interval from the same resamples
            pooled_diff = series["diff_gbm_minus_linear"][idx].mean(axis=1)
            t_out["pooled"]["diff_gbm_minus_linear_interval"] = interval(pooled_diff)

            for state, cells in state_cells.items():
                boot = boot_cell_stats(idx, cells, series)
                cell_rows = []
                for c in range(4):
                    cn = int((cells == c).sum())
                    row = {"cell": STATE_NAMES[state][c], "n": cn,
                           "effective_outcomes": float(cn if target == "T1a1" else cn / 5),
                           "small_cell": bool(cn < 20)}
                    for name in series:
                        row[f"{name}_mean_error"] = float(np.mean(series[name][cells == c])) if cn else None
                        row[f"{name}_interval"] = interval(boot[name][:, c])
                    ref0 = REFS[target][0]
                    row["linear_over_ref"] = float(
                        row["linear_mean_error"] / row[f"ref_{ref0}_mean_error"]) if cn else None
                    cell_rows.append(row)
                ref_c, elev_c = CONTRAST[state]
                contrast_vals = boot["linear"][:, elev_c] - boot["linear"][:, ref_c]
                ref_series = f"ref_{REFS[target][0]}"
                ref_contrast_vals = boot[ref_series][:, elev_c] - boot[ref_series][:, ref_c]
                t_out["states"][state] = {
                    "cells": cell_rows,
                    "ridge_variation_contrast": {
                        "definition": f"linear mean error [{STATE_NAMES[state][elev_c]}] - [{STATE_NAMES[state][ref_c]}]",
                        "value": float(np.mean(series["linear"][cells == elev_c])
                                       - np.mean(series["linear"][cells == ref_c])),
                        "interval": interval(contrast_vals)},
                    "reference_contrast_same_cells": {
                        "reference": REFS[target][0],
                        "value": float(np.mean(series[ref_series][cells == elev_c])
                                       - np.mean(series[ref_series][cells == ref_c])),
                        "interval": interval(ref_contrast_vals)},
                }
            results["versions"][ver]["targets"][target] = t_out
            print(f"done {ver} {target} n={n}")
    OUT.write_text(json.dumps(results, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
