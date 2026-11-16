"""ML Target 4a secondary panels (Stage C-2) — CL / ZC / ZS 5-day returns.

Pre-registered spec: reports/ml_target4a_secondary_spec_2026-10-27.md
(frozen 2026-10-27 before any fitting). Implements ONLY what that spec
freezes: one feature set per panel, the flagship's walk-forward folds
(engine/ml_harness.py, reused unmodified), the flagship's model ladder
(engine/ml_models.py, reused unmodified), the flagship's climatology
rule (engine/ml_t4a.py, reused unmodified), the integrity gate, and
the frozen metrics / graduation bar. Panels are never pooled.

Offline evaluation only — nothing here is issued or scheduled, and
nothing touches the flagship's forward operation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine import ml_harness as H
from engine import ml_models as M
from engine.ml_t4a import climatology_predict

ROOT = Path(__file__).resolve().parent.parent
T4A_DIR = ROOT / "data" / "t4a_weather_futures"
SELF_CHECKS_PATH = T4A_DIR / "secondary_self_checks.json"

TARGET = "label_ret5"
HORIZON = 5

OWN_PRICE = ["ret5", "ret21", "price_z252", "doy_sin", "doy_cos"]
FEATURES = {
    "cl": OWN_PRICE + ["hdd", "cdd", "hdd_7d", "cdd_7d", "hdd_30d",
                       "cdd_30d", "dd_anom7", "wti_spot", "wti_chg5"],
    "zc": OWN_PRICE + ["cb_hdd", "cb_cdd", "cb_hdd_7d", "cb_cdd_7d",
                       "cb_hdd_30d", "cb_cdd_30d", "cb_dd_anom7",
                       "cb_dsci", "cb_dsci_chg4w", "cb_d2p", "us_d2p"],
    "zs": OWN_PRICE + ["cb_hdd", "cb_cdd", "cb_hdd_7d", "cb_cdd_7d",
                       "cb_hdd_30d", "cb_cdd_30d", "cb_dd_anom7",
                       "cb_dsci", "cb_dsci_chg4w", "cb_d2p", "us_d2p"],
}
CLOSE_COL = {"cl": "cl_close", "zc": "zc_close", "zs": "zs_close"}
PANEL_PATH = {p: T4A_DIR / f"panel_t4a_{p}.parquet" for p in FEATURES}
MODELS = ["climatology", "zero_change", "persistence", "ridge", "gbm"]
LEARNED = ["ridge", "gbm"]


def season_for(panel: str, month: int) -> str:
    """Spec §7: CL keeps the flagship slices; ag panels use the
    growing calendar."""
    if panel == "cl":
        if month in (11, 12, 1, 2, 3):
            return "heating"
        if month in (4, 5, 9, 10):
            return "shoulder"
        return "cooling"
    if month in (3, 4, 5):
        return "planting"
    if month in (6, 7, 8):
        return "growing"
    if month in (9, 10, 11):
        return "harvest"
    return "off_season"


def seasons_of(panel: str) -> tuple[str, ...]:
    return (("heating", "shoulder", "cooling") if panel == "cl"
            else ("planting", "growing", "harvest", "off_season"))


def load_panel(panel: str) -> pd.DataFrame:
    df = pd.read_parquet(PANEL_PATH[panel])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    assert df["date"].is_monotonic_increasing and df["date"].is_unique
    return df


def build_sample(panel: str, df: pd.DataFrame) -> pd.DataFrame:
    feats = FEATURES[panel]
    keep = df[feats + [TARGET]].notna().all(axis=1)
    s = df.loc[keep].copy()
    out = pd.DataFrame(index=pd.DatetimeIndex(s["date"], name="date"))
    for c in feats + [TARGET, "ret5", "roll_window"]:
        out[c] = s[c].to_numpy()
    out["full_pos"] = np.where(keep.to_numpy())[0]
    out["season"] = [season_for(panel, d.month) for d in out.index]
    out["doy"] = out.index.dayofyear.to_numpy()
    out = out.reset_index().rename(columns={"index": "date"})
    out["sample_pos"] = np.arange(len(out))
    return out.set_index("date")


# --- Integrity gate (spec §10; runs BEFORE any fitting) -----------------
def run_integrity_gate(panel: str, df: pd.DataFrame,
                       sample: pd.DataFrame, folds: list) -> dict:
    gate: dict = {"checks": {}, "passed": True}

    def record(name, ok, detail):
        gate["checks"][name] = {"pass": bool(ok), "detail": detail}
        gate["passed"] = gate["passed"] and bool(ok)

    feats = FEATURES[panel]
    n_nan = int(sample[feats + [TARGET]].isna().sum().sum())
    record("sample_no_nans", n_nan == 0,
           {"n_rows": int(len(sample)), "n_nan_cells": n_nan,
            "start": str(sample.index[0].date()),
            "end": str(sample.index[-1].date())})

    fold_detail, ok_folds = [], True
    for f in folds:
        tr = sample.index[f.train_idx]
        te = sample.index[f.test_idx]
        gap = int(sample["full_pos"].to_numpy()[f.test_idx[0]]
                  - sample["full_pos"].to_numpy()[f.train_idx].max())
        ok = bool(tr.max() < te.min() and gap > H.EMBARGO_DAYS)
        ok_folds = ok_folds and ok
        fold_detail.append({"fold": int(f.fold_id),
                            "n_train": int(len(f.train_idx)),
                            "n_test": int(len(f.test_idx)),
                            "n_purged": int(f.test_idx[0] - len(f.train_idx)),
                            "train_end": str(tr.max().date()),
                            "test_start": str(te.min().date()),
                            "gap_trading_days": gap})
    record("fold_integrity", ok_folds, {"n_folds": len(folds),
                                        "folds_head": fold_detail[:3],
                                        "folds_tail": fold_detail[-1:]})

    closes = df[CLOSE_COL[panel]].to_numpy(dtype=float)
    pos_by_date = {d: i for i, d in enumerate(pd.to_datetime(df["date"]))}
    lab = df.set_index("date")[TARGET]
    picks = [sample.index[0], sample.index[len(sample) // 2], sample.index[-1]]
    recs, max_diff = [], 0.0
    for d in picks:
        p = pos_by_date[d]
        recomputed = float(np.log(closes[p + 5] / closes[p]))
        stored = float(lab.loc[d])
        max_diff = max(max_diff, abs(recomputed - stored))
        recs.append({"date": str(d.date()), "stored": stored,
                     "recomputed": recomputed,
                     "abs_diff": abs(recomputed - stored)})
    record("label_hand_recomputation", max_diff == 0.0,
           {"dates": recs, "max_abs_diff": max_diff})

    sc = json.loads(SELF_CHECKS_PATH.read_text())
    tr = sc[f"truncation_{panel}"]
    t_max = max(v.get("max_abs_diff", 0.0) for v in tr.values())
    t_nan = sum(v.get("nan_pattern_mismatches", 0) for v in tr.values())
    t_roll = tr["roll_window"]["mismatches"]
    record("truncation_check", t_max == 0.0 and t_nan == 0 and t_roll == 0,
           {"max_abs_diff_all_features": t_max,
            "nan_pattern_mismatches": int(t_nan),
            "roll_window_mismatches": int(t_roll)})

    tk = {"cl": "CL=F", "zc": "ZC=F", "zs": "ZS=F"}[panel]
    pi = sc[f"prices_{tk}"]
    if panel == "cl":
        ok_price = (pi["nonpositive_closes"] == 1
                    and pi["nonpositive_dates"] == ["2020-04-20"]
                    and pi["max_abs_adj_minus_close"] == 0.0
                    and pi["weekend_rows"] == 0
                    and pi["duplicate_dates"] == 0)
    else:
        ok_price = (pi["nonpositive_closes"] == 0
                    and pi["max_abs_adj_minus_close"] == 0.0
                    and pi["weekend_rows"] == 0
                    and pi["duplicate_dates"] == 0)
    record("price_integrity", ok_price, pi)

    if panel in ("zc", "zs"):
        chk = sc["usdm_dsci_crosscheck"]
        mx = max(r["abs_diff"] for r in chk)
        record("drought_dsci_crosscheck", mx <= 0.5,
               {"checks": chk, "max_abs_diff": mx, "tolerance": 0.5})
    return gate


# --- Walk-forward run -----------------------------------------------------
def run_walk_forward(sample: pd.DataFrame, features: list[str], folds: list):
    X = sample[features]
    y_all = sample[TARGET].to_numpy(dtype=float)
    rows, fold_meta, alphas = [], [], {}
    for f in folds:
        Xtr, Xte = X.iloc[f.train_idx], X.iloc[f.test_idx]
        ytr, yte = y_all[f.train_idx], y_all[f.test_idx]
        test_rows = sample.iloc[f.test_idx]
        preds = {
            "climatology": climatology_predict(sample, f.test_idx),
            "zero_change": np.zeros(len(f.test_idx)),
            "persistence": test_rows["ret5"].to_numpy(dtype=float),
        }
        preds["ridge"], hp = M.ridge_fit_predict(Xtr, ytr, Xte)
        alphas[f"fold{f.fold_id}"] = hp["alpha"]
        preds["gbm"], _ = M.gbm_reg_fit_predict(Xtr, ytr, Xte)
        for model in MODELS:
            p = preds[model]
            for i, pos in enumerate(f.test_idx):
                r = test_rows.iloc[i]
                rows.append(dict(
                    date=sample.index[pos], model=model,
                    y_true=float(yte[i]), y_pred=float(p[i]),
                    fold=int(f.fold_id),
                    roll_window=bool(r["roll_window"]),
                    season=str(r["season"])))
        fold_meta.append(dict(fold=int(f.fold_id), **f.meta,
                              n_purged=int(f.test_idx[0] - len(f.train_idx))))
    return pd.DataFrame(rows), fold_meta, alphas


# --- Metrics (spec §7; MAE/RMSE in %) --------------------------------------
def _block(df: pd.DataFrame) -> dict:
    yt, yp = df["y_true"].to_numpy() * 100.0, df["y_pred"].to_numpy() * 100.0
    return {"n": int(len(df)), "mae_pct": float(np.mean(np.abs(yt - yp))),
            "rmse_pct": float(np.sqrt(np.mean((yt - yp) ** 2)))}


def model_metrics(preds_panel: pd.DataFrame, panel: str) -> dict:
    clim = preds_panel[preds_panel.model == "climatology"]
    pers = preds_panel[preds_panel.model == "persistence"]
    zero = preds_panel[preds_panel.model == "zero_change"]
    clim_mae = _block(clim)["mae_pct"]
    pers_mae = _block(pers)["mae_pct"]
    zero_mae = _block(zero)["mae_pct"]
    zero_nonroll = _block(zero[~zero.roll_window])["mae_pct"]
    out = {}
    for model in MODELS:
        d = preds_panel[preds_panel.model == model].sort_values("date")
        b = _block(d)
        entry = {**b,
                 "mae_ratio_vs_climatology": b["mae_pct"] / clim_mae,
                 "mae_ratio_vs_persistence": b["mae_pct"] / pers_mae,
                 "mae_vs_zero_change_pooled": b["mae_pct"] - zero_mae}
        if model == "zero_change":
            entry["directional_accuracy"] = None
            entry["realized_positive_share_of_test_labels"] = float(
                np.mean(d["y_true"].to_numpy() > 0))
        else:
            entry["directional_accuracy"] = float(np.mean(
                np.sign(d["y_pred"].to_numpy())
                == np.sign(d["y_true"].to_numpy())))
        entry["slices"] = {
            "roll": _block(d[d.roll_window]),
            "non_roll": _block(d[~d.roll_window]),
            "seasons": {s: _block(d[d.season == s])
                        for s in seasons_of(panel)},
        }
        entry["slices"]["non_roll"]["mae_vs_zero_change"] = (
            entry["slices"]["non_roll"]["mae_pct"] - zero_nonroll)
        per_fold = {str(fid): _block(g) for fid, g in d.groupby("fold")}
        entry["per_fold"] = per_fold
        if model in LEARNED:
            wins = sum(1 for fid, g in d.groupby("fold")
                       if _block(g)["mae_pct"]
                       < _block(clim[clim.fold == fid])["mae_pct"])
            entry["folds_won_vs_climatology"] = int(wins)
            entry["n_folds"] = int(d["fold"].nunique())
        out[model] = entry
    return out


def graduation_verdicts(metrics: dict, panel: str) -> dict:
    clim = metrics["climatology"]
    out = {}
    for model in LEARNED:
        mm = metrics[model]
        leg_a = bool(mm["mae_pct"] < clim["mae_pct"])
        leg_b = bool(mm["slices"]["non_roll"]["mae_pct"]
                     < clim["slices"]["non_roll"]["mae_pct"])
        lost = [s for s in seasons_of(panel)
                if mm["slices"]["seasons"][s]["mae_pct"]
                > clim["slices"]["seasons"][s]["mae_pct"]]
        if leg_a and leg_b and len(lost) < 2:
            verdict = "GRADUATES"
        elif leg_a and leg_b:
            verdict = "SEASON-SCOPED (not graduation)"
        else:
            verdict = "FAIL (does not graduate)"
        out[model] = {"verdict": verdict,
                      "leg_a_pooled_mae_beats_climatology": leg_a,
                      "leg_b_non_roll_mae_beats_climatology": leg_b,
                      "season_slices_lost_to_climatology": lost,
                      "n_season_slices_lost": len(lost)}
    return out
