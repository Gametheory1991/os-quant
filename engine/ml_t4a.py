"""ML Target 4a — weather -> natural gas futures (T4a-NG-5d).

Pre-registered spec: reports/ml_target4a_experiment_spec.md (written
2026-10-24 before any fitting). This module implements ONLY what that
spec freezes: the FULL (22) / REDUCED (18) feature lists, the identical
2,952-row sample for both variants (coordinator decision D1), expanding
walk-forward folds from engine/ml_harness.py (reused unmodified), the
model ladder from engine/ml_models.py (reused unmodified), the
integrity gate, and the frozen metrics / graduation bar.

Offline evaluation only — nothing here is issued or scheduled.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine import ml_harness as H
from engine import ml_models as M

ROOT = Path(__file__).resolve().parent.parent
T4A_DIR = ROOT / "data" / "t4a_weather_futures"
PANEL_PATH = T4A_DIR / "panel_t4a.parquet"
SELF_CHECKS_PATH = T4A_DIR / "self_checks.json"

TARGET = "label_ret5"
HORIZON = 5
EXPECTED_SAMPLE_N = 2952

FULL_FEATURES = [
    "ret5", "ret21", "price_z252", "doy_sin", "doy_cos",
    "hdd", "cdd", "hdd_7d", "cdd_7d", "hdd_30d", "cdd_30d", "dd_anom7",
    "fc_hdd7", "fc_cdd7", "fc_dd_anom7", "fc_rev",
    "stor_level", "stor_dev5y", "stor_wow", "stor_yoy",
    "hh_spot", "hh_chg5",
]
REDUCED_FEATURES = [f for f in FULL_FEATURES
                    if f not in ("fc_hdd7", "fc_cdd7", "fc_dd_anom7", "fc_rev")]
VARIANTS = {"FULL": FULL_FEATURES, "REDUCED": REDUCED_FEATURES}
MODELS = ["climatology", "zero_change", "persistence", "ridge", "gbm"]
LEARNED = ["ridge", "gbm"]

ANCHORS = {"2024-04-29": 25.77, "2024-10-30": 21.27}


def season_for_month(month: int) -> str:
    """Spec D3: heating Nov-Mar, shoulder Apr-May + Sep-Oct, cooling Jun-Aug."""
    if month in (11, 12, 1, 2, 3):
        return "heating"
    if month in (4, 5, 9, 10):
        return "shoulder"
    return "cooling"  # 6, 7, 8


def load_panel(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path or PANEL_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    assert df["date"].is_monotonic_increasing and df["date"].is_unique
    return df


def build_sample(panel: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Rows where the variant's features AND label_ret5 are all non-NaN,
    indexed by date, carrying full_pos (panel position) for the harness."""
    keep = panel[features + [TARGET]].notna().all(axis=1)
    s = panel.loc[keep].copy()
    out = pd.DataFrame(index=pd.DatetimeIndex(s["date"], name="date"))
    for c in features + [TARGET, "ret5", "roll_window"]:
        out[c] = s[c].to_numpy()
    out["full_pos"] = np.where(keep.to_numpy())[0]
    out["season"] = [season_for_month(d.month) for d in out.index]
    out["doy"] = out.index.dayofyear.to_numpy()
    out = out.reset_index().rename(columns={"index": "date"})
    out["sample_pos"] = np.arange(len(out))
    return out.set_index("date")


# --- Baselines -----------------------------------------------------------------
def climatology_predict(sample: pd.DataFrame, test_pos: np.ndarray) -> np.ndarray:
    """Spec §6 model 1: mean of label_ret5 over sample days s with circular
    day-of-year distance <= 15 from t AND s at sample position <= i - 6
    (label fully realised at least 6 trading days before t). Expanding,
    deterministic, NOT fold-restricted."""
    y = sample[TARGET].to_numpy(dtype=float)
    doy = sample["doy"].to_numpy(dtype=float)
    preds = np.empty(len(test_pos), dtype=float)
    for k, i in enumerate(test_pos):
        dist = np.abs(doy - doy[i])
        dist = np.minimum(dist, 366.0 - dist)
        elig = (dist <= 15) & (np.arange(len(y)) <= i - 6)
        assert elig.any(), f"climatology: no eligible history for sample pos {i}"
        preds[k] = float(np.mean(y[elig]))
    return preds


# --- Integrity gate (spec §7; runs BEFORE any fitting) -------------------------
def run_integrity_gate(panel: pd.DataFrame, samples: dict[str, pd.DataFrame],
                       folds: list) -> dict:
    gate: dict = {"checks": {}, "passed": True}

    def record(name, ok, detail):
        gate["checks"][name] = {"pass": bool(ok), "detail": detail}
        gate["passed"] = gate["passed"] and bool(ok)

    # 1. Zero NaNs + frozen sample counts + identical dates across variants.
    nan_detail, ok_all = {}, True
    for vname, feats in VARIANTS.items():
        s = samples[vname]
        n_nan = int(s[feats + [TARGET]].isna().sum().sum())
        nan_detail[vname] = {"n_rows": int(len(s)), "n_nan_cells": n_nan,
                             "start": str(s.index[0].date()),
                             "end": str(s.index[-1].date())}
        ok_all = ok_all and (n_nan == 0) and (len(s) == EXPECTED_SAMPLE_N)
    same_dates = bool((samples["FULL"].index == samples["REDUCED"].index).all())
    record("sample_no_nans_and_frozen_count", ok_all and same_dates,
           {**nan_detail, "identical_dates_across_variants": same_dates,
            "expected_n_per_variant": EXPECTED_SAMPLE_N})

    # 2. Fold integrity (harness assertions already ran inside make_folds;
    #    re-verify explicitly and report purged counts).
    fold_detail, ok_folds = [], True
    for f in folds:
        tr_dates, te_dates = samples["FULL"].index[f.train_idx], samples["FULL"].index[f.test_idx]
        gap_days = int(samples["FULL"]["full_pos"].to_numpy()[f.test_idx[0]]
                       - samples["FULL"]["full_pos"].to_numpy()[f.train_idx].max())
        ok = bool(tr_dates.max() < te_dates.min() and gap_days > H.EMBARGO_DAYS)
        ok_folds = ok_folds and ok
        fold_detail.append({"fold": int(f.fold_id), "n_train": int(len(f.train_idx)),
                            "n_test": int(len(f.test_idx)),
                            "n_purged": int(f.test_idx[0] - len(f.train_idx)),
                            "train_end": str(tr_dates.max().date()),
                            "test_start": str(te_dates.min().date()),
                            "gap_trading_days": gap_days})
    record("fold_integrity", ok_folds, {"folds": fold_detail,
                                        "rule": "train max date + embargo < test min date; "
                                                "purge: train label window (t, t+5] must end before first test date"})

    # 3. Hand-recompute label_ret5 for 3 sample dates from ng_close.
    closes = panel["ng_close"].to_numpy(dtype=float)
    panel_pos_by_date = {d: i for i, d in enumerate(pd.to_datetime(panel["date"]))}
    sample_dates = samples["FULL"].index
    picks = [sample_dates[0], sample_dates[len(sample_dates) // 2], sample_dates[-1]]
    recs, max_diff = [], 0.0
    panel_label = panel.set_index("date")[TARGET]
    for d in picks:
        p = panel_pos_by_date[d]
        recomputed = float(np.log(closes[p + 5] / closes[p]))
        stored = float(panel_label.loc[d])
        max_diff = max(max_diff, abs(recomputed - stored))
        recs.append({"date": str(d.date()), "stored": stored,
                     "recomputed_from_ng_close": recomputed,
                     "abs_diff": abs(recomputed - stored)})
    record("label_hand_recomputation", max_diff == 0.0,
           {"dates": recs, "max_abs_diff": max_diff})

    # 4. Truncation check carried forward from the dataset self-checks.
    sc = json.loads(SELF_CHECKS_PATH.read_text())
    trunc = sc["truncation_check_2022-12-31"]
    t_max = max(v.get("max_abs_diff", 0.0) for v in trunc.values())
    t_nan = sum(v.get("nan_pattern_mismatches", 0) for v in trunc.values())
    t_roll = trunc["roll_window"]["mismatches"]
    record("truncation_check_carried_forward", t_max == 0.0 and t_nan == 0 and t_roll == 0,
           {"source": "data/t4a_weather_futures/self_checks.json (dataset card self-check 3)",
            "max_abs_diff_all_features": t_max, "nan_pattern_mismatches": int(t_nan),
            "roll_window_mismatches": int(t_roll), "cited_not_redone": True})

    # 5. Audit anchors in roll_window=True rows.
    anchor_detail, ok_anchor = {}, True
    by_date = panel.set_index("date")
    for dstr, card_pct in ANCHORS.items():
        d = pd.Timestamp(dstr)
        p = panel_pos_by_date[d]
        one_day_pct = float((closes[p] / closes[p - 1] - 1.0) * 100.0)
        in_rw = bool(by_date.loc[d, "roll_window"])
        ok_anchor = ok_anchor and in_rw
        anchor_detail[dstr] = {"one_day_move_pct_recomputed": one_day_pct,
                               "one_day_move_pct_dataset_card": card_pct,
                               "roll_window": in_rw}
    record("audit_anchors_in_roll_window", ok_anchor, anchor_detail)
    return gate


# --- Walk-forward run ------------------------------------------------------------
def run_walk_forward(sample: pd.DataFrame, features: list[str], folds: list):
    """Returns (predictions DataFrame in raw log units, fold meta, ridge alphas)."""
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
                    date=sample.index[pos], variant=None, model=model,
                    y_true=float(yte[i]), y_pred=float(p[i]), fold=int(f.fold_id),
                    roll_window=bool(r["roll_window"]), season=str(r["season"]),
                ))
        fold_meta.append(dict(fold=int(f.fold_id), **f.meta,
                              n_purged=int(f.test_idx[0] - len(f.train_idx))))
    return pd.DataFrame(rows), fold_meta, alphas


# --- Metrics (spec §8; MAE/RMSE reported in %) ------------------------------------
def _block(df: pd.DataFrame) -> dict:
    yt, yp = df["y_true"].to_numpy() * 100.0, df["y_pred"].to_numpy() * 100.0
    return {"n": int(len(df)), "mae_pct": float(np.mean(np.abs(yt - yp))),
            "rmse_pct": float(np.sqrt(np.mean((yt - yp) ** 2)))}


def model_metrics(preds_variant: pd.DataFrame) -> dict:
    clim = preds_variant[preds_variant.model == "climatology"]
    pers = preds_variant[preds_variant.model == "persistence"]
    clim_mae = _block(clim)["mae_pct"]
    pers_mae = _block(pers)["mae_pct"]
    out = {}
    for model in MODELS:
        d = preds_variant[preds_variant.model == model].sort_values("date")
        b = _block(d)
        entry = {**b,
                 "mae_ratio_vs_climatology": b["mae_pct"] / clim_mae,
                 "mae_ratio_vs_persistence": b["mae_pct"] / pers_mae}
        if model == "zero_change":
            entry["directional_accuracy"] = None
            entry["realized_positive_share_of_test_labels"] = float(
                np.mean(d["y_true"].to_numpy() > 0))
        else:
            entry["directional_accuracy"] = float(np.mean(
                np.sign(d["y_pred"].to_numpy()) == np.sign(d["y_true"].to_numpy())))
        entry["slices"] = {
            "roll": _block(d[d.roll_window]),
            "non_roll": _block(d[~d.roll_window]),
            "seasons": {s: _block(d[d.season == s])
                        for s in ("heating", "shoulder", "cooling")},
        }
        entry["per_fold"] = {str(fid): _block(g) for fid, g in d.groupby("fold")}
        out[model] = entry
    return out


def graduation_verdicts(metrics_by_variant: dict[str, dict]) -> dict:
    verdicts = {}
    for vname, m in metrics_by_variant.items():
        clim = m["climatology"]
        verdicts[vname] = {}
        for model in LEARNED:
            mm = m[model]
            leg_a = bool(mm["mae_pct"] < clim["mae_pct"])
            leg_b = bool(mm["slices"]["non_roll"]["mae_pct"]
                         < clim["slices"]["non_roll"]["mae_pct"])
            season_losses = [s for s in ("heating", "shoulder", "cooling")
                             if mm["slices"]["seasons"][s]["mae_pct"]
                             > clim["slices"]["seasons"][s]["mae_pct"]]
            if leg_a and leg_b and len(season_losses) < 2:
                verdict = "GRADUATES"
            elif leg_a and leg_b:
                verdict = "SEASON-SCOPED (not graduation)"
            else:
                verdict = "FAIL (does not graduate)"
            verdicts[vname][model] = {
                "verdict": verdict,
                "leg_a_pooled_mae_beats_climatology": leg_a,
                "leg_b_non_roll_mae_beats_climatology": leg_b,
                "season_slices_lost_to_climatology": season_losses,
                "n_season_slices_lost": len(season_losses),
            }
    return verdicts
