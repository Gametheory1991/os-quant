"""ML Stage C-3 — volatility-state / move-size targets (V-a, V-b).

Frozen spec: reports/ml_stage_c3_volatility_move_spec_2026-10-27.md
(written 2026-10-27 before any fitting). Implements ONLY what that spec
freezes: leg V-a (VIX top-trailing-quintile 5-day-ahead classification on
the Target 1 frozen-14 features) and leg V-b (absolute 5-day natural gas
return regression on the Target 4a REDUCED-18 panel). Walk-forward folds,
purge/embargo, baselines, and the model ladder are reused unmodified from
engine/ml_harness.py, engine/ml_models.py and engine/ml_t4a.py.

Offline evaluation only — nothing here is issued or scheduled.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine import ml_harness as H
from engine import ml_models as M
from engine import ml_t4a as T

ROOT = Path(__file__).resolve().parent.parent
VOL_CACHE = (Path.home() / "workspace" / "goals"
             / "ultimate-arbitrage-screener-mispricing-dislocation-flagging-machine"
             / "hidden_files" / "vol_cache")

HORIZON = 5
TRAILING_WINDOW = 252          # panel positions, ending at t (spec §2)
QUANTILE = 0.80                # top trailing quintile
FROZEN_END_VB = pd.Timestamp("2026-10-16")   # frozen Target 4a sample end
EXPECTED_VB_N = 2952

VA_MODELS = ["climatology", "state_rule", "logistic", "gbm"]
VA_LEARNED = ["logistic", "gbm"]
VB_MODELS = ["trailing_mean", "seasonal_climatology_abs", "ridge", "gbm"]
VB_LEARNED = ["ridge", "gbm"]
VB_BASELINES = ["trailing_mean", "seasonal_climatology_abs"]


# ===========================================================================
# Leg V-a — VIX top-trailing-quintile state
# ===========================================================================
def build_va_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """Threshold and label by PANEL position (spec §2).

    thr[p] = 80th percentile of vix[p-251 .. p] (all available at p);
    y[p]   = 1{ vix[p+5] >= thr[p] }. NaN where either is not computable.
    Also event_state[p] = 1{ vix[p] >= thr[p] } (state_rule input)."""
    vix = panel["vix"].to_numpy(dtype=float)
    n = len(vix)
    assert np.isfinite(vix).all(), "panel vix column contains NaN"
    thr = np.full(n, np.nan)
    for p in range(TRAILING_WINDOW - 1, n):
        thr[p] = float(np.quantile(vix[p - TRAILING_WINDOW + 1: p + 1], QUANTILE))
    y = np.full(n, np.nan)
    ok = np.isfinite(thr[: n - HORIZON])
    fut = vix[HORIZON:]
    y[: n - HORIZON] = np.where(ok, (fut >= thr[: n - HORIZON]).astype(float), np.nan)
    out = pd.DataFrame(index=panel.index)
    out["thr"] = thr
    out["y"] = y
    out["event_state"] = np.where(np.isfinite(thr), (vix >= thr).astype(float), np.nan)
    return out


def build_va_sample(panel: pd.DataFrame) -> pd.DataFrame:
    X = H.build_feature_matrix(panel)
    lab = build_va_labels(panel)
    df = X.join(lab)
    df["full_pos"] = np.arange(len(panel))
    keep = df[H.FEATURE_NAMES].notna().all(axis=1) & df["y"].notna()
    sample = df.loc[keep].copy()
    sample["sample_pos"] = np.arange(len(sample))
    return sample


def run_va_walk_forward(sample: pd.DataFrame, folds: list):
    X = sample[H.FEATURE_NAMES]
    y_all = sample["y"].to_numpy(dtype=float)
    ev_all = sample["event_state"].to_numpy(dtype=float)
    rows, fold_meta, hypers = [], [], {}
    for f in folds:
        Xtr, Xte = X.iloc[f.train_idx], X.iloc[f.test_idx]
        ytr, yte = y_all[f.train_idx], y_all[f.test_idx]
        thr_te = sample["thr"].to_numpy()[f.test_idx]
        preds = {}
        preds["climatology"], hp_c = M.climatology_clf_predict(ytr, len(f.test_idx))
        preds["state_rule"], _ = M.event_rule_predict(
            ev_all[f.train_idx], ytr, ev_all[f.test_idx])
        preds["logistic"], hp_l = M.logistic_fit_predict(Xtr, ytr, Xte)
        preds["gbm"], _ = M.gbm_clf_fit_predict(Xtr, ytr, Xte)
        hypers[f"fold{f.fold_id}"] = {"logistic_C": hp_l["C"],
                                      "climatology_rate": hp_c["train_rate"]}
        for model in VA_MODELS:
            p = preds[model]
            for i, pos in enumerate(f.test_idx):
                rows.append(dict(date=sample.index[pos], fold=int(f.fold_id),
                                 model=model, y_true=float(yte[i]),
                                 y_proba=float(p[i]), threshold=float(thr_te[i])))
        fold_meta.append(dict(fold=int(f.fold_id), **f.meta,
                              n_purged=int(f.test_idx[0] - len(f.train_idx))))
    return pd.DataFrame(rows), fold_meta, hypers


def va_metrics(preds: pd.DataFrame) -> dict:
    clim = preds[preds.model == "climatology"].sort_values("date")
    null_rate = float(clim["y_proba"].mean())
    base_rate = float(clim["y_true"].mean())
    out = {"n_test": int(len(clim)), "base_rate": base_rate,
           "climatology_null_rate": null_rate, "models": {}}
    for model in VA_MODELS:
        d = preds[preds.model == model].sort_values("date")
        entry = {"brier": H.brier(d["y_true"], d["y_proba"]),
                 "auc": H.auc(d["y_true"], d["y_proba"]),
                 "per_fold_brier": {str(fid): H.brier(g["y_true"], g["y_proba"])
                                    for fid, g in d.groupby("fold")}}
        if model in VA_LEARNED:
            entry["alert"] = H.alert_budget_test(d["y_true"], d["y_proba"], null_rate)
        out["models"][model] = entry
    clim_brier = out["models"]["climatology"]["brier"]
    verdicts = {}
    for model in VA_LEARNED:
        e = out["models"][model]
        leg_brier = bool(e["brier"] < clim_brier)
        leg_alert = bool(e["alert"]["binomial_p"] < 0.05)
        verdicts[model] = {
            "verdict": ("PASS (within-experiment only — not a graduation)"
                        if (leg_brier and leg_alert) else
                        "FAIL (does not pass the within-experiment bar)"),
            "leg_brier_beats_climatology": leg_brier,
            "leg_alert_binomial_p_lt_0.05": leg_alert,
            "brier": e["brier"], "climatology_brier": clim_brier,
            "alert_hit_rate": e["alert"]["alert_hit_rate"],
            "alert_binomial_p": e["alert"]["binomial_p"],
            "n_alert": e["alert"]["n_alert"], "alert_hits": e["alert"]["alert_hits"],
        }
    out["verdicts"] = verdicts
    return out


def va_integrity_gate(panel: pd.DataFrame, sample: pd.DataFrame, folds: list) -> dict:
    gate: dict = {"checks": {}, "passed": True}

    def record(name, ok, detail):
        gate["checks"][name] = {"pass": bool(ok), "detail": detail}
        gate["passed"] = gate["passed"] and bool(ok)

    n_nan = int(sample[H.FEATURE_NAMES + ["y", "thr"]].isna().sum().sum())
    record("sample_no_nans", n_nan == 0,
           {"n_rows": int(len(sample)), "n_nan_cells": n_nan,
            "start": str(sample.index[0].date()), "end": str(sample.index[-1].date())})

    fold_detail, ok_folds = [], True
    full_pos = sample["full_pos"].to_numpy()
    for f in folds:
        gap = int(full_pos[f.test_idx[0]] - full_pos[f.train_idx].max())
        ok = bool(sample.index[f.train_idx].max() < sample.index[f.test_idx].min()
                  and gap > H.EMBARGO_DAYS)
        ok_folds = ok_folds and ok
        fold_detail.append({"fold": int(f.fold_id), "n_train": int(len(f.train_idx)),
                            "n_test": int(len(f.test_idx)),
                            "n_purged": int(f.test_idx[0] - len(f.train_idx)),
                            "gap_trading_days": gap})
    record("fold_integrity", ok_folds, {"folds": fold_detail})

    vix = panel["vix"].to_numpy(dtype=float)
    picks = [sample.index[0], sample.index[len(sample) // 2], sample.index[-1]]
    recs, max_diff = [], 0.0
    for d in picks:
        p = int(sample.loc[d, "full_pos"])
        thr_r = float(np.quantile(vix[p - TRAILING_WINDOW + 1: p + 1], QUANTILE))
        y_r = float(vix[p + HORIZON] >= thr_r)
        max_diff = max(max_diff, abs(thr_r - float(sample.loc[d, "thr"])),
                       abs(y_r - float(sample.loc[d, "y"])))
        recs.append({"date": str(d.date()), "thr_stored": float(sample.loc[d, "thr"]),
                     "thr_recomputed": thr_r, "y_stored": float(sample.loc[d, "y"]),
                     "y_recomputed": y_r})
    record("label_hand_recomputation", max_diff == 0.0,
           {"dates": recs, "max_abs_diff": max_diff})

    cache_detail, ok_cache = {}, True
    for name in ("VIX", "VVIX", "VIX3M"):
        c = pd.read_csv(VOL_CACHE / f"{name}.csv", parse_dates=["date"])
        good = bool(c["date"].is_monotonic_increasing and c["date"].is_unique
                    and c["close"].notna().all())
        ok_cache = ok_cache and good
        cache_detail[name] = {"n_rows": int(len(c)), "start": str(c["date"].min().date()),
                              "end": str(c["date"].max().date()),
                              "monotone_unique_no_nan": good}
    cvix = pd.read_csv(VOL_CACHE / "VIX.csv", parse_dates=["date"]).set_index("date")["close"]
    j = pd.DataFrame({"store": panel["vix"], "cache": cvix}).dropna()
    diff = (j["store"] - j["cache"]).abs()
    cache_detail["store_vs_cache_vix"] = {
        "n_common_dates": int(len(j)), "median_abs_diff": float(diff.median()),
        "max_abs_diff": float(diff.max()),
        "share_days_abs_diff_gt_0.05": float((diff > 0.05).mean()),
        "note": "different transports (store: Yahoo series as-of panel date; "
                "cache: Cboe closes) — reported, not a pass criterion (spec §4)"}
    record("vol_cache_integrity", ok_cache, cache_detail)
    return gate


# ===========================================================================
# Leg V-b — absolute 5-day natural gas return (Target 4a REDUCED panel)
# ===========================================================================
def build_vb_sample(panel: pd.DataFrame) -> pd.DataFrame:
    restricted = panel[panel["date"] <= FROZEN_END_VB].copy()
    sample = T.build_sample(restricted, T.REDUCED_FEATURES)
    sample["y_abs"] = sample[T.TARGET].abs()
    return sample


def _expanding_baseline(sample: pd.DataFrame, test_pos: np.ndarray,
                        seasonal: bool) -> np.ndarray:
    y = sample["y_abs"].to_numpy(dtype=float)
    doy = sample["doy"].to_numpy(dtype=float)
    preds = np.empty(len(test_pos), dtype=float)
    for k, i in enumerate(test_pos):
        elig = np.arange(len(y)) <= i - 6
        if seasonal:
            dist = np.abs(doy - doy[i])
            dist = np.minimum(dist, 366.0 - dist)
            elig = elig & (dist <= 15)
        assert elig.any(), f"V-b baseline: no eligible history for sample pos {i}"
        preds[k] = float(np.mean(y[elig]))
    return preds


def run_vb_walk_forward(sample: pd.DataFrame, folds: list):
    X = sample[T.REDUCED_FEATURES]
    y_all = sample["y_abs"].to_numpy(dtype=float)
    rows, fold_meta, alphas = [], [], {}
    for f in folds:
        Xtr, Xte = X.iloc[f.train_idx], X.iloc[f.test_idx]
        ytr, yte = y_all[f.train_idx], y_all[f.test_idx]
        test_rows = sample.iloc[f.test_idx]
        preds = {
            "trailing_mean": _expanding_baseline(sample, f.test_idx, seasonal=False),
            "seasonal_climatology_abs": _expanding_baseline(sample, f.test_idx, seasonal=True),
        }
        preds["ridge"], hp = M.ridge_fit_predict(Xtr, ytr, Xte)
        alphas[f"fold{f.fold_id}"] = hp["alpha"]
        preds["gbm"], _ = M.gbm_reg_fit_predict(Xtr, ytr, Xte)
        for model in VB_MODELS:
            p = preds[model]
            for i, pos in enumerate(f.test_idx):
                r = test_rows.iloc[i]
                rows.append(dict(date=sample.index[pos], fold=int(f.fold_id),
                                 model=model, y_true=float(yte[i]),
                                 y_pred=float(p[i]),
                                 roll_window=bool(r["roll_window"]),
                                 season=str(r["season"])))
        fold_meta.append(dict(fold=int(f.fold_id), **f.meta,
                              n_purged=int(f.test_idx[0] - len(f.train_idx))))
    return pd.DataFrame(rows), fold_meta, alphas


def _block(df: pd.DataFrame) -> dict:
    yt, yp = df["y_true"].to_numpy() * 100.0, df["y_pred"].to_numpy() * 100.0
    return {"n": int(len(df)), "mae_pct": float(np.mean(np.abs(yt - yp))),
            "rmse_pct": float(np.sqrt(np.mean((yt - yp) ** 2)))}


def vb_metrics(preds: pd.DataFrame) -> dict:
    base_mae = {b: _block(preds[preds.model == b])["mae_pct"] for b in VB_BASELINES}
    base_nr = {b: _block(preds[(preds.model == b) & (~preds.roll_window)])["mae_pct"]
               for b in VB_BASELINES}
    out = {"models": {}, "verdicts": {}}
    for model in VB_MODELS:
        d = preds[preds.model == model].sort_values("date")
        b = _block(d)
        entry = {**b,
                 "mae_ratio_vs_trailing_mean": b["mae_pct"] / base_mae["trailing_mean"],
                 "mae_ratio_vs_seasonal_climatology_abs":
                     b["mae_pct"] / base_mae["seasonal_climatology_abs"],
                 "slices": {"roll": _block(d[d.roll_window]),
                            "non_roll": _block(d[~d.roll_window]),
                            "seasons": {s: _block(d[d.season == s])
                                        for s in ("heating", "shoulder", "cooling")}},
                 "per_fold": {str(fid): _block(g) for fid, g in d.groupby("fold")}}
        out["models"][model] = entry
    for model in VB_LEARNED:
        e = out["models"][model]
        leg_a = all(e["mae_pct"] < base_mae[b] for b in VB_BASELINES)
        leg_b = all(e["slices"]["non_roll"]["mae_pct"] < base_nr[b] for b in VB_BASELINES)
        out["verdicts"][model] = {
            "verdict": ("PASS (within-experiment only — not a graduation)"
                        if (leg_a and leg_b) else
                        "FAIL (does not pass the within-experiment bar)"),
            "leg_a_pooled_mae_beats_both_baselines": bool(leg_a),
            "leg_b_non_roll_mae_beats_both_baselines": bool(leg_b),
            "baseline_pooled_mae_pct": base_mae,
            "baseline_non_roll_mae_pct": base_nr,
        }
    return out


def vb_integrity_gate(panel: pd.DataFrame, sample: pd.DataFrame, folds: list) -> dict:
    gate: dict = {"checks": {}, "passed": True}

    def record(name, ok, detail):
        gate["checks"][name] = {"pass": bool(ok), "detail": detail}
        gate["passed"] = gate["passed"] and bool(ok)

    n_nan = int(sample[T.REDUCED_FEATURES + [T.TARGET]].isna().sum().sum())
    record("sample_no_nans_and_frozen_count",
           n_nan == 0 and len(sample) == EXPECTED_VB_N,
           {"n_rows": int(len(sample)), "expected_n": EXPECTED_VB_N,
            "n_nan_cells": n_nan, "start": str(sample.index[0].date()),
            "end": str(sample.index[-1].date())})

    fold_detail, ok_folds = [], True
    full_pos = sample["full_pos"].to_numpy()
    for f in folds:
        gap = int(full_pos[f.test_idx[0]] - full_pos[f.train_idx].max())
        ok = bool(sample.index[f.train_idx].max() < sample.index[f.test_idx].min()
                  and gap > H.EMBARGO_DAYS)
        ok_folds = ok_folds and ok
        fold_detail.append({"fold": int(f.fold_id), "n_train": int(len(f.train_idx)),
                            "n_test": int(len(f.test_idx)),
                            "n_purged": int(f.test_idx[0] - len(f.train_idx)),
                            "gap_trading_days": gap})
    record("fold_integrity", ok_folds, {"folds": fold_detail, "n_folds": len(folds)})

    closes = panel["ng_close"].to_numpy(dtype=float)
    pos_by_date = {d: i for i, d in enumerate(pd.to_datetime(panel["date"]))}
    stored = panel.set_index("date")[T.TARGET]
    picks = [sample.index[0], sample.index[len(sample) // 2], sample.index[-1]]
    recs, max_diff = [], 0.0
    for d in picks:
        p = pos_by_date[d]
        recomputed = float(abs(np.log(closes[p + 5] / closes[p])))
        stored_abs = float(abs(stored.loc[d]))
        max_diff = max(max_diff, abs(recomputed - stored_abs))
        recs.append({"date": str(d.date()), "stored_abs_label": stored_abs,
                     "recomputed_abs_from_ng_close": recomputed,
                     "abs_diff": abs(recomputed - stored_abs)})
    record("label_hand_recomputation", max_diff == 0.0,
           {"dates": recs, "max_abs_diff": max_diff})

    sc = json.loads(T.SELF_CHECKS_PATH.read_text())
    trunc = sc["truncation_check_2022-12-31"]
    t_max = max(v.get("max_abs_diff", 0.0) for v in trunc.values())
    t_nan = sum(v.get("nan_pattern_mismatches", 0) for v in trunc.values())
    t_roll = trunc["roll_window"]["mismatches"]
    record("truncation_check_carried_forward",
           t_max == 0.0 and t_nan == 0 and t_roll == 0,
           {"source": "data/t4a_weather_futures/self_checks.json",
            "max_abs_diff_all_features": t_max,
            "nan_pattern_mismatches": int(t_nan),
            "roll_window_mismatches": int(t_roll), "cited_not_redone": True})
    return gate
