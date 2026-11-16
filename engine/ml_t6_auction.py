"""ML Stage C-1 (proposal T6) — auction-day liquidity conditioning and
the joint auction-widening event.

Frozen spec: reports/ml_t6_auction_conditioning_spec_2026-10-27.md
(written before any fit). All protocol machinery (folds, purge,
embargo, thresholds, metrics) is reused unmodified from
engine/ml_harness.py and engine/ml_models.py; the Phase 1 binomial
function is reused from engine/ml_event_study.py. This module adds
only the coupon derivations and the three frozen computations.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from engine import ml_event_study as ES
from engine import ml_harness as H
from engine import ml_models as M

ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "data" / "ml"
AUCTIONS_CSV = ML_DIR / "auctions.csv"
PANEL_END_ORIGINAL = "2026-10-02"  # original v1 panel end (spec §5)
RECORD_T1A1_RIDGE_MAE = 1.1936506923576809e-05  # reports/ml_target1_results.json
MACHINERY_RTOL = 1e-6

AUCTION_BLOCK = ["auction_amt", "coupon_day", "coupon_amt"]


# ---------------------------------------------------------------------------
# Coupon derivations (point-in-time: auctions dated exactly t only)
# ---------------------------------------------------------------------------
def load_auctions(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or AUCTIONS_CSV, parse_dates=["auction_date"])
    df["auction_date"] = df["auction_date"].dt.normalize()
    df["offering_amt"] = pd.to_numeric(df["offering_amt"], errors="coerce")
    return df


def coupon_calendar(auctions: pd.DataFrame) -> pd.DataFrame:
    """Per-date coupon aggregates: coupon_day (0/1) and coupon_amt ($B).

    Coupon := security_type in {Note, Bond} (spec §3). Only auctions
    dated exactly that date contribute — no forward fill, no week
    aggregation — so the value at t uses no auction dated after t.
    """
    c = auctions[auctions["security_type"].isin(["Note", "Bond"])]
    g = c.groupby("auction_date")["offering_amt"].agg(["size", "sum"])
    out = pd.DataFrame(index=g.index)
    out["coupon_day"] = 1
    out["coupon_amt"] = g["sum"].to_numpy(dtype=float) / 1e9
    return out


def coupon_features_on(index: pd.DatetimeIndex,
                       auctions: pd.DataFrame) -> pd.DataFrame:
    """coupon_day / coupon_amt aligned to ``index`` (0 / 0.0 elsewhere)."""
    cal = coupon_calendar(auctions)
    out = pd.DataFrame(index=index)
    out["coupon_day"] = index.isin(cal.index).astype(float)
    amt = cal["coupon_amt"].reindex(index)
    out["coupon_amt"] = amt.fillna(0.0).to_numpy(dtype=float)
    # consistency: coupon_amt > 0 exactly where coupon_day == 1
    assert ((out["coupon_amt"] > 0) == (out["coupon_day"] == 1)).all()
    return out


def auction_leg(sample: pd.DataFrame, panel_index: pd.DatetimeIndex,
                coupon_dates: pd.DatetimeIndex, horizon: int = 5) -> np.ndarray:
    """A(t) = 1{>= 1 coupon auction on panel positions t+1 .. t+horizon}.

    Positions are the sample's full-panel positions (full_pos), the
    same positions that govern purge/embargo in the harness.
    """
    coup = set(panel_index.get_indexer(coupon_dates))
    coup = {p for p in coup if p >= 0}
    fp = sample["full_pos"].to_numpy()
    out = np.zeros(len(sample), dtype=float)
    for i, p in enumerate(fp):
        out[i] = 1.0 if any((p + k) in coup for k in range(1, horizon + 1)) else 0.0
    return out


# ---------------------------------------------------------------------------
# Leg (a) — event study (Phase 1 form; spec §4)
# ---------------------------------------------------------------------------
def trailing_median(s: pd.Series, window: int = 252,
                    min_periods: int = 126) -> pd.Series:
    return s.rolling(window, min_periods=min_periods).median()


def event_study(panel: pd.DataFrame, auctions: pd.DataFrame) -> dict:
    illiq = panel["illiq_agg"].astype(float)
    med = trailing_median(illiq)
    evaluated_mask = illiq.notna() & med.notna()
    ev_dates = panel.index[evaluated_mask]
    ill = illiq.loc[ev_dates]
    elevated = (ill >= med.loc[ev_dates])

    coup = coupon_features_on(panel.index, auctions)
    pos = {d: i for i, d in enumerate(panel.index)}
    coupon_pos = {pos[d] for d in panel.index[coup["coupon_day"].to_numpy() == 1]}
    w0 = pd.DatetimeIndex([d for d in ev_dates if pos[d] in coupon_pos])
    w1 = pd.DatetimeIndex([panel.index[p + 1] for p in sorted(coupon_pos)
                            if p + 1 < len(panel)])
    w1 = w1[w1.isin(ev_dates)]
    w01 = w0.union(w1)
    wr = ev_dates[panel.loc[ev_dates, "refunding_week"].to_numpy() == 1]
    baseline_dates = ev_dates.difference(w01.union(wr))

    base_ill = ill.loc[baseline_dates]
    base_elev = elevated.loc[baseline_dates]
    base_rate = float(base_elev.mean())

    def window_stats(dates: pd.DatetimeIndex) -> dict:
        sel = ill.loc[dates]
        hits = int(elevated.loc[dates].sum())
        p = ES.binomial_p_ge(hits, len(dates), base_rate)
        return {
            "n": int(len(dates)),
            "mean_illiq": float(sel.mean()),
            "median_illiq": float(sel.median()),
            "mean_ratio_vs_baseline": float(sel.mean() / base_ill.mean()),
            "hits": hits,
            "hit_rate": float(hits / len(dates)) if len(dates) else None,
            "binomial_p": float(p),
            "pass_p_lt_0.05": bool(p < 0.05),
        }

    windows = {
        "W0_coupon_day": window_stats(w0),
        "W+1_day_after_coupon": window_stats(w1),
        "W01_window_0_to_+1": window_stats(w01),
        "WR_refunding_week_days_expost": window_stats(wr),
    }

    # Secondary: refunding week as the unit (spec §4).
    mondays = ev_dates - pd.to_timedelta(ev_dates.weekday, unit="D")
    frame = pd.DataFrame({"illiq": ill.to_numpy(),
                          "monday": mondays,
                          "refunding": panel.loc[ev_dates, "refunding_week"]
                          .to_numpy() == 1}, index=ev_dates)
    week_mean = frame.groupby("monday")["illiq"].mean()
    week_ref = frame.groupby("monday")["refunding"].first()
    base_median = float(base_ill.median())
    ref_weeks = week_mean[week_ref]
    nonref_weeks = week_mean[~week_ref]
    ref_hits = int((ref_weeks > base_median).sum())
    nonref_rate = float((nonref_weeks > base_median).mean())
    p_week = ES.binomial_p_ge(ref_hits, len(ref_weeks), nonref_rate)
    week_level = {
        "n_refunding_weeks": int(len(ref_weeks)),
        "n_nonrefunding_weeks": int(len(nonref_weeks)),
        "refunding_week_mean_illiq": float(ref_weeks.mean()),
        "nonrefunding_week_mean_illiq": float(nonref_weeks.mean()),
        "hits": ref_hits,
        "hit_rate": float(ref_hits / len(ref_weeks)) if len(ref_weeks) else None,
        "baseline_hit_rate": nonref_rate,
        "binomial_p": float(p_week),
        "pass_p_lt_0.05": bool(p_week < 0.05),
    }

    return {
        "evaluated_n": int(len(ev_dates)),
        "evaluated_start": str(ev_dates.min().date()),
        "evaluated_end": str(ev_dates.max().date()),
        "baseline": {
            "n": int(len(baseline_dates)),
            "mean_illiq": float(base_ill.mean()),
            "median_illiq": float(base_ill.median()),
            "hits": int(base_elev.sum()),
            "hit_rate": base_rate,
        },
        "windows": windows,
        "refunding_week_level_secondary": week_level,
        "refunding_distinct_weeks_evaluated": int(len(ref_weeks)),
    }


# ---------------------------------------------------------------------------
# Legs (a)-ridge and (b) — shared truncated-panel sample (spec §5–§6)
# ---------------------------------------------------------------------------
def truncated_sample() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    panel_full = H.load_panel()
    panel = panel_full.loc[:PANEL_END_ORIGINAL]
    oas = H.load_oas()
    sample = H.build_sample(panel, oas)
    return panel, sample, oas


def augmented_matrix(panel: pd.DataFrame, sample: pd.DataFrame,
                     auctions: pd.DataFrame) -> pd.DataFrame:
    """v1-frozen14 features plus the §3 auction block, on sample rows."""
    X = H.build_feature_matrix(panel).loc[sample.index].copy()
    coup = coupon_features_on(panel.index, auctions).loc[sample.index]
    X["auction_amt"] = panel.loc[sample.index, "auction_amt"].astype(float)
    X["coupon_day"] = coup["coupon_day"]
    X["coupon_amt"] = coup["coupon_amt"]
    assert list(X.columns) == H.FEATURE_NAMES + AUCTION_BLOCK
    assert X.notna().all().all()
    return X


def run_ridge_leg(panel, sample, auctions) -> tuple[dict, pd.DataFrame]:
    X_base = H.build_feature_matrix(panel).loc[sample.index]
    X_aug = augmented_matrix(panel, sample, auctions)
    y = sample["T1a1"].to_numpy(dtype=float)
    folds = H.make_folds(sample, horizon=1)
    rows, per_fold = [], {}
    preds = {"base": [], "aug": [], "pers": []}
    alphas = {}
    for f in folds:
        tr, te = f.train_idx, f.test_idx
        pb, hb = M.ridge_fit_predict(X_base.to_numpy()[tr], y[tr],
                                     X_base.to_numpy()[te])
        pa, ha = M.ridge_fit_predict(X_aug.to_numpy()[tr], y[tr],
                                     X_aug.to_numpy()[te])
        pp = M.persistence_predict(sample.iloc[te], "T1a1")
        yt = y[te]
        per_fold[str(f.fold_id)] = {
            "n_test": int(len(te)),
            "mae_base": H.mae(yt, pb), "mae_aug": H.mae(yt, pa),
            "mae_persistence": H.mae(yt, pp),
            "alpha_base": hb["alpha"], "alpha_aug": ha["alpha"],
        }
        alphas[f"fold{f.fold_id}"] = {"base": hb, "augmented": ha}
        for name, pr in (("base", pb), ("aug", pa), ("pers", pp)):
            preds[name].append(pr)
            if name != "pers":
                for d, v in zip(sample.index[te], pr):
                    rows.append({"date": d, "fold": f.fold_id,
                                 "target": "T1a1",
                                 "model": "ridge_frozen14" if name == "base"
                                 else "T6a-ridge-auctionblock",
                                 "y_true": float(sample.loc[d, "T1a1"]),
                                 "y_pred": float(v), "y_proba": np.nan})
    y_test = np.concatenate([y[f.test_idx] for f in folds])
    pb = np.concatenate(preds["base"]); pa = np.concatenate(preds["aug"])
    pp = np.concatenate(preds["pers"])
    mae_b, mae_a, mae_p = H.mae(y_test, pb), H.mae(y_test, pa), H.mae(y_test, pp)
    rel = abs(mae_b - RECORD_T1A1_RIDGE_MAE) / RECORD_T1A1_RIDGE_MAE
    out = {
        "n_test": int(len(y_test)),
        "mae_base_ridge": mae_b, "mae_augmented_ridge": mae_a,
        "mae_persistence": mae_p,
        "mae_ratio_base_vs_persistence": mae_b / mae_p,
        "mae_ratio_augmented_vs_persistence": mae_a / mae_p,
        "mae_ratio_augmented_vs_base": mae_a / mae_b,
        "augmented_beats_persistence": bool(mae_a < mae_p),
        "augmented_beats_base": bool(mae_a < mae_b),
        "alphas": alphas, "per_fold": per_fold,
        "machinery_check": {
            "record_mae": RECORD_T1A1_RIDGE_MAE,
            "rerun_mae": mae_b, "relative_difference": float(rel),
            "tolerance": MACHINERY_RTOL, "passed": bool(rel <= MACHINERY_RTOL),
        },
    }
    return out, pd.DataFrame(rows)


def run_joint_leg(panel, sample, auctions) -> tuple[dict, pd.DataFrame]:
    X = H.build_feature_matrix(panel).loc[sample.index]
    coup = coupon_features_on(panel.index, auctions)
    coupon_dates = panel.index[coup["coupon_day"].to_numpy() == 1]
    a_leg = auction_leg(sample, panel.index, coupon_dates, horizon=5)
    widen = sample["oas_hy_widen5"].to_numpy(dtype=float)
    folds = H.make_folds(sample, horizon=5)
    rows = []
    y_all, p_log, p_clim, fold_of = [], [], [], []
    per_fold, thresholds = {}, {}
    for f in folds:
        tr, te = f.train_idx, f.test_idx
        thr = H.fold_threshold(sample, tr, "oas_hy_widen5")
        thresholds[str(f.fold_id)] = thr
        j_tr = ((widen[tr] >= thr) & (a_leg[tr] == 1)).astype(float)
        j_te = ((widen[te] >= thr) & (a_leg[te] == 1)).astype(float)
        pl, hpar = M.logistic_fit_predict(X.to_numpy()[tr], j_tr,
                                          X.to_numpy()[te])
        pc, _ = M.climatology_clf_predict(j_tr, len(te))
        per_fold[str(f.fold_id)] = {
            "threshold": thr, "n_test": int(len(te)),
            "joint_base_rate_test": float(j_te.mean()),
            "widening_rate_test": float((widen[te] >= thr).mean()),
            "auction_leg_rate_test": float(a_leg[te].mean()),
            "brier_logistic": H.brier(j_te, pl),
            "brier_climatology": H.brier(j_te, pc),
            "C": hpar["C"],
        }
        y_all.append(j_te); p_log.append(pl); p_clim.append(pc)
        fold_of.append(np.full(len(te), f.fold_id))
        for d, yt, v, c in zip(sample.index[te], j_te, pl, pc):
            rows.append({"date": d, "fold": f.fold_id, "target": "T6b_joint",
                         "model": "logistic_frozen14", "y_true": float(yt),
                         "y_pred": np.nan, "y_proba": float(v)})
            rows.append({"date": d, "fold": f.fold_id, "target": "T6b_joint",
                         "model": "climatology_joint", "y_true": float(yt),
                         "y_pred": np.nan, "y_proba": float(c)})
    y = np.concatenate(y_all); pl = np.concatenate(p_log)
    pc = np.concatenate(p_clim)
    brier_l, brier_c = H.brier(y, pl), H.brier(y, pc)
    null_rate = float(np.mean(pc))
    alert = H.alert_budget_test(y, pl, null_rate)
    passed = bool(alert["binomial_p"] < 0.05 and brier_l < brier_c)
    out = {
        "n_test": int(len(y)),
        "effective_nonoverlapping": float(len(y) / 5),
        "joint_base_rate_pooled_test": float(y.mean()),
        "thresholds_per_fold": thresholds,
        "brier_logistic": brier_l, "brier_climatology": brier_c,
        "brier_below_climatology": bool(brier_l < brier_c),
        "auc_logistic": H.auc(y, pl),
        "alert": alert,
        "alert_p_lt_0.05": bool(alert["binomial_p"] < 0.05),
        "graduation_pass_unchanged_bar": passed,
        "per_fold": per_fold,
    }
    return out, pd.DataFrame(rows)


def run_all() -> dict:
    auctions = load_auctions()
    panel_full = H.load_panel()
    es = event_study(panel_full, auctions)
    panel, sample, _oas = truncated_sample()
    ridge, ridge_preds = run_ridge_leg(panel, sample, auctions)
    joint, joint_preds = run_joint_leg(panel, sample, auctions)
    preds = pd.concat([ridge_preds, joint_preds], ignore_index=True)
    results = {
        "experiment": "ML Stage C-1 (proposal T6): auction-day liquidity "
                      "conditioning + joint auction-widening event",
        "spec": "reports/ml_t6_auction_conditioning_spec_2026-10-27.md",
        "trials_ledger_rows": [7805, 7806, 7807],
        "panel_current_vintage": {
            "n_rows": int(len(panel_full)),
            "start": str(panel_full.index.min().date()),
            "end": str(panel_full.index.max().date()),
        },
        "modelling_sample": {
            "n_rows": int(len(sample)),
            "start": str(sample.index.min().date()),
            "end": str(sample.index.max().date()),
            "construction": "v1 common sample on the panel truncated at "
                            "2026-10-02 (spec §5)",
        },
        "leg_a_event_study": es,
        "leg_a_ridge": ridge,
        "leg_b_joint": joint,
    }
    return results, preds
