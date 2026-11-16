"""Phase 4 machinery (build + dry-run ONLY): retrain the full frozen
ladder on all labelled sample rows and produce scorecard entries as of
the last panel date, in the knowledge_agent_design.md §3c schema.
Live issuance is gated on the coordinator's verify gate — this module
never writes a scorecard.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from engine import ml_harness as H
from engine import ml_models as M

MODEL_VERSION_SUFFIX = "v1-frozen14"
TARGET_DESCR = {
    "T1a1": "T1a1: illiq_agg at t+1",
    "T1a5": "T1a5: mean of illiq_agg over t+1..t+5",
    "T1b_HY": "T1b_HY: HY OAS widens by >= threshold over t+1..t+5",
    "T1b_IG": "T1b_IG: IG OAS widens by >= threshold over t+1..t+5",
}


def _asof_features(panel: pd.DataFrame, asof) -> pd.DataFrame:
    X = H.build_feature_matrix(panel)
    row = X.loc[[asof]]
    assert row[H.FEATURE_NAMES].notna().all(axis=1).iloc[0], f"features not complete at {asof}"
    return row


def build_scorecard_entries(panel=None, oas=None, asof=None) -> list[dict]:
    panel = panel if panel is not None else H.load_panel()
    oas = oas if oas is not None else H.load_oas()
    sample = H.build_sample(panel, oas)
    asof = asof or panel.index.max()
    X_all = sample[H.FEATURE_NAMES]
    X_asof = _asof_features(panel, asof)
    labels = H.build_labels(panel, oas)
    asof_lab = labels.loc[asof]
    vintage = {
        "feature_store_through": str(panel.index.max().date()),
        "feature_store_rows": int(len(panel)),
        "oas_through": str(max(s.index.max() for s in oas.values()).date()),
        "sample_rows": int(len(sample)),
        "sample_through": str(sample.index.max().date()),
    }
    snapshot = {f: float(X_asof[f].iloc[0]) for f in H.FEATURE_NAMES}
    snapshot["illiq_t"] = float(asof_lab["illiq_t"])
    issued_at = datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")

    entries: list[dict] = []
    seq = 0

    def add(target, model_version, prediction):
        nonlocal seq
        seq += 1
        entries.append(
            dict(
                prediction_id=f"{str(asof.date())}-{target}-{seq:02d}",
                issued_at=issued_at,
                target=TARGET_DESCR[target],
                horizon="1 trading day" if H.HORIZON[target] == 1 else "5 trading days",
                model_version=model_version,
                inputs_snapshot={"asof": str(asof.date()), "features": dict(snapshot),
                                 "data_vintage": dict(vintage)},
                prediction=prediction,
                outcome="pending",
                error="pending",
            )
        )

    # --- T1a regression ladder, trained on all labelled sample rows ---
    for target in ("T1a1", "T1a5"):
        y = sample[target].to_numpy(dtype=float)
        if target == "T1a1":
            add(target, "baseline:persistence", float(asof_lab["illiq_t"]))
        else:
            add(target, "baseline:persistence", float(asof_lab["illiq_trail5_mean"]))
        add(target, "baseline:climatology", float(np.mean(y)))
        p, hp = M.ridge_fit_predict(X_all, y, X_asof)
        add(target, f"ridge:{MODEL_VERSION_SUFFIX}", {"point": float(p[0]), **hp})
        p, _ = M.gbm_reg_fit_predict(X_all, y, X_asof)
        add(target, f"gbm:{MODEL_VERSION_SUFFIX}", {"point": float(p[0])})

    # --- T1b classification ladder per series ---
    event_all = sample["event_topdecile"].to_numpy(dtype=float)
    event_asof = float(asof_lab["event_topdecile"])
    for target, widen_col in (("T1b_HY", "oas_hy_widen5"), ("T1b_IG", "oas_ig_widen5")):
        thr = float(np.quantile(sample[widen_col].to_numpy(), 0.80))
        y = (sample[widen_col].to_numpy() >= thr).astype(int)
        p, _ = M.climatology_clf_predict(y, 1)
        add(target, "baseline:climatology", {"probability": float(p[0]), "threshold_pp": thr})
        p, _ = M.event_rule_predict(
            event_all, y, np.array([event_asof])
        )
        add(target, "baseline:event_rule", {"probability": float(p[0]), "threshold_pp": thr})
        p, hp = M.logistic_fit_predict(X_all, y, X_asof)
        add(target, f"logistic:{MODEL_VERSION_SUFFIX}",
            {"probability": float(p[0]), "threshold_pp": thr, **hp})
        p, _ = M.gbm_clf_fit_predict(X_all, y, X_asof)
        add(target, f"gbm:{MODEL_VERSION_SUFFIX}", {"probability": float(p[0]), "threshold_pp": thr})

    return entries
