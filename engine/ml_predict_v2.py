"""ML Target 1 v2 shadow issuance — trained models only.

Mirrors engine/ml_predict.py's build_scorecard_entries, but for the
v2-frozen32 feature set (engine/ml_harness_v2.py). Emits ONLY the 8
trained-model entries (no baselines — they are version-independent
and already issued under v1), with shadow prediction IDs prefixed
``v2s-``. This module never writes a scorecard; writing is owned by
engine/ml_shadow.py via engine/ml_scorecard.py.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from engine import ml_harness as H
from engine import ml_harness_v2 as H2
from engine import ml_models as M
from engine.ml_predict import TARGET_DESCR


def _asof_features_v2(panel: pd.DataFrame, asof) -> pd.DataFrame:
    X = H2.build_feature_matrix_v2(panel)
    row = X.loc[[asof]]
    assert row[H2.FEATURE_NAMES_V2].notna().all(axis=1).iloc[0], f"features not complete at {asof}"
    return row


def build_scorecard_entries_v2(panel=None, oas=None, asof=None) -> list[dict]:
    panel = panel if panel is not None else H.load_panel()
    oas = oas if oas is not None else H.load_oas()
    sample = H2.build_sample_v2(panel, oas)
    asof = asof or panel.index.max()
    X_all = sample[H2.FEATURE_NAMES_V2]
    X_asof = _asof_features_v2(panel, asof)
    vintage = {
        "feature_store_through": str(panel.index.max().date()),
        "feature_store_rows": int(len(panel)),
        "oas_through": str(max(s.index.max() for s in oas.values()).date()),
        "sample_rows": int(len(sample)),
        "sample_through": str(sample.index.max().date()),
    }
    snapshot = {f: float(X_asof[f].iloc[0]) for f in H2.FEATURE_NAMES_V2}
    issued_at = datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")

    entries: list[dict] = []
    seq = 0

    def add(target, model_version, prediction):
        nonlocal seq
        seq += 1
        entries.append(
            dict(
                prediction_id=f"v2s-{str(asof.date())}-{target}-{seq:02d}",
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

    # --- T1a regression: trained models only ---
    for target in ("T1a1", "T1a5"):
        y = sample[target].to_numpy(dtype=float)
        p, hp = M.ridge_fit_predict(X_all, y, X_asof)
        add(target, f"ridge:{H2.MODEL_VERSION_SUFFIX_V2}", {"point": float(p[0]), **hp})
        p, _ = M.gbm_reg_fit_predict(X_all, y, X_asof)
        add(target, f"gbm:{H2.MODEL_VERSION_SUFFIX_V2}", {"point": float(p[0])})

    # --- T1b classification: trained models only ---
    for target, widen_col in (("T1b_HY", "oas_hy_widen5"), ("T1b_IG", "oas_ig_widen5")):
        thr = float(np.quantile(sample[widen_col].to_numpy(), 0.80))
        y = (sample[widen_col].to_numpy() >= thr).astype(int)
        p, hp = M.logistic_fit_predict(X_all, y, X_asof)
        add(target, f"logistic:{H2.MODEL_VERSION_SUFFIX_V2}",
            {"probability": float(p[0]), "threshold_pp": thr, **hp})
        p, _ = M.gbm_clf_fit_predict(X_all, y, X_asof)
        add(target, f"gbm:{H2.MODEL_VERSION_SUFFIX_V2}", {"probability": float(p[0]), "threshold_pp": thr})

    return entries


# Alias matching the v1 module's function name, for callers that
# mirror the v1 call shape.
build_scorecard_entries = build_scorecard_entries_v2
