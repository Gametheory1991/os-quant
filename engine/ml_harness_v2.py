"""ML Target 1 v2 — expanded-feature harness (offline experiment only).

Pre-registered spec: reports/ml_target1_v2_experiment_spec.md (written
2026-10-24 before any v2 fit). This module defines ONLY the v2 feature
set and the intersection-sample builder. All protocol machinery (folds,
purge, embargo, thresholds, metrics, labels) is reused unmodified from
engine/ml_harness.py. v1 defaults are untouched: engine/ml_harness.py,
engine/ml_predict.py and run_ml_daily.py still use the 14 frozen v1
features and MODEL_VERSION_SUFFIX = "v1-frozen14".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine import ml_harness as H

# v2 additions: (model feature name, store column, transform) — all store
# columns verbatim, no transform. Rationales and the dropped-candidate
# audit are in the pre-registered spec §3.
FEATURE_DEFS_V2_ADDITIONS = [
    ("illiq_vel7_252", "illiq_vel7_252", None),
    ("illiq_vel30_252", "illiq_vel30_252", None),
    ("illiq_chg21", "illiq_chg21", None),
    ("adv_chg21", "adv_chg21", None),
    ("adv_vel7_252", "adv_vel7_252", None),
    ("adv_vel30_252", "adv_vel30_252", None),
    ("oas_ig", "oas_ig", None),
    ("oas_ig_chg5d", "oas_ig_chg5d", None),
    ("oas_ig_z252", "oas_ig_z252", None),
    ("dealer_net_chg4w", "dealer_net_chg4w", None),
    ("auction_amt", "auction_amt", None),
    ("btc_last", "btc_last", None),
    ("ret_shy", "ret_shy", None),
    ("ret_ief", "ret_ief", None),
    ("ret_tlt", "ret_tlt", None),
    ("vix_regime_high", "vix_regime_high", None),
    ("corp_adv_lag_z12m", "corp_adv_lag_z12m", None),
    ("ticket_agg", "ticket_agg", None),
]

FEATURE_DEFS_V2 = list(H.FEATURE_DEFS) + FEATURE_DEFS_V2_ADDITIONS
FEATURE_NAMES_V2 = [d[0] for d in FEATURE_DEFS_V2]
MODEL_VERSION_SUFFIX_V2 = f"v2-frozen{len(FEATURE_DEFS_V2)}"

assert len(FEATURE_DEFS_V2) == 32, len(FEATURE_DEFS_V2)
assert len(set(FEATURE_NAMES_V2)) == 32
# oas_hy is a v1 feature; it must not be duplicated by the additions.
assert "oas_hy" not in [d[0] for d in FEATURE_DEFS_V2_ADDITIONS]


def build_feature_matrix_v2(panel: pd.DataFrame) -> pd.DataFrame:
    """The 32 v2 features, indexed like the panel (store columns at the
    same date; log only for log_illiq_agg, exactly as v1)."""
    X = pd.DataFrame(index=panel.index)
    for name, col, tf in FEATURE_DEFS_V2:
        v = panel[col].astype(float)
        X[name] = np.log(v) if tf == "log" else v
    return X


def build_sample_v2(panel: pd.DataFrame, oas: dict[str, pd.Series]) -> pd.DataFrame:
    """Intersection (comparison) sample: the v1 common sample (all 14 v1
    features non-NaN, all labels computable, via H.build_sample) further
    restricted to rows where every v2 addition is also non-NaN. Both v1
    and v2 are evaluated on exactly this sample, so folds are identical.

    full_pos (position in the full panel) is preserved from the v1 sample
    so purge/embargo distances in H.make_folds are unchanged; sample_pos
    is re-numbered over the filtered sample.
    """
    sample = H.build_sample(panel, oas)
    X2 = build_feature_matrix_v2(panel)
    add_names = [d[0] for d in FEATURE_DEFS_V2_ADDITIONS]
    extra = X2.loc[sample.index, add_names]
    keep = extra.notna().all(axis=1).to_numpy()
    out = sample.loc[keep].copy()
    for name in add_names:
        out[name] = extra.loc[out.index, name].to_numpy(dtype=float)
    out["sample_pos"] = np.arange(len(out))
    return out
