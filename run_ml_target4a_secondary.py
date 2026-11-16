"""Driver for ML Target 4a secondary panels (Stage C-2).

Pre-registered spec: reports/ml_target4a_secondary_spec_2026-10-27.md —
written 2026-10-27 BEFORE any secondary-panel fitting. Order of
operations is part of the spec: per panel, build sample -> integrity
gate (any FAIL stops that panel before fitting) -> walk-forward ->
metrics -> frozen graduation verdicts. Panels are never pooled.
Offline evaluation only.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from engine import ml_harness as H
from engine import ml_t4a_secondary as S

ROOT = Path(__file__).resolve().parent


def main() -> None:
    all_preds, results = [], {"experiment": (
        "ML Target 4a secondary panels (Stage C-2) — CL=F / ZC=F / ZS=F "
        "5-trading-day log returns"),
        "spec": "reports/ml_target4a_secondary_spec_2026-10-27.md "
                "(pre-registered 2026-10-27, before fitting)",
        "panels": {}}
    for panel in ("cl", "zc", "zs"):
        df = S.load_panel(panel)
        sample = S.build_sample(panel, df)
        folds = H.make_folds(sample, horizon=S.HORIZON)
        gate = S.run_integrity_gate(panel, df, sample, folds)
        if not gate["passed"]:
            results["panels"][panel] = {"integrity_gate": gate,
                                       "status": "STOPPED_AT_GATE"}
            print(f"[{panel}] INTEGRITY GATE FAILED — no fitting")
            continue
        preds, fold_meta, alphas = S.run_walk_forward(
            sample, S.FEATURES[panel], folds)
        preds.insert(1, "panel", panel)
        all_preds.append(preds)
        metrics = S.model_metrics(preds, panel)
        verdicts = S.graduation_verdicts(metrics, panel)
        results["panels"][panel] = {
            "sample": {"n": int(len(sample)),
                       "start": str(sample.index[0].date()),
                       "end": str(sample.index[-1].date()),
                       "n_features": len(S.FEATURES[panel]),
                       "features": S.FEATURES[panel]},
            "folds": {"n_folds": len(folds),
                      "total_test_rows": int(sum(len(f.test_idx)
                                                 for f in folds)),
                      "folds": fold_meta},
            "integrity_gate": gate,
            "ridge_alphas_by_fold": alphas,
            "metrics_pct": metrics,
            "verdicts": verdicts,
        }
        print(f"[{panel}] sample {len(sample)} rows, {len(folds)} folds, "
              f"gate PASS; verdicts: "
              + ", ".join(f"{m}={verdicts[m]['verdict']}"
                          for m in S.LEARNED))
    if all_preds:
        out = pd.concat(all_preds, ignore_index=True)
        out.to_parquet(
            ROOT / "data/t4a_weather_futures"
                   "/wf_predictions_t4a_secondary.parquet", index=False)
    (ROOT / "reports/ml_target4a_secondary_results.json").write_text(
        json.dumps(results, indent=2, default=str))
    print("wrote reports/ml_target4a_secondary_results.json")


if __name__ == "__main__":
    main()
