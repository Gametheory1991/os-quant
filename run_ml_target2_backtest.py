"""Runner for ML Stage C — Target 2 backtest phase.

Frozen spec: reports/ml_target2_experiment_spec_2026-10-27.md including
Amendment A1. Trials 7820-7827 were booked in
~/workspace/goals/quant-research-backtest-engine/knowledge/trials_ledger.md
BEFORE this runner was first executed. Offline evaluation only; no
forward issuance is wired.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine import ml_harness as H
from engine import ml_target2_backtest as B

ROOT = Path(__file__).resolve().parent


def main() -> None:
    np.random.seed(H.SEED)
    results: dict = {
        "experiment": "ML Stage C — Target 2: Treasury auction bid-to-cover prediction (backtest phase)",
        "spec": "reports/ml_target2_experiment_spec_2026-10-27.md (including Amendment A1)",
        "dataset_card": "reports/ml_target2_dataset_card_2026-10-27.md (as amended)",
        "seed": H.SEED,
        "ledger_rows": {"T2a_coupon_ridge": 7820, "T2a_coupon_gbm": 7821,
                        "T2a_bills_ridge": 7822, "T2b_coupon_ridge": 7823,
                        "T2b_coupon_gbm": 7824, "T2b_bills_ridge": 7825,
                        "T2c_coupon_logistic": 7826, "T2c_bills_logistic": 7827},
        "ledger_family_after": {"family": "ML model configurations", "count": 56},
        "ledger_program_total_after": 7827,
        "panels": {},
    }
    all_preds = []
    for kind in ("coupon", "bills"):
        sample, folds, preds, fold_meta, hypers = B.run_panel(kind)
        exp = B.EXPECTED[kind]
        results["panels"][kind] = {
            "sample": {"n_labelled": int(len(sample)),
                       "start": str(sample["auction_date"].min().date()),
                       "end": str(sample["auction_date"].max().date()),
                       "weak_events_in_sample": int(sample["weak"].sum())},
            "n_folds": len(folds),
            "pooled_test": exp["pooled_test"],
            "pooled_test_weak": exp["pooled_weak"],
            "folds": fold_meta,
            "hyperparams_by_fold": hypers,
            "T2a": B.regression_metrics(preds, "t2a"),
            "T2b": B.regression_metrics(preds, "t2b"),
            "T2c": B.classification_metrics(preds),
        }
        all_preds.append(preds)
        print(f"{kind}: sample {len(sample)}, {len(folds)} folds, "
              f"{len(preds)} pooled test rows")
        for st in ("T2a", "T2b"):
            m = results["panels"][kind][st]
            print(f"  {st} baseline MAE {m['baseline']['mae']:.6f}")
            for name, e in m["models"].items():
                print(f"    {name:6s} MAE {e['mae']:.6f} ratio {e['mae_ratio_vs_baseline']:.4f} "
                      f"folds won {e['folds_won']}/{e['n_folds']} — {e['verdict']}")
        t = results["panels"][kind]["T2c"]
        print(f"  T2c logistic Brier {t['brier_logistic']:.6f} vs climatology "
              f"{t['brier_climatology']:.6f}; AUC {t['auc_logistic']}; alert "
              f"{t['alert']['alert_hits']}/{t['alert']['n_alert']} p={t['alert']['binomial_p']:.4f} "
              f"— {t['verdict']} (failed legs: {t['failed_legs']})")
    preds_all = pd.concat(all_preds, ignore_index=True)
    out_parquet = ROOT / "data/ml/wf_predictions_target2_2026-10-27.parquet"
    preds_all.assign(auction_date=preds_all["auction_date"].dt.strftime("%Y-%m-%d")).to_parquet(
        out_parquet, index=False)
    out_json = ROOT / "reports/ml_target2_results_2026-10-27.json"
    out_json.write_text(json.dumps(results, indent=1, default=str))
    print(f"wrote {out_parquet} ({len(preds_all)} rows)")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
