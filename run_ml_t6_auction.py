"""Runner for ML Stage C-1 (proposal T6). Frozen spec:
reports/ml_t6_auction_conditioning_spec_2026-10-27.md.
Trials booked in the ledger (rows 7805-7807) before this runner was run.
Writes the results JSON and the predictions parquet; prints a summary.
The human-readable report is written separately from these outputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from engine import ml_t6_auction as T6

ROOT = Path(__file__).resolve().parent
RESULTS_JSON = ROOT / "reports" / "ml_t6_auction_conditioning_results_2026-10-27.json"
PREDS = ROOT / "data" / "ml" / "wf_predictions_t6_2026-10-27.parquet"


def main() -> None:
    np.random.seed(13)
    results, preds = T6.run_all()
    mc = results["leg_a_ridge"]["machinery_check"]
    if not mc["passed"]:
        raise SystemExit(
            f"MACHINERY FAILURE: base ridge re-run MAE {mc['rerun_mae']:.6e} "
            f"vs record {mc['record_mae']:.6e} (rel. diff. "
            f"{mc['relative_difference']:.2e} > {mc['tolerance']:.0e}); "
            "no result is reported from this run.")
    RESULTS_JSON.write_text(json.dumps(results, indent=2) + "\n")
    preds.to_parquet(PREDS)
    es = results["leg_a_event_study"]
    print(f"Event study: evaluated n={es['evaluated_n']} "
          f"({es['evaluated_start']}..{es['evaluated_end']}); "
          f"baseline n={es['baseline']['n']}, hit rate "
          f"{es['baseline']['hit_rate']:.4f}, mean illiq "
          f"{es['baseline']['mean_illiq']:.3e}")
    for name, w in es["windows"].items():
        print(f"  {name}: n={w['n']} mean={w['mean_illiq']:.3e} "
              f"ratio={w['mean_ratio_vs_baseline']:.4f} hits={w['hits']} "
              f"hit_rate={w['hit_rate']:.4f} p={w['binomial_p']:.4f} "
              f"-> {'PASS' if w['pass_p_lt_0.05'] else 'FAIL'}")
    wk = es["refunding_week_level_secondary"]
    print(f"  week-level refunding: n_weeks={wk['n_refunding_weeks']} "
          f"hits={wk['hits']} p={wk['binomial_p']:.4f} "
          f"-> {'PASS' if wk['pass_p_lt_0.05'] else 'FAIL'}")
    r = results["leg_a_ridge"]
    print(f"Ridge T1a1 (n_test={r['n_test']}): base MAE {r['mae_base_ridge']:.6e} "
          f"(ratio vs persistence {r['mae_ratio_base_vs_persistence']:.4f}); "
          f"augmented MAE {r['mae_augmented_ridge']:.6e} "
          f"(ratio vs persistence {r['mae_ratio_augmented_vs_persistence']:.4f}; "
          f"vs base {r['mae_ratio_augmented_vs_base']:.4f})")
    j = results["leg_b_joint"]
    print(f"Joint leg (b) (n_test={j['n_test']}, base rate "
          f"{j['joint_base_rate_pooled_test']:.4f}): logistic Brier "
          f"{j['brier_logistic']:.4f} vs climatology {j['brier_climatology']:.4f}; "
          f"alert hits {j['alert']['alert_hits']}/{j['alert']['n_alert']}, "
          f"p={j['alert']['binomial_p']:.4f} -> "
          f"{'PASS' if j['graduation_pass_unchanged_bar'] else 'FAIL'} "
          "the unchanged bar")
    print(f"Wrote {RESULTS_JSON} and {PREDS} ({len(preds)} prediction rows)")


if __name__ == "__main__":
    main()
