"""ML Stage C Target 2 — Phase 1 build check (NO fitting).

Loads data/ml/auctions.csv, builds both panels, labels, features, the
label-complete samples, and the frozen walk-forward folds, and asserts
the realised counts against the dataset card
(reports/ml_target2_dataset_card_2026-10-27.md) and the specification
as amended (Amendment A1, 2026-10-27, in
reports/ml_target2_experiment_spec_2026-10-27.md). It fits no model,
books no trial, and writes no results. The backtest phase is a
separate program under its own integrity gate.

Amendment A1 repair embodied here: the pooled test assignment is
asserted to be free of cross-fold duplication (every pooled count
below is a count of DISTINCT auctions), after the fold construction
was found to assign two coupon auctions to two test blocks each
(pooled coupon assignments 209 against 207 distinct auctions).
"""
from __future__ import annotations

import json

from engine import ml_target2_auction as T2

COUPON_FOLDS = dict(initial_train=150, test_block=40, embargo_days=7)
BILLS_FOLDS = dict(initial_train=400, test_block=100, embargo_days=7)

# Expected DISTINCT pooled test auctions per panel, as restated by
# Amendment A1: coupon 207 (was 209 assignments with 2 duplicated),
# bills 660 (unchanged; no duplication was present).
EXPECTED_POOLED_DISTINCT = {"coupon": 207, "bills": 660}
# Expected DISTINCT pooled test weak events: coupon 48 (was 49 with
# duplication), bills 158 (unchanged).
EXPECTED_POOLED_WEAK = {"coupon": 48, "bills": 158}


def main() -> None:
    auctions = T2.load_auctions()
    assert len(auctions) == 1621
    refunding = T2.refunding_mondays(auctions)
    assert len(refunding) == 15
    out = {"auctions": len(auctions), "refunding_weeks": len(refunding)}
    for kind, panel_fn, params, expected in (
        ("coupon", T2.coupon_panel, COUPON_FOLDS, 357),
        ("bills", T2.bills_panel, BILLS_FOLDS, 1060),
    ):
        panel = panel_fn(auctions)
        sample = T2.build_sample(panel, auctions, kind)
        assert len(sample) == expected, (kind, len(sample), expected)
        folds = T2.make_folds(sample, refunding=refunding, **params)
        # Amendment A1: cross-fold test uniqueness — the pooled test
        # assignment must contain no position twice, so the summed
        # fold sizes equal the number of distinct test auctions.
        distinct_test = T2.assert_cross_fold_test_uniqueness(folds)
        pooled_assignments = int(sum(len(f.test_idx) for f in folds))
        assert pooled_assignments == distinct_test, (
            kind, pooled_assignments, distinct_test)
        assert distinct_test == EXPECTED_POOLED_DISTINCT[kind], (
            kind, distinct_test, EXPECTED_POOLED_DISTINCT[kind])
        pooled_weak = int(sum(int(sample["weak"].iloc[f.test_idx].sum())
                              for f in folds))
        assert pooled_weak == EXPECTED_POOLED_WEAK[kind], (
            kind, pooled_weak, EXPECTED_POOLED_WEAK[kind])
        out[kind] = {
            "panel_rows": len(panel),
            "sample_rows": len(sample),
            "n_folds": len(folds),
            "pooled_test": distinct_test,
            "pooled_test_weak": pooled_weak,
            "folds": [f.meta for f in folds],
            "weak_events_in_sample": int(sample["weak"].sum()),
        }
    print(json.dumps(out, indent=1, default=str))
    print("BUILD CHECK PASSED — no model was fit.")


if __name__ == "__main__":
    main()
