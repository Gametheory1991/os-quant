"""Standing feature-block ablation framework (Stage B-3, proposal M2).

Frozen first-cycle specification:
reports/ml_feature_block_ablation_spec_2026-10-27.md. This module is
the STANDING component that specification freezes: future model
versions call it with (feature set, block definitions, protocol) and
receive the same cell structure, walk-forward prediction machinery,
and verdict mapping. It is deliberately protocol-agnostic — samples,
folds, labels, model constructions, and leg definitions belong to
each cycle's own specification and are passed in, never hard-coded
here.

Ablations run through this component are DIAGNOSTIC ONLY: they can
demote a claim or block a promotion; they can never graduate a model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

VERDICTS = ("sustains", "weakens", "overturns")


# --- Block definitions (data, validated) ---------------------------------------
@dataclass(frozen=True)
class FeatureBlock:
    """A named functional feature block: an exact list of feature
    names plus a one-line definition (the cycle spec carries the full
    definition; this is its machine-readable form)."""

    name: str
    features: tuple[str, ...]
    definition: str = ""

    def __post_init__(self):
        if not self.name:
            raise ValueError("block name must be non-empty")
        if not self.features:
            raise ValueError(f"block {self.name!r}: feature list must be non-empty")
        if len(set(self.features)) != len(self.features):
            raise ValueError(f"block {self.name!r}: duplicate feature in block")


def validate_blocks(all_features: list[str], blocks: list[FeatureBlock]) -> None:
    """Every block member must belong to the feature set; blocks must
    be pairwise disjoint; removing any single block must leave a
    non-empty feature set. Raises ValueError on any violation."""
    known = set(all_features)
    if len(known) != len(all_features):
        raise ValueError("feature set contains duplicates")
    seen: dict[str, str] = {}
    for b in blocks:
        unknown = [f for f in b.features if f not in known]
        if unknown:
            raise ValueError(
                f"block {b.name!r}: features not in the feature set: {unknown}"
            )
        for f in b.features:
            if f in seen:
                raise ValueError(
                    f"feature {f!r} appears in two blocks: {seen[f]!r} and {b.name!r}"
                )
            seen[f] = b.name
        if len(all_features) - len(b.features) <= 0:
            raise ValueError(
                f"block {b.name!r}: removing it would empty the feature set"
            )


def kept_features(all_features: list[str], block: FeatureBlock) -> list[str]:
    """The feature set with one block removed, original order kept."""
    removed = set(block.features)
    return [f for f in all_features if f not in removed]


# --- Cells ----------------------------------------------------------------------
@dataclass(frozen=True)
class AblationCell:
    """One (champion, block) pair of a cycle."""

    cell_id: str
    target: str
    champion: str
    block_name: str
    removed: tuple[str, ...]
    kept: tuple[str, ...]


def enumerate_cells(
    target: str,
    champion: str,
    all_features: list[str],
    blocks: list[FeatureBlock],
    cell_id_prefix: str = "ABL",
) -> list[AblationCell]:
    """One cell per block, in block order. Validates the blocks first."""
    validate_blocks(all_features, blocks)
    return [
        AblationCell(
            cell_id=f"{cell_id_prefix}-{i:02d}",
            target=target,
            champion=champion,
            block_name=b.name,
            removed=tuple(b.features),
            kept=tuple(kept_features(all_features, b)),
        )
        for i, b in enumerate(blocks, start=1)
    ]


# --- Verdict mapping (Stage A §A leg (iv) language; do not drift) ---------------
def verdict_from_legs(legs: dict[str, bool]) -> str:
    """sustains — every leg still passes; weakens — exactly some legs
    pass; overturns — no leg passes. The mapping is mechanical and is
    the fix-pack specification's pre-registered reading carried into
    the standing framework: both legs / one leg / neither."""
    if not legs:
        raise ValueError("legs must be a non-empty mapping")
    n_pass = sum(bool(v) for v in legs.values())
    if n_pass == len(legs):
        return "sustains"
    if n_pass == 0:
        return "overturns"
    return "weakens"


# --- Walk-forward prediction machinery -------------------------------------------
def walk_forward_predict(
    sample: pd.DataFrame,
    features: list[str],
    y: np.ndarray,
    folds,
    fit_predict,
) -> pd.DataFrame:
    """Out-of-sample predictions on the folds' test rows for one
    feature set. `fit_predict(Xtr, ytr, Xte)` returns the test
    predictions (point forecasts or class probabilities — the
    component does not interpret them). Folds are the harness's Fold
    objects (train_idx / test_idx are sample positions); fold
    integrity was asserted when the folds were built, by the harness,
    and is not re-decided here.

    Returns a DataFrame: date, fold, y_true, y_pred — one row per
    pooled test row, in fold order. Champion and ablated runs through
    this function on the same folds are on identical days by
    construction."""
    X = sample[features]
    rows = []
    for f in folds:
        Xtr, Xte = X.iloc[f.train_idx], X.iloc[f.test_idx]
        ytr, yte = y[f.train_idx], y[f.test_idx]
        p = np.asarray(fit_predict(Xtr, ytr, Xte), dtype=float)
        assert len(p) == len(f.test_idx), "fit_predict returned wrong length"
        for i, pos in enumerate(f.test_idx):
            rows.append(
                dict(
                    date=sample.index[pos],
                    fold=int(f.fold_id),
                    y_true=float(yte[i]),
                    y_pred=float(p[i]),
                )
            )
    return pd.DataFrame(rows)


@dataclass
class CycleResult:
    """One champion's ablation cycle: champion (full-feature)
    predictions plus one ablated prediction frame per cell."""

    target: str
    champion: str
    champion_preds: pd.DataFrame
    cells: list[AblationCell] = field(default_factory=list)
    ablated_preds: dict[str, pd.DataFrame] = field(default_factory=dict)


def run_ablation_cycle(
    sample: pd.DataFrame,
    all_features: list[str],
    y: np.ndarray,
    folds,
    fit_predict,
    blocks: list[FeatureBlock],
    target: str = "",
    champion: str = "",
    cell_id_prefix: str = "ABL",
) -> CycleResult:
    """Run the champion on its full feature set, then one ablated run
    per block, all on identical folds. Asserts that every ablated run
    covers exactly the champion's test days — if it does not, the
    comparison is not on identical days and the cycle must stop."""
    cells = enumerate_cells(target, champion, all_features, blocks, cell_id_prefix)
    champ = walk_forward_predict(sample, list(all_features), y, folds, fit_predict)
    out = CycleResult(target=target, champion=champion, champion_preds=champ)
    champ_days = champ["date"].to_numpy()
    for cell in cells:
        preds = walk_forward_predict(sample, list(cell.kept), y, folds, fit_predict)
        assert len(preds) == len(champ), (
            f"cell {cell.cell_id}: {len(preds)} test rows vs champion {len(champ)}"
        )
        assert (preds["date"].to_numpy() == champ_days).all(), (
            f"cell {cell.cell_id}: ablated test days differ from champion days"
        )
        assert (preds["y_true"].to_numpy() == champ["y_true"].to_numpy()).all(), (
            f"cell {cell.cell_id}: ablated labels differ from champion labels"
        )
        out.cells.append(cell)
        out.ablated_preds[cell.cell_id] = preds
    return out
