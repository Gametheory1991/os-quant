"""Tests for the standing feature-block ablation component
(engine/ml_ablation.py; spec reports/ml_feature_block_ablation_spec_2026-10-27.md)."""
import numpy as np
import pandas as pd
import pytest

from engine import ml_ablation as A
from engine import ml_harness as H


FEATURES = ["f_a", "f_b", "f_c", "f_d", "f_e"]
BLOCKS = [
    A.FeatureBlock("credit", ("f_b", "f_c"), "credit block"),
    A.FeatureBlock("lagtgt", ("f_a",), "lagged target block"),
]


def test_block_rejects_duplicates_and_empty():
    with pytest.raises(ValueError):
        A.FeatureBlock("x", ("f_a", "f_a"))
    with pytest.raises(ValueError):
        A.FeatureBlock("x", ())
    with pytest.raises(ValueError):
        A.FeatureBlock("", ("f_a",))


def test_validate_blocks_accepts_valid():
    A.validate_blocks(FEATURES, BLOCKS)


def test_validate_blocks_rejects_unknown_feature():
    with pytest.raises(ValueError, match="not in the feature set"):
        A.validate_blocks(FEATURES, [A.FeatureBlock("bad", ("f_zz",))])


def test_validate_blocks_rejects_overlapping_blocks():
    overlapping = [
        A.FeatureBlock("one", ("f_a", "f_b")),
        A.FeatureBlock("two", ("f_b", "f_c")),
    ]
    with pytest.raises(ValueError, match="two blocks"):
        A.validate_blocks(FEATURES, overlapping)


def test_validate_blocks_rejects_emptying_removal():
    with pytest.raises(ValueError, match="empty the feature set"):
        A.validate_blocks(["f_a"], [A.FeatureBlock("all", ("f_a",))])


def test_kept_features_preserves_order():
    kept = A.kept_features(FEATURES, BLOCKS[0])
    assert kept == ["f_a", "f_d", "f_e"]


def test_enumerate_cells_one_per_block():
    cells = A.enumerate_cells("T", "champ", FEATURES, BLOCKS, "B3")
    assert [c.cell_id for c in cells] == ["B3-01", "B3-02"]
    assert cells[0].removed == ("f_b", "f_c")
    assert cells[0].kept == ("f_a", "f_d", "f_e")
    assert cells[1].kept == ("f_b", "f_c", "f_d", "f_e")


@pytest.mark.parametrize(
    "legs,expected",
    [
        ({"i": True, "ii": True}, "sustains"),
        ({"i": True, "ii": False}, "weakens"),
        ({"i": False, "ii": True}, "weakens"),
        ({"i": False, "ii": False}, "overturns"),
        ({"only": True}, "sustains"),
        ({"only": False}, "overturns"),
    ],
)
def test_verdict_from_legs(legs, expected):
    assert A.verdict_from_legs(legs) == expected


def test_verdict_from_legs_rejects_empty():
    with pytest.raises(ValueError):
        A.verdict_from_legs({})


def _toy_sample(n=40):
    idx = pd.date_range("2024-01-01", periods=n, freq="B", name="date")
    rng = np.random.default_rng(0)
    sample = pd.DataFrame(
        {f: rng.normal(size=n) for f in FEATURES}, index=idx
    )
    sample["full_pos"] = np.arange(n)
    sample["sample_pos"] = np.arange(n)
    return sample


def test_walk_forward_predict_uses_only_given_features():
    sample = _toy_sample()
    y = np.arange(len(sample), dtype=float)
    folds = H.make_folds(sample, horizon=1, initial_train=20, test_block=10,
                         embargo_days=2)
    seen_columns = []

    def mean_fit_predict(Xtr, ytr, Xte):
        seen_columns.append(list(Xtr.columns))
        return np.full(len(Xte), float(np.mean(ytr)))

    preds = A.walk_forward_predict(sample, ["f_a", "f_d"], y, folds,
                                   mean_fit_predict)
    assert set(preds.columns) == {"date", "fold", "y_true", "y_pred"}
    assert all(cols == ["f_a", "f_d"] for cols in seen_columns)
    expected_n = sum(len(f.test_idx) for f in folds)
    assert len(preds) == expected_n
    # Predictions equal the training-fold mean of y for their fold.
    for fid, g in preds.groupby("fold"):
        f = folds[fid]
        assert np.allclose(g["y_pred"], np.mean(y[f.train_idx]))


def test_run_ablation_cycle_identical_days_and_labels():
    sample = _toy_sample()
    y = np.arange(len(sample), dtype=float)
    folds = H.make_folds(sample, horizon=1, initial_train=20, test_block=10,
                         embargo_days=2)

    def mean_fit_predict(Xtr, ytr, Xte):
        return np.full(len(Xte), float(np.mean(ytr)))

    res = A.run_ablation_cycle(sample, FEATURES, y, folds, mean_fit_predict,
                               BLOCKS, target="T", champion="champ",
                               cell_id_prefix="B3")
    assert len(res.cells) == 2
    for cell in res.cells:
        ap = res.ablated_preds[cell.cell_id]
        assert (ap["date"].to_numpy()
                == res.champion_preds["date"].to_numpy()).all()
        assert (ap["y_true"].to_numpy()
                == res.champion_preds["y_true"].to_numpy()).all()
