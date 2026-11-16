"""Walk-forward harness integrity tests (frozen spec:
reports/ml_target1_experiment_spec.md).

1. A synthetic series with a huge spike at the end: no training fold may
   contain rows whose label window touches its test block, and the
   embargo gap must hold for every fold (make_folds asserts internally;
   these tests re-verify from the outside).
2. Label construction uses only future rows for the label and never
   for features.
3. The per-fold OAS threshold is computed from training rows only.
"""
import numpy as np
import pandas as pd
import pytest

from engine import ml_harness as H


def _synthetic_panel(n=900, spike=True):
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2023-01-02", periods=n)
    illiq = np.abs(rng.normal(3e-5, 1e-5, n))
    if n > 700:  # holiday-stub style gaps, as in the real panel
        illiq[400] = np.nan
        illiq[650] = np.nan
    if spike:
        illiq[-10:] *= 50.0  # huge spike at the very end
    panel = pd.DataFrame(index=idx)
    panel["illiq_agg"] = illiq
    # Features: all non-NaN so the sample is the full panel minus label tail.
    for name, col, _tf in H.FEATURE_DEFS:
        if col not in panel.columns:
            panel[col] = rng.normal(size=n)
    panel["illiq_agg_z252"] = rng.normal(size=n)
    oas_idx = idx  # OAS observed on every panel date
    base = np.linspace(3.0, 3.5, n)
    oas = {
        "HY": pd.Series(base + rng.normal(0, 0.05, n), index=oas_idx),
        "IG": pd.Series(base * 0.3 + rng.normal(0, 0.02, n), index=oas_idx),
    }
    return panel, oas


def test_folds_purge_and_embargo_with_end_spike():
    panel, oas = _synthetic_panel()
    sample = H.build_sample(panel, oas)
    assert len(sample) > H.INITIAL_TRAIN + 2 * H.TEST_BLOCK
    for horizon in (1, 5):
        folds = H.make_folds(sample, horizon=horizon)
        assert len(folds) >= 2
        full_pos = sample["full_pos"].to_numpy()
        tested = []
        for f in folds:
            tr_full = full_pos[f.train_idx]
            te_full = full_pos[f.test_idx]
            # No training label window (t, t+h] touches the test block start.
            assert (tr_full + horizon < te_full.min()).all()
            # Embargo: >=5 trading days strictly between last train and first test.
            assert te_full.min() - tr_full.max() > H.EMBARGO_DAYS
            # Train strictly before test; disjoint.
            assert sample.index[f.train_idx].max() < sample.index[f.test_idx].min()
            assert len(np.intersect1d(f.train_idx, f.test_idx)) == 0
            tested.extend(f.test_idx.tolist())
        # Test blocks are contiguous, ordered, non-overlapping, after initial train.
        assert tested == sorted(tested)
        assert min(tested) == H.INITIAL_TRAIN
        # The end spike lands only in test labels / late folds, never as a
        # training feature row in an earlier fold than its date allows:
        spike_pos = full_pos.max()
        for f in folds:
            if spike_pos in full_pos[f.train_idx]:
                assert sample.index[f.test_idx].min() > panel.index[-1] - pd.Timedelta(days=40)


def test_labels_use_only_future_rows_and_features_untouched():
    panel, oas = _synthetic_panel(n=120, spike=False)
    X_before = H.build_feature_matrix(panel)
    lab = H.build_labels(panel, oas)
    illiq = panel["illiq_agg"].to_numpy()
    # T1a1 is exactly the next row; T1a5 is exactly the mean of the next 5.
    assert lab["T1a1"].iloc[10] == pytest.approx(illiq[11])
    assert lab["T1a5"].iloc[10] == pytest.approx(illiq[11:16].mean())
    # Labels are not computable at the tail (no future rows) -> NaN, never
    # filled from the past.
    assert np.isnan(lab["T1a1"].iloc[-1])
    assert lab["T1a5"].iloc[-5:].isna().all()
    # OAS widening uses the OAS series' own future steps only.
    w = H.oas_widening(panel.index, oas["HY"], 5)
    assert w.iloc[3] == pytest.approx(oas["HY"].iloc[8] - oas["HY"].iloc[3])
    # Building labels did not change the feature matrix (no leakage path
    # from labels into features).
    pd.testing.assert_frame_equal(X_before, H.build_feature_matrix(panel))
    # Truncation invariance of labels: labels computed on a truncated panel
    # match the full-panel labels wherever the window fits — labels depend
    # only on their own future window, never on full-sample statistics.
    cut = 60
    lab_trunc = H.build_labels(panel.iloc[: cut + 6], oas)
    pd.testing.assert_series_equal(
        lab["T1a5"].iloc[:cut], lab_trunc["T1a5"].iloc[:cut]
    )


def test_fold_threshold_uses_training_rows_only():
    panel, oas = _synthetic_panel()
    sample = H.build_sample(panel, oas)
    folds = H.make_folds(sample, horizon=5)
    f = folds[0]
    thr = H.fold_threshold(sample, f.train_idx, "oas_hy_widen5")
    expected = np.quantile(sample["oas_hy_widen5"].to_numpy()[f.train_idx], 0.80)
    assert thr == pytest.approx(expected)
    # Poison the test rows' widenings: the threshold must not move.
    poisoned = sample.copy()
    poisoned.loc[poisoned.index[f.test_idx], "oas_hy_widen5"] = 1e6
    assert H.fold_threshold(poisoned, f.train_idx, "oas_hy_widen5") == pytest.approx(thr)
