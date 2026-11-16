"""Tests for ML Target 1 v2 shadow mode (approved by Harry 2026-10-24).

Covers: shadow ID parsing through the broadened scorecard regex, the
v1 scorecard still parsing to the same entry count, the shadow batch
shape (8 entries, trained models only, v2-frozen32), the inputs
snapshot carrying exactly the 32 v2 features, and duplicate shadow
append refusal.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from engine import ml_harness as H
from engine import ml_harness_v2 as H2
from engine import ml_scorecard as SC
from engine import ml_shadow
from engine.ml_predict_v2 import build_scorecard_entries_v2

TRAINED_MODELS = {
    "ridge:v2-frozen32",
    "gbm:v2-frozen32",
    "logistic:v2-frozen32",
}


@pytest.fixture(scope="module")
def shadow_entries():
    panel = H.load_panel()
    oas = H.load_oas()
    X2 = H2.build_feature_matrix_v2(panel)
    asof = X2.index[X2.notna().all(axis=1)].max()
    return build_scorecard_entries_v2(panel, oas, asof=asof)


def test_shadow_id_parsing(shadow_entries):
    e = shadow_entries[0]
    assert e["prediction_id"].startswith("v2s-")
    text = "# header\n\n" + SC.format_entry(e) + "\n"
    parsed = SC.parse_entries(text)
    assert len(parsed) == 1
    p = parsed[0]
    assert p["prediction_id"] == e["prediction_id"]
    assert p["target_key"] == "T1a1"
    # asof must come from the inputs_snapshot, correctly, despite the prefix.
    assert p["asof"] == e["inputs_snapshot"]["asof"]
    assert p["outcome"] == "pending"


def test_v1_scorecard_still_parses_same_count():
    # Read-only parse of the real KB v1 scorecard: the broadened regex
    # must parse exactly the entries whose headings the original
    # v1-only pattern would have counted (96 entries at implementation
    # time, 2026-10-24).
    text = Path(SC.SCORECARD_PATH).read_text()
    parsed = SC.parse_entries(text)
    v1_headings = re.findall(r"^## \d{4}-\d{2}-\d{2}-\S+\s*$", text, flags=re.M)
    assert len(parsed) == len(v1_headings)
    assert all(not p["prediction_id"].startswith("v2s-") for p in parsed)


def test_build_entries_eight_trained_only(shadow_entries):
    assert len(shadow_entries) == 8
    assert {e["model_version"] for e in shadow_entries} == TRAINED_MODELS
    assert all(
        e["model_version"].endswith(H2.MODEL_VERSION_SUFFIX_V2)
        for e in shadow_entries
    )
    assert H2.MODEL_VERSION_SUFFIX_V2 == "v2-frozen32"
    assert not any("baseline" in e["model_version"] for e in shadow_entries)
    # Order: T1a1 ridge+GBM, T1a5 ridge+GBM, T1b_HY logistic+GBM,
    # T1b_IG logistic+GBM, IDs v2s-<date>-<Target>-01..08.
    expected = [
        ("T1a1", "ridge:v2-frozen32"),
        ("T1a1", "gbm:v2-frozen32"),
        ("T1a5", "ridge:v2-frozen32"),
        ("T1a5", "gbm:v2-frozen32"),
        ("T1b_HY", "logistic:v2-frozen32"),
        ("T1b_HY", "gbm:v2-frozen32"),
        ("T1b_IG", "logistic:v2-frozen32"),
        ("T1b_IG", "gbm:v2-frozen32"),
    ]
    for i, (e, (target, model)) in enumerate(zip(shadow_entries, expected), start=1):
        assert e["target"].split(":")[0] == target
        assert e["model_version"] == model
        asof = e["inputs_snapshot"]["asof"]
        assert e["prediction_id"] == f"v2s-{asof}-{target}-{i:02d}"


def test_snapshot_features_exactly_v2_32(shadow_entries):
    assert len(H2.FEATURE_NAMES_V2) == 32
    for e in shadow_entries:
        feats = e["inputs_snapshot"]["features"]
        assert list(feats.keys()) == list(H2.FEATURE_NAMES_V2)
        assert len(feats) == 32
        assert "refunding_week" not in feats
        assert "illiq_t" not in feats


def test_duplicate_append_refused(tmp_path, shadow_entries):
    path = tmp_path / "prediction_scorecard_v2_shadow.md"
    ml_shadow.ensure_shadow_file(path)
    assert SC.append_entries(shadow_entries, path=path) == 8
    with pytest.raises(ValueError):
        SC.append_entries(shadow_entries, path=path)
    # The refused append wrote nothing extra.
    parsed = SC.parse_entries(path.read_text())
    assert len(parsed) == 8
    assert all(p["prediction_id"].startswith("v2s-") for p in parsed)
