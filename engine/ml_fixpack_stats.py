"""Statistics helpers for the ML v2 fix-pack settling tests.

Pre-registered spec: reports/ml_v2_fixpack_tests_spec_2026-10-24.md.
Pure functions only — declustering of alert days and circular block
bootstrap machinery for Tests (i) and (ii). No model fitting lives
here; Tests (iii) and (iv) reuse the frozen v2 runner's machinery.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def decluster_alerts(positions: np.ndarray, y: np.ndarray,
                     blackout: int = 5) -> dict:
    """Greedy declustering of alert events in chronological order.

    ``positions`` are the events' full-panel trading-day positions
    (strictly increasing, same order as ``y``). The first event is
    retained; each subsequent event is retained iff its position is
    >= ``blackout`` greater than the last retained event's position,
    so retained 5-day label windows (t, t+5] are disjoint.

    Returns retained indices (into the input arrays), the retained
    positions, hit count (sum of y over retained), and hit rate.
    """
    positions = np.asarray(positions)
    y = np.asarray(y)
    assert len(positions) == len(y)
    assert (np.diff(positions) > 0).all(), "positions must be increasing"
    kept: list[int] = []
    last_pos = None
    for i, p in enumerate(positions):
        if last_pos is None or p - last_pos >= blackout:
            kept.append(i)
            last_pos = p
    kept_idx = np.array(kept, dtype=int)
    hits = int(y[kept_idx].sum()) if len(kept_idx) else 0
    return {
        "kept_idx": kept_idx,
        "kept_positions": positions[kept_idx],
        "n_events": int(len(kept_idx)),
        "hits": hits,
        "hit_rate": float(hits / len(kept_idx)) if len(kept_idx) else float("nan"),
    }


def binomial_greater_p(k: int, n: int, null_rate: float) -> float:
    """One-sided binomial p, identical convention to the harness
    alert test (scipy binomtest, alternative='greater')."""
    return float(stats.binomtest(int(k), int(n), float(null_rate),
                                 alternative="greater").pvalue)


def alert_selection(y: np.ndarray, p: np.ndarray,
                    budget: float = 0.10) -> dict:
    """Top-decile alert selection, identical rule to
    ml_harness.alert_budget_test (stable descending sort, ties keep
    input order) but returning the selected indices as well."""
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    n_alert = int(np.ceil(budget * n))
    order = np.argsort(-p, kind="stable")
    alert_idx = order[:n_alert]
    hits = int(y[alert_idx].sum())
    return {
        "alert_idx": alert_idx,
        "n_alert": n_alert,
        "hits": hits,
        "hit_rate": float(hits / n_alert) if n_alert else float("nan"),
    }


def circular_block_indices(n: int, block: int,
                           rng: np.random.Generator) -> np.ndarray:
    """One circular block-bootstrap index draw of length n: block
    starts uniform on 0..n-1, blocks wrap around the end, the
    concatenation is truncated to n (the dashboard screen's frozen
    convention, spec §2)."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    idx = np.concatenate([(s + np.arange(block)) % n for s in starts])
    return idx[:n]


def block_bootstrap(n: int, block: int, n_resamples: int,
                    seed: int = 13) -> np.ndarray:
    """Stack of resample index arrays, shape (n_resamples, n)."""
    rng = np.random.default_rng(seed)
    return np.stack([circular_block_indices(n, block, rng)
                     for _ in range(n_resamples)])


def paired_brier_diffs(y: np.ndarray, p_model: np.ndarray,
                       p_clim: np.ndarray) -> np.ndarray:
    """Per-day paired Brier difference (model minus climatology);
    negative means the model was better that day."""
    y = np.asarray(y, dtype=float)
    return (np.asarray(p_model, dtype=float) - y) ** 2 - (
        np.asarray(p_clim, dtype=float) - y) ** 2


def percentile_interval(values: np.ndarray,
                        lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(np.percentile(values, lo)), float(np.percentile(values, hi))
