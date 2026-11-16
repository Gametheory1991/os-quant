"""ML Phase 1 — event study: Treasury illiquidity spikes -> 5d OAS widening.

Experiment (ml_engine_design.md §1, Phase 1; the microstructure memo's
first experiment and Target 1's evidence gate), fixed BEFORE the run:

PRIMARY event rule (point-in-time):
    Trading day t (a panel date in ``data/ml/feature_store.parquet``) is an
    event day iff ``illiq_agg_t >= Q90(illiq_agg over the trailing 252
    trading days ending at t)``, rolling window 252, ``min_periods=126``.
    The trailing window includes t itself and nothing after t; no
    full-sample quantiles are used anywhere in the primary rule.

Sample:
    Per OAS series (HY: FRED BAMLH0A0HYM2, ``data/ml/oas_hy.csv``; IG:
    BAMLC0A0CM, ``data/ml/oas_ig.csv``), a panel date t is OAS-eligible iff
    the series has a value at t AND at t+5, where t+5 is 5 positions later
    on that OAS series' own trading-day calendar (so forward windows may
    extend past the panel end 2026-10-02; the OAS CSVs run to 2026-10-12).
    The EVALUATED sample further requires the event flag to be
    classifiable at t: ``illiq_agg_t`` is not NaN and the trailing
    quantile is defined (>= 126 trailing observations). Days failing that
    are counted and reported, never silently treated as non-events.
    A sensitivity block recomputes the primary statistics on the full
    OAS-eligible sample with unclassifiable days treated as non-events.

Outcomes:
    dHY = OAS_HY(t+5) - OAS_HY(t), dIG likewise (percentage points).
    "Widening" = delta > 0.

Statistics (per series):
    n event days, event hits and hit rate, non-event hits and hit rate
    (the chance baseline), mean/median delta on event vs non-event days,
    one-sided binomial test P(X >= event hits | n_events,
    p = non-event hit rate). PASS iff binomial p < 0.05.

Robustness (secondary):
    (a) Declustered events: event days whose panel-trading-day positions
        differ by < 5 belong to one cluster; keep only each cluster's
        first day. Hit rate + binomial p recomputed against the primary
        non-event baseline.
    (b) Top-20 variant: the 20 largest ``illiq_agg`` days in the
        evaluated sample, same statistics vs the non-top-20 baseline.
        This variant ranks on the full sample and is therefore
        DESCRIPTIVE, not point-in-time.

Run:  .venv/bin/python -m engine.ml_event_study
Prints all statistics and writes reports/ml_phase1_results.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "data" / "ml"
FEATURE_STORE = ML_DIR / "feature_store.parquet"
REPORTS_DIR = ROOT / "reports"
RESULTS_JSON = REPORTS_DIR / "ml_phase1_results.json"

WINDOW = 252
MIN_PERIODS = 126
QUANTILE = 0.90
HORIZON = 5
ALPHA = 0.05
CLUSTER_GAP = 5  # event days < 5 trading days apart belong to one cluster
TOP_N = 20

SERIES = {
    "HY": {"file": "oas_hy.csv", "fred": "BAMLH0A0HYM2"},
    "IG": {"file": "oas_ig.csv", "fred": "BAMLC0A0CM"},
}


# ---------------------------------------------------------------------------
# Event rule (point-in-time)
# ---------------------------------------------------------------------------
def trailing_threshold(
    illiq: pd.Series,
    window: int = WINDOW,
    min_periods: int = MIN_PERIODS,
    quantile: float = QUANTILE,
) -> pd.Series:
    """Trailing quantile of ``illiq`` over ``window`` days ending at t.

    Uses only observations dated <= t (the window includes t). NaN until
    at least ``min_periods`` non-NaN observations are in the window.
    """
    return illiq.rolling(window, min_periods=min_periods).quantile(quantile)


def event_flags(
    illiq: pd.Series,
    window: int = WINDOW,
    min_periods: int = MIN_PERIODS,
    quantile: float = QUANTILE,
) -> pd.Series:
    """Boolean event flags (nullable dtype).

    True  -> illiq_t >= trailing threshold at t (event day)
    False -> illiq_t <  trailing threshold at t (non-event day)
    <NA>  -> not classifiable at t (illiq_t is NaN or the trailing
             quantile is undefined); such days must be excluded from
             evaluation, not counted as non-events.
    """
    thr = trailing_threshold(illiq, window, min_periods, quantile)
    flags = (illiq >= thr).astype("boolean")
    flags[illiq.isna() | thr.isna()] = pd.NA
    return flags


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def find_clusters(
    event_dates: pd.DatetimeIndex,
    trading_days: pd.DatetimeIndex,
    min_gap: int = CLUSTER_GAP,
) -> list[list[pd.Timestamp]]:
    """Group event dates into clusters.

    Consecutive event days whose positions in ``trading_days`` differ by
    < ``min_gap`` belong to the same cluster. Returns a list of clusters,
    each a list of event dates in chronological order.
    """
    if len(event_dates) == 0:
        return []
    pos = {d: i for i, d in enumerate(trading_days)}
    ordered = sorted(event_dates)
    clusters: list[list[pd.Timestamp]] = [[ordered[0]]]
    for prev, cur in zip(ordered[:-1], ordered[1:]):
        if pos[cur] - pos[prev] < min_gap:
            clusters[-1].append(cur)
        else:
            clusters.append([cur])
    return clusters


def decluster_events(
    event_dates: pd.DatetimeIndex,
    trading_days: pd.DatetimeIndex,
    min_gap: int = CLUSTER_GAP,
) -> pd.DatetimeIndex:
    """Keep only the first event day of each cluster."""
    clusters = find_clusters(event_dates, trading_days, min_gap)
    return pd.DatetimeIndex([c[0] for c in clusters])


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def binomial_p_ge(hits: int, n: int, p: float) -> float:
    """One-sided binomial tail P(X >= hits | n, p), exact, stdlib only."""
    if n == 0:
        return float("nan")
    if hits <= 0:
        return 1.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    return float(
        sum(
            math.comb(n, i) * p**i * (1.0 - p) ** (n - i)
            for i in range(hits, n + 1)
        )
    )


def summarize(selected: pd.Series, baseline: pd.Series) -> dict:
    """Hit-rate statistics for a selected day set vs a baseline day set.

    Both arguments are Series of 5-day OAS deltas (percentage points).
    """
    sel_hits = int((selected > 0).sum())
    base_hits = int((baseline > 0).sum())
    base_rate = base_hits / len(baseline) if len(baseline) else float("nan")
    p_val = binomial_p_ge(sel_hits, len(selected), base_rate)
    return {
        "n": int(len(selected)),
        "hits": sel_hits,
        "hit_rate": float(sel_hits / len(selected)) if len(selected) else None,
        "mean_delta_pp": float(selected.mean()) if len(selected) else None,
        "median_delta_pp": float(selected.median()) if len(selected) else None,
        "baseline_n": int(len(baseline)),
        "baseline_hits": base_hits,
        "baseline_hit_rate": float(base_rate),
        "baseline_mean_delta_pp": float(baseline.mean()),
        "baseline_median_delta_pp": float(baseline.median()),
        "binomial_p": p_val,
        "pass_p_lt_0.05": bool(p_val < ALPHA) if not np.isnan(p_val) else False,
    }


# ---------------------------------------------------------------------------
# Data loading / outcome construction
# ---------------------------------------------------------------------------
def load_oas(path: Path) -> pd.Series:
    """Load a FRED OAS CSV (date, value) as a clean sorted Series (percent)."""
    df = pd.read_csv(path, parse_dates=["observation_date"])
    value_col = [c for c in df.columns if c != "observation_date"][0]
    s = pd.Series(
        pd.to_numeric(df[value_col], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(df["observation_date"]),
        name=value_col,
    ).sort_index()
    s = s[~s.index.duplicated(keep="last")].dropna()
    return s


def forward_delta(oas: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """OAS(t+horizon) - OAS(t) on the OAS series' own trading-day calendar.

    Positional shift on the cleaned OAS series: a date only gets a delta
    if both t and the date ``horizon`` positions later exist in the
    series (which may extend past the panel end).
    """
    return oas.shift(-horizon) - oas


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def run() -> dict:
    panel = pd.read_parquet(FEATURE_STORE)
    illiq = panel["illiq_agg"]
    flags = event_flags(illiq)
    panel_pos = {d: i for i, d in enumerate(panel.index)}

    results: dict = {
        "experiment": "ML Phase 1 event study: illiq_agg top-decile (trailing, "
        "point-in-time) days -> 5-trading-day HY/IG OAS widening",
        "event_rule": {
            "window": WINDOW,
            "min_periods": MIN_PERIODS,
            "quantile": QUANTILE,
            "horizon_trading_days": HORIZON,
            "alpha": ALPHA,
        },
        "panel": {
            "start": str(panel.index.min().date()),
            "end": str(panel.index.max().date()),
            "n_days": int(len(panel)),
        },
        "series": {},
    }

    for name, cfg in SERIES.items():
        oas = load_oas(ML_DIR / cfg["file"])
        delta = forward_delta(oas)

        # OAS-eligible panel dates: OAS value at t and at t+5 both exist.
        eligible_dates = panel.index[delta.reindex(panel.index).notna()]
        eligible = pd.DataFrame(
            {
                "delta": delta.reindex(eligible_dates),
                "illiq": illiq.reindex(eligible_dates),
                "flag": flags.reindex(eligible_dates),
            }
        )
        evaluated = eligible[eligible["flag"].notna()].copy()
        evaluated["flag"] = evaluated["flag"].astype(bool)
        n_warmup = int((flags.reindex(eligible_dates).isna()
                        & illiq.reindex(eligible_dates).notna()).sum())
        n_illiq_nan = int(illiq.reindex(eligible_dates).isna().sum())

        event_dates = evaluated.index[evaluated["flag"]]
        event_delta = evaluated.loc[event_dates, "delta"]
        nonevent_delta = evaluated.loc[~evaluated["flag"], "delta"]

        primary = summarize(event_delta, nonevent_delta)
        clusters = find_clusters(event_dates, panel.index)
        n_clusters = len(clusters)

        # (a) Declustered: first day of each cluster, primary baseline.
        dec_dates = decluster_events(event_dates, panel.index)
        declustered = summarize(evaluated.loc[dec_dates, "delta"], nonevent_delta)

        # (b) Top-20 by illiq_agg within the evaluated sample
        # (full-sample ranking -> descriptive, not point-in-time).
        top_dates = evaluated["illiq"].nlargest(TOP_N).index
        top20 = summarize(
            evaluated.loc[top_dates, "delta"],
            evaluated.loc[~evaluated.index.isin(top_dates), "delta"],
        )

        # Sensitivity: full OAS-eligible sample, unclassifiable = non-event.
        sens_flag = eligible["flag"].fillna(False).astype(bool)
        sensitivity = summarize(
            eligible.loc[sens_flag, "delta"], eligible.loc[~sens_flag, "delta"]
        )

        results["series"][name] = {
            "fred_series": cfg["fred"],
            "oas_csv": {
                "start": str(oas.index.min().date()),
                "end": str(oas.index.max().date()),
                "n_obs": int(len(oas)),
            },
            "sample": {
                "oas_eligible_n": int(len(eligible)),
                "oas_eligible_start": str(eligible.index.min().date()),
                "oas_eligible_end": str(eligible.index.max().date()),
                "excluded_warmup_quantile_undefined_n": n_warmup,
                "excluded_illiq_nan_n": n_illiq_nan,
                "evaluated_n": int(len(evaluated)),
                "evaluated_start": str(evaluated.index.min().date()),
                "evaluated_end": str(evaluated.index.max().date()),
            },
            "n_events_primary": int(len(event_dates)),
            "n_event_clusters_primary": int(n_clusters),
            "event_dates_primary": [str(d.date()) for d in event_dates],
            "primary": primary,
            "verdict": "PASS" if primary["pass_p_lt_0.05"] else "FAIL",
            "declustered": {
                **declustered,
                "kept_dates": [str(d.date()) for d in dec_dates],
                "verdict": "PASS" if declustered["pass_p_lt_0.05"] else "FAIL",
            },
            "top20_descriptive_full_sample_ranking": {
                **top20,
                "dates": [str(d.date()) for d in top_dates],
                "verdict": "PASS" if top20["pass_p_lt_0.05"] else "FAIL",
            },
            "sensitivity_all_oas_eligible_unclassifiable_as_nonevent": sensitivity,
        }
    return results


def _fmt_block(label: str, s: dict) -> str:
    return (
        f"  {label}: n={s['n']} hits={s['hits']} hit_rate={s['hit_rate']:.4f} "
        f"mean={s['mean_delta_pp']:+.4f}pp median={s['median_delta_pp']:+.4f}pp | "
        f"baseline n={s['baseline_n']} hit_rate={s['baseline_hit_rate']:.4f} "
        f"mean={s['baseline_mean_delta_pp']:+.4f}pp "
        f"median={s['baseline_median_delta_pp']:+.4f}pp | "
        f"binomial p={s['binomial_p']:.4f} -> "
        f"{'PASS' if s['pass_p_lt_0.05'] else 'FAIL'}"
    )


def main() -> None:
    results = run()
    print("ML Phase 1 event study — illiq_agg trailing-Q90 days -> 5d OAS widening")
    print(
        f"Panel: {results['panel']['start']}..{results['panel']['end']} "
        f"({results['panel']['n_days']} days); rule: illiq >= trailing-252d "
        f"Q90 (min_periods=126); horizon 5 trading days; PASS bar p<0.05"
    )
    for name, r in results["series"].items():
        smp = r["sample"]
        print(f"\n[{name}] FRED {r['fred_series']} "
              f"({r['oas_csv']['start']}..{r['oas_csv']['end']}, {r['oas_csv']['n_obs']} obs)")
        print(
            f"  Sample: OAS-eligible n={smp['oas_eligible_n']} "
            f"({smp['oas_eligible_start']}..{smp['oas_eligible_end']}); excluded "
            f"{smp['excluded_warmup_quantile_undefined_n']} warm-up (quantile "
            f"undefined) + {smp['excluded_illiq_nan_n']} illiq-NaN; evaluated "
            f"n={smp['evaluated_n']} ({smp['evaluated_start']}..{smp['evaluated_end']})"
        )
        print(
            f"  Events (primary): {r['n_events_primary']} in "
            f"{r['n_event_clusters_primary']} clusters"
        )
        print(_fmt_block("PRIMARY    ", r["primary"]))
        print(_fmt_block("DECLUSTERED", r["declustered"]))
        print(_fmt_block("TOP20(desc)", r["top20_descriptive_full_sample_ranking"]))
        print(_fmt_block("SENSITIVITY", r["sensitivity_all_oas_eligible_unclassifiable_as_nonevent"]))
        print(f"  VERDICT [{name}]: {r['verdict']}")
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote {RESULTS_JSON}")


if __name__ == "__main__":
    main()
