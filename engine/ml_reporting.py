"""Stage A reporting wiring (E2 power accounting + E3 calibration).

Approved by Harry 2026-10-27 as part of Stage A of the plan in
``goals/market-knowledge-agent-ml-prediction-engine/files/
next_testing_options_2026-10-27.md`` (proposals E2 and E3). This
module is REPORTING ONLY:

- it fits no model and selects nothing (zero trials in any family);
- it reads only predictions already stored in a scorecard file,
  with their settled outcomes — it never re-fits and never
  re-scores;
- the calibration element is a reported element, never a
  graduation leg, and nothing here alters any existing verdict,
  bar, or grading.

Formulas (implemented exactly as stated in the Stage A task; no
separate frozen-formulas note existed on disk at implementation
time, 2026-10-27 — flagged in the implementation report):

- Effective independent outcomes = settled predictions / horizon
  (the program's existing convention for overlapping horizons;
  e.g. 5 for every 5-trading-day target, 1 for T1a1).
- Significance level alpha = 0.05 (the frozen level), one-sided in
  the direction of improvement, with conventional power 0.80.
  z_alpha = Phi^-1(0.95), z_power = Phi^-1(0.80).
- Alert leg (classification): alerts are the frozen top-decile set
  (budget 0.10, the harness / fix-pack selection rule) of the
  model's settled predictions. The base rate is the target's frozen
  climatology probability where climatology entries are stored in
  the same scorecard (their mean over settled climatology entries),
  otherwise the settled event frequency. The effective alert count
  is the raw alert count / horizon, floored to whole independent
  alerts (conservative — flooring can only enlarge the MDE). The
  MDE is the smallest true hit rate p1 > base rate whose exact
  one-sided binomial test at the frozen base rate, with that
  effective count, attains power >= 0.80: the critical count k* is
  the smallest k with P_{p0}(X >= k) <= 0.05, and p1 is found by
  bisection on P_{p1}(X >= k*) >= 0.80. Where no k can reject at
  0.05, the MDE is reported as not attainable at the current count.
- Brier leg (classification): paired day-by-day Brier differences
  d = Brier(model) - Brier(climatology) on as-of dates where both
  are settled in the same scorecard.
  MDE = (z_alpha + z_power) * sd(d) / sqrt(n_eff), with sd the
  sample standard deviation (ddof=1) of the paired differences and
  n_eff = n_paired / horizon.
- Mean absolute error leg (regression, the analogous form): paired
  day-by-day absolute-error differences against the target's
  comparator (persistence for Target 1; climatology for Target
  4a), same formula as the Brier leg.
- Where an MDE exceeds the corresponding backtest effect (tables
  below, sourced from the frozen results files named in the table
  comments), the block states that plainly.

Calibration (E3), classification targets only, computed on the
full settled set of the scorecard only and labelled descriptive:

- Reliability bins: 10 equal-width bins on [0, 1] — [0.0, 0.1),
  [0.1, 0.2), ..., [0.9, 1.0] — each non-empty bin reports its
  count, mean forecast, and realized event frequency.
- Calibration intercept and slope: logistic regression of the
  settled 0/1 outcome on the model's forecast on the log-odds
  scale, logit(p), fitted by maximum likelihood (scipy BFGS on the
  exact Bernoulli log-likelihood; forecasts clipped to
  [0.001, 0.999] for the log-odds transform only, per Section C.3
  of Protocol amendment v2026-10-27-A). Intercept 0
  and slope 1 indicate perfect calibration; a slope below 1
  indicates forecasts that are too extreme.
- Expected calibration error: the bin-size-weighted mean absolute
  gap between mean forecast and realized frequency over the
  stated bins above.

Failure isolation (hard requirement, T4a / shadow precedent):
``refresh_scorecard_reporting`` never raises. Any failure in the
computation is recorded as an ``ml_reporting_error`` note in the
affected file's header, any pre-existing reporting block is left
untouched, and the caller proceeds. Issued entries are never
modified: the block lives in the file header (before the first
entry heading), delimited by markers, and is replaced in place on
each refresh.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from scipy import optimize, stats

from engine import ml_scorecard as SC
from engine.ml_fixpack_stats import alert_selection

ALPHA = 0.05
POWER = 0.80
_Z_ALPHA = float(stats.norm.ppf(1.0 - ALPHA))  # 1.6448536269514722
_Z_POWER = float(stats.norm.ppf(POWER))       # 0.8416212335729143
ALERT_BUDGET = 0.10
N_CALIBRATION_BINS = 10
_LOGIT_CLIP = 1e-3

BEGIN_MARKER = "<!-- BEGIN STAGE-A REPORTING BLOCK (E2/E3) -->"
END_MARKER = "<!-- END STAGE-A REPORTING BLOCK (E2/E3) -->"
_STATUS_PREFIX = "- ml_reporting_status:"
_ERROR_PREFIX = "- ml_reporting_error:"

_CLASSIFICATION_TARGETS = {"T1b_HY", "T1b_IG"}
_HORIZON_FALLBACK = {"T1a1": 1, "T1a5": 5, "T1b_HY": 5, "T1b_IG": 5,
                     "T4a-NG-5d": 5}

# Backtest effects, computed from the frozen results files at
# implementation time (2026-10-27). Brier effect = climatology
# Brier - model Brier (positive = the model improved on
# climatology); alert effect = alert hit rate - frozen null rate.
# Sources: reports/ml_target1_results.json (v1 lines) and
# reports/ml_target1_v2_results.json, targets.*.versions.v2 (v2
# lines). MAE effect = comparator MAE - model MAE (comparator:
# persistence for Target 1, from the same files; climatology for
# Target 4a, from reports/ml_target4a_results.json metrics_pct
# REDUCED — (6.517473483305825 - 6.404230538927461) / 100 in raw
# log-return units).
BACKTEST_BRIER_ALERT_EFFECTS: dict[tuple[str, str], tuple[float, float]] = {
    ("logistic:v1-frozen14", "T1b_HY"): (0.0011192512334191884, 0.10142277691737642),
    ("gbm:v1-frozen14", "T1b_HY"): (-0.03957676498245988, -0.12934645385185436),
    ("baseline:event_rule", "T1b_HY"): (-0.0004994444968650924, -0.0908849153903159),
    ("logistic:v1-frozen14", "T1b_IG"): (-0.003779727155202306, -0.043439180022923546),
    ("gbm:v1-frozen14", "T1b_IG"): (-0.07723858507942, -0.043439180022923546),
    ("baseline:event_rule", "T1b_IG"): (0.00025567496545897406, -0.004977641561385082),
    ("logistic:v2-frozen32", "T1b_HY"): (0.0033897354665189117, 0.2734976928795644),
    ("gbm:v2-frozen32", "T1b_HY"): (-0.021725359298995855, 0.014238433620305135),
    ("logistic:v2-frozen32", "T1b_IG"): (0.0011375591953013131, 0.096182181965799),
    ("gbm:v2-frozen32", "T1b_IG"): (-0.032764148460311976, 0.2813673671509842),
}
BACKTEST_MAE_EFFECTS: dict[tuple[str, str], float] = {
    ("ridge:v1-frozen14", "T1a1"): 2.3831471548056857e-06,
    ("gbm:v1-frozen14", "T1a1"): 9.519701337346151e-08,
    ("ridge:v1-frozen14", "T1a5"): -1.9461998086354986e-06,
    ("gbm:v1-frozen14", "T1a5"): -1.4334153071476813e-06,
    ("ridge:v2-frozen32", "T1a1"): 2.23766865898332e-06,
    ("gbm:v2-frozen32", "T1a1"): 1.3426531853161972e-06,
    ("ridge:v2-frozen32", "T1a5"): -7.883660449862318e-07,
    ("gbm:v2-frozen32", "T1a5"): 1.708743471452862e-07,
    ("ridge:t4a-reduced18", "T4a-NG-5d"): 0.00113242954378364,
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _fmt(v: float) -> str:
    return format(float(v), ".6g")


def _horizon(entry: dict) -> int:
    m = re.search(r"(\d+)", entry["fields"].get("horizon", ""))
    if m:
        return int(m.group(1))
    return _HORIZON_FALLBACK.get(entry["target_key"], 1)


def effective_outcomes(n_settled: int, horizon: int) -> float:
    """Settled predictions / horizon (the program's convention)."""
    return float(n_settled) / float(horizon)


def _settled(entries: list[dict]) -> list[dict]:
    return [e for e in entries if e["outcome"] != "pending"]


def _error_value(entry: dict, name: str) -> float | None:
    m = re.search(rf"{name}=([-\d.eE+]+)", entry["error"])
    return float(m.group(1)) if m else None


def _event_outcome(entry: dict) -> int | None:
    m = re.search(r"event=([01])\b", entry["outcome"])
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Power computations (E2) — pure functions
# ---------------------------------------------------------------------------
def mde_paired(diffs, horizon: int) -> dict:
    """MDE for a paired score difference (Brier or absolute error).

    ``diffs`` are day-by-day (model - comparator) differences on
    paired settled days. Returns the MDE magnitude (an improvement
    of this size is the smallest the current effective count can
    distinguish from chance at alpha=0.05 with power 0.80)."""
    d = np.asarray(list(diffs), dtype=float)
    n = int(len(d))
    n_eff = effective_outcomes(n, horizon)
    if n < 2:
        return {"mde": None, "n_paired": n, "n_eff": n_eff, "sd": None,
                "reason": f"fewer than 2 paired settled days (n_paired={n})"}
    sd = float(np.std(d, ddof=1))
    if not np.isfinite(sd) or sd == 0.0:
        return {"mde": None, "n_paired": n, "n_eff": n_eff, "sd": sd,
                "reason": "paired differences have zero standard deviation"}
    mde = (_Z_ALPHA + _Z_POWER) * sd / math.sqrt(n_eff)
    return {"mde": float(mde), "n_paired": n, "n_eff": n_eff, "sd": sd,
            "reason": None}


def mde_alert(base_rate: float, n_alert_effective: float) -> dict:
    """Exact binomial MDE for the alert leg.

    The effective alert count is floored to whole independent
    alerts (conservative). Returns the minimum detectable hit
    rate and its excess over ``base_rate``, or an unattainable /
    not-computable verdict with its reason."""
    p0 = float(base_rate)
    n = int(math.floor(float(n_alert_effective)))
    if n < 1:
        return {"mde_hit_rate": None, "mde_excess": None, "n_alert_eff_int": n,
                "critical_k": None,
                "reason": (f"effective alert count {_fmt(n_alert_effective)} is "
                           "below one independent alert")}
    # Critical count: smallest k with P_{p0}(X >= k) <= ALPHA.
    k_star = None
    for k in range(0, n + 2):
        if float(stats.binom.sf(k - 1, n, p0)) <= ALPHA:
            k_star = k
            break
    if k_star is None or k_star > n:
        return {"mde_hit_rate": None, "mde_excess": None, "n_alert_eff_int": n,
                "critical_k": k_star,
                "reason": (f"no rejection is possible at the 0.05 level with "
                           f"{n} independent alert(s) at base rate {_fmt(p0)} "
                           "(even a perfect hit rate is not significant)")}
    if k_star == 0:
        return {"mde_hit_rate": p0, "mde_excess": 0.0, "n_alert_eff_int": n,
                "critical_k": 0, "reason": None}
    # Bisection: smallest p1 with P_{p1}(X >= k*) >= POWER.
    lo, hi = p0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(stats.binom.sf(k_star - 1, n, mid)) >= POWER:
            hi = mid
        else:
            lo = mid
    p1 = hi
    return {"mde_hit_rate": float(p1), "mde_excess": float(p1 - p0),
            "n_alert_eff_int": n, "critical_k": int(k_star), "reason": None}


# ---------------------------------------------------------------------------
# Calibration computations (E3) — pure functions on stored predictions
# ---------------------------------------------------------------------------
def reliability_bins(probs, outcomes, n_bins: int = N_CALIBRATION_BINS) -> list[dict]:
    """Equal-width reliability bins on [0, 1]; non-empty bins only."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict] = []
    for b in range(n_bins):
        lo, hi = float(edges[b]), float(edges[b + 1])
        if b < n_bins - 1:
            mask = (p >= lo) & (p < hi)
        else:
            mask = (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        out.append({
            "lo": lo, "hi": hi, "n": int(mask.sum()),
            "mean_forecast": float(p[mask].mean()),
            "realized_frequency": float(y[mask].mean()),
        })
    return out


def expected_calibration_error(bins: list[dict], n_total: int) -> float | None:
    if not bins or n_total <= 0:
        return None
    return float(sum(
        (b["n"] / n_total) * abs(b["mean_forecast"] - b["realized_frequency"])
        for b in bins
    ))


def calibration_intercept_slope(probs, outcomes) -> dict:
    """Logistic regression of outcome on logit(forecast), by exact
    maximum likelihood. Intercept 0 / slope 1 = perfect calibration."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if len(p) < 3:
        return {"intercept": None, "slope": None,
                "reason": f"fewer than 3 settled predictions (n={len(p)})"}
    if len(np.unique(y)) < 2:
        return {"intercept": None, "slope": None,
                "reason": "settled outcomes contain a single class only"}
    pc = np.clip(p, _LOGIT_CLIP, 1.0 - _LOGIT_CLIP)
    x = np.log(pc / (1.0 - pc))
    if float(np.std(x)) == 0.0:
        return {"intercept": None, "slope": None,
                "reason": "forecasts are constant; the log-odds regressor has no variation"}

    def nll(beta):
        z = beta[0] + beta[1] * x
        # log(1 + exp(z)) computed stably; log-likelihood of Bernoulli.
        ll = np.sum(y * z - np.logaddexp(0.0, z))
        return -float(ll)

    def grad(beta):
        z = beta[0] + beta[1] * x
        mu = 1.0 / (1.0 + np.exp(-z))
        r = mu - y
        return np.array([float(np.sum(r)), float(np.sum(r * x))])

    res = optimize.minimize(nll, x0=np.array([0.0, 1.0]), jac=grad,
                            method="BFGS")
    if not res.success or not np.all(np.isfinite(res.x)):
        return {"intercept": None, "slope": None,
                "reason": (f"logistic calibration fit did not converge "
                           f"({str(res.message).rstrip('.')})")}
    a, b = float(res.x[0]), float(res.x[1])
    if abs(a) > 25.0 or abs(b) > 25.0:
        return {"intercept": None, "slope": None,
                "reason": ("logistic calibration fit separated (a coefficient "
                           "diverged beyond the reporting bound of 25)")}
    return {"intercept": a, "slope": b, "reason": None}


def calibration_assessment(probs, outcomes) -> dict:
    """Bins + intercept/slope + ECE for one (target, model) line."""
    p = [float(v) for v in probs]
    y = [int(v) for v in outcomes]
    bins = reliability_bins(p, y)
    ece = expected_calibration_error(bins, len(p))
    fit = calibration_intercept_slope(p, y)
    return {"bins": bins, "ece": ece, "intercept": fit["intercept"],
            "slope": fit["slope"], "fit_reason": fit["reason"], "n": len(p)}


# ---------------------------------------------------------------------------
# Extraction from parsed scorecard entries
# ---------------------------------------------------------------------------
def _grouped(entries: list[dict]) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for e in entries:
        groups.setdefault((e["target_key"], e["model_version"]), []).append(e)
    return groups


def _paired_by_asof(entries: list[dict]) -> dict[tuple[str, str], dict[str, dict]]:
    """(target, asof) -> {model_version: settled entry}."""
    out: dict[tuple[str, str], dict[str, dict]] = {}
    for e in _settled(entries):
        out.setdefault((e["target_key"], e["asof"]), {})[e["model_version"]] = e
    return out


def _base_rate(entries: list[dict], target_key: str) -> tuple[float | None, str]:
    """Frozen base rate for a classification target: the mean
    stored climatology probability over settled climatology
    entries in this scorecard; falling back to the settled event
    frequency over distinct as-of dates."""
    clim = [e for e in _settled(entries)
            if e["target_key"] == target_key
            and e["model_version"] == "baseline:climatology"]
    probs = []
    for e in clim:
        try:
            probs.append(SC._prediction_prob(e["fields"].get("prediction", ""))[0])
        except ValueError:
            continue
    if probs:
        return float(np.mean(probs)), "mean stored climatology probability in this scorecard"
    seen: dict[str, int] = {}
    for e in _settled(entries):
        if e["target_key"] != target_key:
            continue
        y = _event_outcome(e)
        if y is not None:
            seen.setdefault(e["asof"], y)
    if seen:
        return float(np.mean(list(seen.values()))), "settled event frequency (no climatology entries stored in this scorecard)"
    return None, "no settled climatology entries and no settled events"


def _classification_pairs(entry_list: list[dict]) -> tuple[list[float], list[int]]:
    probs, outcomes = [], []
    for e in entry_list:
        y = _event_outcome(e)
        if y is None:
            continue
        try:
            p, _thr = SC._prediction_prob(e["fields"].get("prediction", ""))
        except ValueError:
            continue
        probs.append(p)
        outcomes.append(y)
    return probs, outcomes


def _comparator_model(target_key: str, paired: dict) -> str | None:
    """Comparator for paired differences: climatology for
    classification and Target 4a; persistence for Target 1
    regression (falling back to climatology)."""
    models = set()
    for (tk, _asof), by_model in paired.items():
        if tk == target_key:
            models.update(by_model)
    if target_key in _CLASSIFICATION_TARGETS or target_key == "T4a-NG-5d":
        return "baseline:climatology" if "baseline:climatology" in models else None
    if "baseline:persistence" in models:
        return "baseline:persistence"
    if "baseline:climatology" in models:
        return "baseline:climatology"
    return None


# ---------------------------------------------------------------------------
# Block construction
# ---------------------------------------------------------------------------
def build_reporting_block(entries: list[dict]) -> list[str]:
    """The standing E2/E3 block (list of markdown lines) computed
    from parsed scorecard entries. Raises on failure — the caller
    (refresh_scorecard_reporting) isolates failures."""
    groups = _grouped(entries)
    paired = _paired_by_asof(entries)
    lines: list[str] = [
        "## Standing reporting block — power accounting (E2) and calibration (E3)",
        "",
        "Stage A reporting (approved 2026-10-27). This block is reporting only: it fits "
        "nothing, selects nothing, and counts as zero trials in every family. It is "
        "recomputed from the predictions stored in this file and their settled "
        "outcomes on each daily pass; issued entries are never edited by it. "
        "Conventions: effective independent outcomes = settled predictions divided "
        "by the horizon in trading days; minimum detectable effects (MDE) are at "
        "the frozen significance level 0.05 (one-sided, in the direction of "
        "improvement) with conventional power 0.80; the alert leg uses the exact "
        "one-sided binomial test at the frozen base rate and the current effective "
        "alert count (raw alert count divided by the horizon, floored to whole "
        "independent alerts); the Brier and mean absolute error legs use the "
        "standard deviation of the paired day-by-day score differences against "
        "the stated comparator. The calibration element is descriptive only, is "
        "computed on the full settled set only, and is not a graduation leg; it "
        "does not alter any verdict, bar, or grading.",
        "",
        "### Power accounting (E2)",
        "",
    ]
    if not groups:
        lines.append("- No entries in this scorecard yet; no settled predictions to report.")
    for (tk, mv), ents in sorted(groups.items()):
        settled = _settled(ents)
        n = len(settled)
        horizon = _horizon(ents[0]) if ents else _HORIZON_FALLBACK.get(tk, 1)
        n_eff = effective_outcomes(n, horizon)
        head = (f"- {tk} / {mv}: settled predictions = {n}; effective independent "
                f"outcomes = {_fmt(n_eff)} ({n} settled / {horizon}-trading-day horizon)")
        if mv in ("baseline:climatology", "baseline:persistence", "baseline:zero-change"):
            lines.append(head + ". This line is a comparator baseline; no MDE leg is computed against itself.")
            continue
        if n == 0:
            lines.append(head + ". No settled predictions yet; no MDE is computable.")
            continue
        comparator = _comparator_model(tk, paired)
        if tk in _CLASSIFICATION_TARGETS:
            # Brier leg.
            if comparator is None:
                lines.append(head + ". Brier leg: not computable — no settled climatology "
                                    "entries are stored in this scorecard to pair against.")
            else:
                diffs = []
                for e in settled:
                    by_model = paired.get((tk, e["asof"]), {})
                    clim = by_model.get(comparator)
                    b_model = _error_value(e, "brier")
                    b_clim = _error_value(clim, "brier") if clim is not None else None
                    if b_model is not None and b_clim is not None:
                        diffs.append(b_model - b_clim)
                res = mde_paired(diffs, horizon)
                if res["mde"] is None:
                    lines.append(head + f". Brier leg: not computable — {res['reason']} "
                                        f"(settled predictions for this line = {n}); no MDE is stated.")
                else:
                    leg = (f". Brier leg: minimum detectable Brier improvement over climatology "
                           f"= {_fmt(res['mde'])} (paired days = {res['n_paired']}, effective "
                           f"paired outcomes = {_fmt(res['n_eff'])}, standard deviation of paired "
                           f"Brier differences = {_fmt(res['sd'])})")
                    bt = BACKTEST_BRIER_ALERT_EFFECTS.get((mv, tk))
                    if bt is not None and res["mde"] > bt[0]:
                        leg += (f". This MDE exceeds the corresponding backtest effect: the "
                                f"backtest Brier improvement over climatology was {_fmt(bt[0])}, "
                                f"so the current forward sample cannot distinguish an effect of "
                                f"the backtest size from chance at the stated level and power.")
                    lines.append(head + leg)
            # Alert leg (appended to the same line's block as its own bullet).
            probs, outcomes = _classification_pairs(settled)
            base, base_src = _base_rate(entries, tk)
            if base is None or not probs:
                lines.append(f"  - Alert leg ({tk} / {mv}): not computable — {base_src}.")
            else:
                sel = alert_selection(np.asarray(outcomes), np.asarray(probs),
                                      budget=ALERT_BUDGET)
                n_alert_eff = effective_outcomes(int(sel["n_alert"]), horizon)
                ares = mde_alert(base, n_alert_eff)
                if ares["mde_excess"] is None:
                    lines.append(
                        f"  - Alert leg ({tk} / {mv}): not computable — {ares['reason']} "
                        f"(alerts = top-decile set of settled predictions, {sel['n_alert']} raw "
                        f"alert(s), base rate = {_fmt(base)} from {base_src}).")
                else:
                    leg = (f"  - Alert leg ({tk} / {mv}): minimum detectable alert hit-rate "
                           f"excess over the base rate = {_fmt(ares['mde_excess'])} (minimum "
                           f"detectable hit rate = {_fmt(ares['mde_hit_rate'])} against base "
                           f"rate = {_fmt(base)} from {base_src}; {ares['n_alert_eff_int']} "
                           f"effective independent alert(s) from {sel['n_alert']} raw alert(s), "
                           f"exact binomial test, critical count = {ares['critical_k']})")
                    bt = BACKTEST_BRIER_ALERT_EFFECTS.get((mv, tk))
                    if bt is not None and ares["mde_excess"] > bt[1]:
                        leg += (f". This MDE exceeds the corresponding backtest effect: the "
                                f"backtest alert hit-rate excess over the base rate was "
                                f"{_fmt(bt[1])}, so the current forward sample cannot "
                                f"distinguish an effect of the backtest size from chance at "
                                f"the stated level and power")
                    leg += "."
                    lines.append(leg)
        else:
            # Regression: paired absolute-error leg vs the comparator.
            if comparator is None:
                lines.append(head + ". Mean absolute error leg: not computable — no settled "
                                    "comparator entries are stored in this scorecard to pair against.")
                continue
            diffs = []
            for e in settled:
                by_model = paired.get((tk, e["asof"]), {})
                comp = by_model.get(comparator)
                a_model = _error_value(e, "abs_error")
                a_comp = _error_value(comp, "abs_error") if comp is not None else None
                if a_model is not None and a_comp is not None:
                    diffs.append(a_model - a_comp)
            res = mde_paired(diffs, horizon)
            if res["mde"] is None:
                lines.append(head + f". Mean absolute error leg: not computable — {res['reason']} "
                                    f"(settled predictions for this line = {n}); no MDE is stated.")
            else:
                leg = (f". Mean absolute error leg: minimum detectable mean absolute error "
                       f"improvement over {comparator} = {_fmt(res['mde'])} (paired days = "
                       f"{res['n_paired']}, effective paired outcomes = {_fmt(res['n_eff'])}, "
                       f"standard deviation of paired absolute-error differences = {_fmt(res['sd'])})")
                bt = BACKTEST_MAE_EFFECTS.get((mv, tk))
                if bt is not None and res["mde"] > bt:
                    leg += (f". This MDE exceeds the corresponding backtest effect: the backtest "
                            f"mean absolute error improvement over the comparator was {_fmt(bt)}, "
                            f"so the current forward sample cannot distinguish an effect of the "
                            f"backtest size from chance at the stated level and power.")
                lines.append(head + leg)
    lines += [
        "",
        "### Calibration assessment (E3) — descriptive only, not a graduation leg",
        "",
        "Computed on stored predictions only (no re-fit), on the full settled set of "
        "this scorecard only, for the classification targets. Reliability bins are "
        "the 10 equal-width bins on [0, 1] stated in the method note above; only "
        "non-empty bins are listed. The calibration intercept and slope come from a "
        "logistic regression of the settled outcome on the model's forecast on the "
        "log-odds scale, with forecasts clipped to [0.001, 0.999] before the logit "
        "is taken (the clipping interval stated here wherever calibration estimates "
        "are printed, per Section C.3); intercept 0 and slope 1 indicate perfect calibration, and a "
        "slope below 1 indicates forecasts that are too extreme. Expected "
        "calibration error is the bin-size-weighted mean absolute gap between mean "
        "forecast and realized frequency over the stated bins. Each line carries "
        "its effective independent outcome count; at forward-record sizes these "
        "figures describe the settled set and support no verdict.",
        "",
    ]
    class_lines: list[str] = []
    for (tk, mv), ents in sorted(groups.items()):
        if tk not in _CLASSIFICATION_TARGETS or mv == "baseline:climatology":
            continue
        settled = _settled(ents)
        horizon = _horizon(ents[0]) if ents else _HORIZON_FALLBACK.get(tk, 1)
        n_eff = effective_outcomes(len(settled), horizon)
        probs, outcomes = _classification_pairs(settled)
        prefix = (f"- {tk} / {mv} (descriptive; settled = {len(settled)}, effective "
                  f"independent outcomes = {_fmt(n_eff)}): ")
        if not probs:
            class_lines.append(prefix + "calibration not computable — no settled "
                                        "classification predictions are stored yet; "
                                        "this is reported, not imputed.")
            continue
        cal = calibration_assessment(probs, outcomes)
        if cal["bins"]:
            bin_txt = "; ".join(
                f"[{b['lo']:.1f}, {b['hi']:.1f}{']' if b['hi'] == 1.0 else ')'}: "
                f"n={b['n']}, mean forecast={_fmt(b['mean_forecast'])}, realized "
                f"frequency={_fmt(b['realized_frequency'])}"
                for b in cal["bins"]
            )
        else:
            bin_txt = "no non-empty bins"
        txt = prefix + f"reliability bins: {bin_txt}. "
        if cal["intercept"] is None:
            txt += (f"Calibration intercept and slope: not computable — "
                    f"{cal['fit_reason']}; this is reported, not imputed. ")
        else:
            txt += (f"Calibration intercept = {_fmt(cal['intercept'])}, slope = "
                    f"{_fmt(cal['slope'])}. ")
        if cal["ece"] is None:
            txt += "Expected calibration error: not computable."
        else:
            txt += (f"Expected calibration error = {_fmt(cal['ece'])} (binning: "
                    f"{N_CALIBRATION_BINS} equal-width bins on [0, 1]).")
        class_lines.append(txt)
    if class_lines:
        lines.extend(class_lines)
    else:
        lines.append("- No classification targets are present in this scorecard; "
                     "the calibration element does not apply.")
    return lines


# ---------------------------------------------------------------------------
# Header splicing + failure-isolated refresh
# ---------------------------------------------------------------------------
def _first_entry_index(lines: list[str]) -> int:
    for i, ln in enumerate(lines):
        if SC._ID_RE.match(ln):
            return i
    return len(lines)


def _splice(text: str, block_lines: list[str] | None, status: str,
            error: str) -> str:
    """Return the scorecard text with header status lines set and,
    when ``block_lines`` is given, the delimited reporting block
    replaced (or inserted) in the header. Entry content — every
    line from the first entry heading onward — is preserved
    byte-for-byte. When ``block_lines`` is None, any existing block
    is preserved as-is."""
    lines = text.splitlines()
    first = _first_entry_index(lines)
    header, rest = lines[:first], lines[first:]
    # Extract and remove any existing block from the header.
    existing_block: list[str] = []
    cleaned: list[str] = []
    i = 0
    while i < len(header):
        if header[i].strip() == BEGIN_MARKER:
            j = i
            while j < len(header) and header[j].strip() != END_MARKER:
                j += 1
            existing_block = header[i:j + 1] if j < len(header) else header[i:]
            i = (j + 1) if j < len(header) else len(header)
            continue
        cleaned.append(header[i])
        i += 1
    cleaned = [ln for ln in cleaned
               if not ln.startswith(_STATUS_PREFIX)
               and not ln.startswith(_ERROR_PREFIX)]
    while cleaned and cleaned[-1] == "":
        cleaned.pop()
    block = block_lines
    if block is None:
        block_out = existing_block
    else:
        block_out = [BEGIN_MARKER, *block, END_MARKER]
    out_header = cleaned + [
        f"{_STATUS_PREFIX} {status}",
        f"{_ERROR_PREFIX} {error}",
    ]
    if block_out:
        out_header += [""] + block_out
    out = out_header + [""] + rest if rest else out_header
    return "\n".join(out).rstrip("\n") + "\n"


def refresh_scorecard_reporting(path) -> dict:
    """Recompute the E2/E3 block for one scorecard file, in place.

    NEVER raises (failure isolation, hard requirement): a failure
    in the computation leaves any existing block untouched, records
    the failure as an ``ml_reporting_error`` note in the file
    header, and returns a failed status for the caller's summary.
    """
    path = Path(path)
    try:
        text = path.read_text() if path.exists() else ""
        entries = SC.parse_entries(text) if text else []
        block = build_reporting_block(entries)
        new_text = _splice(text, block, status="ok", error="none")
        if text and new_text != text:
            path.write_text(new_text)
        return {"status": "ok", "path": str(path)}
    except Exception as exc:  # noqa: BLE001 - reporting must never break a pass
        msg = (str(exc) or exc.__class__.__name__)[-300:]
        try:
            if path.exists():
                text = path.read_text()
                new_text = _splice(text, None, status="failed", error=msg)
                if new_text != text:
                    path.write_text(new_text)
        except Exception:
            pass
        return {"status": "failed", "error": msg, "path": str(path)}


__all__ = [
    "ALPHA", "POWER", "BEGIN_MARKER", "END_MARKER",
    "effective_outcomes", "mde_paired", "mde_alert",
    "reliability_bins", "expected_calibration_error",
    "calibration_intercept_slope", "calibration_assessment",
    "build_reporting_block", "refresh_scorecard_reporting",
]
