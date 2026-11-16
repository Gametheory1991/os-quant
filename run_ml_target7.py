#!/usr/bin/env python
"""Runner for ML Target 7 — CPI hedge (S2) optimization test.

Frozen spec: ~/workspace/goals/market-knowledge-agent-ml-prediction-engine/
files/ml_target7_cpi_hedge_experiment_spec_2026-10-31.md (frozen 2026-10-31
before any fitting). Offline evaluation only; no forward issuance.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine.backtest import run_backtest
from engine import stats as eng_stats

HOME = Path.home()
QE = HOME / "workspace/quant-engine"
CACHE = QE / "data/cache"
BELTRAN = QE / "data/beltran_li_yield_impacts/series"
LAKE = HOME / "workspace/data-lake-staging/fred/snapshot=2026-10"
T7DIR = QE / "data/ml_target7"
REPORTS = QE / "reports"
LEDGER = HOME / "workspace/goals/quant-research-backtest-engine/knowledge/trials_ledger.md"

START, END = pd.Timestamp("2006-04-10"), pd.Timestamp("2026-10-16")
CAPITAL = 100_000.0
SEED = 13
GBM_PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                  min_samples_leaf=20, l2_regularization=1.0, random_state=SEED)
LOGISTIC_CS = [0.01, 0.1, 1.0, 10.0]
MIN_TRAIN = 60

FEATURES = ["cpi_yoy", "ppi_yoy",
            "t10yie_level", "t10yie_chg_1m", "mich", "dgs10_level",
            "dgs10_chg_3m", "dgs2_level", "dgs2_chg_3m", "spread_10y2y",
            "dff_level", "vix_level", "xlp_ret_1m", "xlp_ret_3m",
            "xlp_ret_12m", "qqq_ret_1m", "qqq_ret_3m", "qqq_ret_12m",
            "spy_ret_1m", "spy_ret_3m", "spy_ret_12m", "rel_ret_1m",
            "rel_ret_3m", "xlp_rvol21", "qqq_rvol21"]


def load_adj(ticker: str) -> pd.Series:
    df = pd.read_parquet(CACHE / f"{ticker}.parquet")
    s = df["adjclose"].astype(float).sort_index()
    s.index = pd.to_datetime(s.index).normalize()
    return s


def load_beltran(sid: str) -> pd.Series:
    df = pd.read_parquet(BELTRAN / f"{sid}.parquet")
    s = df["value"].astype(float)
    s.index = pd.to_datetime(s.index).normalize()
    return s.sort_index()


def load_fred_json(sid: str) -> pd.Series:
    d = json.loads((T7DIR / f"fred_{sid}.json").read_text())
    rows = [(o["date"], o["value"]) for o in d["observations"] if o["value"] != "."]
    s = pd.Series({pd.Timestamp(dt): float(v) for dt, v in rows}).sort_index()
    s.index = pd.to_datetime(s.index).normalize()
    return s


def load_lake(release: int, sid: str) -> pd.Series:
    parts = sorted((LAKE / f"release_id={release}" / "freq=D").glob("part-*.parquet"))
    frames = [pd.read_parquet(p) for p in parts]
    df = pd.concat(frames, ignore_index=True)
    sub = df[df["series_id"] == sid].copy()
    sub["date"] = pd.to_datetime(sub["date"]).dt.normalize()
    s = sub.set_index("date")["value"].astype(float).sort_index()
    return s[~s.index.duplicated(keep="last")]


def _write_retry(path: Path, text: str, attempts: int = 8) -> None:
    """Write with retries: this host intermittently denies writes with
    PermissionError (Operation not permitted); a completed 58-minute run
    was lost to one such denial at its final write."""
    import time
    last = None
    for _ in range(attempts):
        try:
            path.write_text(text)
            return
        except PermissionError as e:  # host-level transient denial
            last = e
            time.sleep(10)
    raise last


def main() -> None:
    np.random.seed(SEED)
    # ---- ledger N for DSR (highest row + 8 bookings of this target) ----
    ledger_text = LEDGER.read_text()
    row_nums = [int(m) for m in re.findall(r"^\|\s*(\d{3,5})\s*\|", ledger_text, flags=re.M)]
    ledger_max = max(row_nums)
    n_trials_dsr = ledger_max + 8

    # ---- prices (full history for features; window panel for backtests) ----
    px_full = pd.DataFrame({t: load_adj(t) for t in ["XLP", "QQQ", "SPY"]}).dropna()
    cal_full = px_full.index
    px_win = px_full.loc[(px_full.index >= START) & (px_full.index <= END), ["XLP", "QQQ"]].copy()
    cal_win = px_win.index

    # ---- monthly series ----
    cpi = load_beltran("CPIAUCSL")
    core = load_fred_json("CPILFESL")
    ppi = load_fred_json("PPIACO")
    mich_s = load_fred_json("MICH")

    def yoy(s: pd.Series) -> pd.Series:
        return s / s.shift(12) - 1

    def ann3(s: pd.Series) -> pd.Series:
        return (s / s.shift(3)) ** 4 - 1

    monthly = pd.DataFrame({
        "cpi_yoy": yoy(cpi), "cpi_3m_ann": ann3(cpi),
        "core_yoy": yoy(core), "core_3m_ann": ann3(core),
        "ppi_yoy": yoy(ppi), "mich": mich_s,
    })

    # ---- daily series, asof-filled onto the full trading calendar ----
    daily_raw = {
        "t10yie": load_lake(304, "T10YIE"),
        "dgs10": load_lake(18, "DGS10"),
        "dgs2": load_lake(18, "DGS2"),
        "dff": load_lake(18, "DFF"),
        "vix": load_beltran("VIXCLS"),
    }
    daily = {}
    for k, s in daily_raw.items():
        daily[k] = s.reindex(cal_full, method="ffill")

    logret = np.log(px_full / px_full.shift(1))

    def price_feats(asof_pos: int) -> dict:
        out = {}
        for t, pre in [("XLP", "xlp"), ("QQQ", "qqq"), ("SPY", "spy")]:
            s = px_full[t]
            for name, w in [("1m", 21), ("3m", 63), ("12m", 252)]:
                out[f"{pre}_ret_{name}"] = float(s.iloc[asof_pos] / s.iloc[asof_pos - w] - 1)
        out["rel_ret_1m"] = out["qqq_ret_1m"] - out["xlp_ret_1m"]
        out["rel_ret_3m"] = out["qqq_ret_3m"] - out["xlp_ret_3m"]
        for t, pre in [("XLP", "xlp"), ("QQQ", "qqq")]:
            out[f"{pre}_rvol21"] = float(logret[t].iloc[asof_pos - 20: asof_pos + 1].std(ddof=1))
        return out

    # ---- decision calendar (identical construction to the original) ----
    cpi_yoy_s = yoy(cpi)
    decisions = []
    for ref_month, v in cpi_yoy_s.items():
        if pd.isna(v):
            continue
        knowable = ref_month + pd.DateOffset(months=1) + pd.Timedelta(days=14)
        if knowable < cal_win[0] - pd.Timedelta(days=7):
            continue
        idx = cal_win.searchsorted(knowable, side="right")
        if idx >= len(cal_win):
            continue
        ex = cal_win[idx]
        if not (START <= ex <= END):
            continue
        decisions.append({"ref_month": ref_month, "knowable": knowable,
                          "execution": ex, "cpi_yoy": float(v)})
    assert len(decisions) == 246, f"expected 246 decisions, got {len(decisions)}"
    assert sum(1 for d in decisions if d["cpi_yoy"] > 0.04) == 38

    # features + labels per decision
    for i, d in enumerate(decisions):
        mrow = monthly.loc[d["ref_month"]]
        feats = {k: float(mrow[k]) for k in ["cpi_yoy", "ppi_yoy", "mich"]}
        fdate_pos = int(cal_full.searchsorted(d["knowable"], side="right") - 1)
        fdate = cal_full[fdate_pos]
        d["feature_date"] = fdate
        feats["t10yie_level"] = float(daily["t10yie"].iloc[fdate_pos])
        feats["t10yie_chg_1m"] = float(daily["t10yie"].iloc[fdate_pos] - daily["t10yie"].iloc[fdate_pos - 21])
        feats["dgs10_level"] = float(daily["dgs10"].iloc[fdate_pos])
        feats["dgs10_chg_3m"] = float(daily["dgs10"].iloc[fdate_pos] - daily["dgs10"].iloc[fdate_pos - 63])
        feats["dgs2_level"] = float(daily["dgs2"].iloc[fdate_pos])
        feats["dgs2_chg_3m"] = float(daily["dgs2"].iloc[fdate_pos] - daily["dgs2"].iloc[fdate_pos - 63])
        feats["spread_10y2y"] = feats["dgs10_level"] - feats["dgs2_level"]
        feats["dff_level"] = float(daily["dff"].iloc[fdate_pos])
        feats["vix_level"] = float(daily["vix"].iloc[fdate_pos])
        feats.update(price_feats(fdate_pos))
        d["features"] = feats
        # label: QQQ vs XLP total return, execution close to next execution close
        if i + 1 < len(decisions):
            nxt = decisions[i + 1]["execution"]
            rq = float(px_win.loc[nxt, "QQQ"] / px_win.loc[d["execution"], "QQQ"] - 1)
            rx = float(px_win.loc[nxt, "XLP"] / px_win.loc[d["execution"], "XLP"] - 1)
            d["label"] = int(rq > rx)
            d["label_rel_ret"] = rq - rx
        else:
            d["label"] = None
            d["label_rel_ret"] = None
    X = np.array([[d["features"][f] for f in FEATURES] for d in decisions], dtype=float)
    assert np.isfinite(X).all(), "non-finite feature value in decision matrix"
    y = np.array([np.nan if d["label"] is None else float(d["label"]) for d in decisions])

    exec_dates = [d["execution"] for d in decisions]
    know_dates = [d["knowable"] for d in decisions]

    def eligible_train(i: int) -> np.ndarray:
        return np.array([j for j in range(i)
                         if j + 1 < len(decisions) and exec_dates[j + 1] < know_dates[i]
                         and not np.isnan(y[j])], dtype=int)

    # first test decision: first index with >= MIN_TRAIN eligible rows
    first_test = next(i for i in range(len(decisions)) if len(eligible_train(i)) >= MIN_TRAIN)
    oos_start = exec_dates[first_test]
    print(f"decisions=246 first_test_idx={first_test} oos_start={oos_start.date()} "
          f"eligible_train={len(eligible_train(first_test))} ledger_max={ledger_max} N_dsr={n_trials_dsr}")

    # ---- walk-forward predictions (checkpointed: the first full run's
    # computation completed but its final write was denied by the host;
    # probabilities are therefore persisted before the arm stage) ----
    models = ["logistic", "gbm"]
    probs = {m: {} for m in models}       # decision idx -> probability
    hypers = {m: {} for m in models}      # decision idx -> selected hyperparams
    fallback_used = []
    ckpt_path = T7DIR / "wf_checkpoint.json"
    if ckpt_path.exists():
        ck = json.loads(ckpt_path.read_text())
        assert ck["features"] == FEATURES and ck["first_test"] == first_test
        for m in models:
            probs[m] = {int(k): float(v) for k, v in ck["probs"][m].items()}
            hypers[m] = {int(k): v for k, v in ck["hypers"][m].items()}
        fallback_used = ck["fallback_used"]
        print(f"loaded walk-forward checkpoint {ckpt_path} ({len(probs['logistic'])} decisions)")
        _skip_fit = True
    else:
        _skip_fit = False
    if not _skip_fit:
        for i in range(first_test, len(decisions)):
            tr = eligible_train(i)
            Xtr, ytr = X[tr], y[tr].astype(int)
            Xte = X[i:i + 1]
            if len(np.unique(ytr)) < 2:
                for m in models:
                    probs[m][i] = float(ytr.mean())
                    hypers[m][i] = {"single_class_fallback": True}
                fallback_used.append(i)
                continue
            # Engine configuration (ml_models.logistic_fit_predict): C selected
            # from LOGISTIC_CS by TimeSeriesSplit(3) on training rows only,
            # scored by negative log loss. Implemented as a manual loop with
            # identical folds, scoring, and first-best tie-breaking, because
            # this host denies the semaphore creation joblib's pool requires
            # (PermissionError in the first Step 3 attempt, before any arm was
            # computed; the selection semantics are unchanged).
            tscv = TimeSeriesSplit(3)
            best_c, best_score = None, -np.inf
            for cval in LOGISTIC_CS:
                fold_scores = []
                for tr_idx, va_idx in tscv.split(Xtr):
                    if len(np.unique(ytr[tr_idx])) < 2:
                        fold_scores.append(-log_loss(ytr[va_idx],
                                                     np.full(len(va_idx), float(ytr[tr_idx].mean())),
                                                     labels=[0, 1]))
                        continue
                    p = Pipeline([("scaler", StandardScaler()),
                                  ("model", LogisticRegression(max_iter=2000, C=cval))])
                    p.fit(Xtr[tr_idx], ytr[tr_idx])
                    fold_scores.append(-log_loss(ytr[va_idx], p.predict_proba(Xtr[va_idx])[:, 1],
                                                 labels=[0, 1]))
                ms = float(np.mean(fold_scores))
                if ms > best_score:
                    best_score, best_c = ms, cval
            pipe = Pipeline([("scaler", StandardScaler()),
                             ("model", LogisticRegression(max_iter=2000, C=best_c))])
            pipe.fit(Xtr, ytr)
            probs["logistic"][i] = float(pipe.predict_proba(Xte)[0, 1])
            hypers["logistic"][i] = {"C": float(best_c)}
            gbm = HistGradientBoostingClassifier(**GBM_PARAMS)
            gbm.fit(Xtr, ytr)
            probs["gbm"][i] = float(gbm.predict_proba(Xte)[0, 1])
            hypers["gbm"][i] = {}
        _write_retry(ckpt_path, json.dumps({
            "features": FEATURES, "first_test": int(first_test),
            "probs": {m: {str(k): v for k, v in probs[m].items()} for m in models},
            "hypers": {m: {str(k): v for k, v in hypers[m].items()} for m in models},
            "fallback_used": [int(i) for i in fallback_used]}, indent=1))
        print(f"wrote walk-forward checkpoint {ckpt_path}")

    # classifier metrics per model (labelled OOS decisions only)
    clf_metrics = {}
    for m in models:
        idxs = [i for i in range(first_test, len(decisions)) if decisions[i]["label"] is not None]
        yy = np.array([decisions[i]["label"] for i in idxs], dtype=float)
        pp = np.array([probs[m][i] for i in idxs], dtype=float)
        clf_metrics[m] = {
            "n_labelled_oos": int(len(idxs)),
            "auc": float(roc_auc_score(yy, pp)) if len(np.unique(yy)) > 1 else None,
            "brier": float(np.mean((pp - yy) ** 2)),
            "base_rate": float(yy.mean()),
        }

    # ---- arms ----
    px_oos = px_win.loc[(px_win.index >= oos_start) & (px_win.index <= END), ["XLP", "QQQ"]].copy()
    oos_decisions = list(range(first_test, len(decisions)))
    exec_to_idx = {exec_dates[i]: i for i in oos_decisions}

    def run_arm(rule) -> tuple[pd.DataFrame, list[dict], int, float]:
        """rule(decision_dict, prob_by_model, w_xlp_shadow_frac) ->
        (direction, step_fraction_of_sold_holding) or None.
        direction: +1 XLP->QQQ, -1 QQQ->XLP. Returns (bt, records, n_rot, total_step)."""
        rets = px_oos.pct_change().fillna(0.0)
        hold = {"XLP": CAPITAL * 0.999, "QQQ": 0.0}
        rows: dict = {}
        records: list[dict] = []
        n_rot = 0
        for k, dte in enumerate(px_oos.index):
            if k > 0:
                for c in ["XLP", "QQQ"]:
                    hold[c] *= (1 + rets.loc[dte, c])
            if k == 0:
                rows[dte] = {"XLP": 1.0, "QQQ": 0.0}
            if dte in exec_to_idx:
                i = exec_to_idx[dte]
                dec = decisions[i]
                total = hold["XLP"] + hold["QQQ"]
                pr = {m: probs[m].get(i) for m in models}
                act = rule(dec, pr)
                action_txt, step = "none", 0.0
                if act is not None:
                    direction, frac = act
                    if direction > 0 and hold["XLP"] > 0 and frac > 0:
                        move = frac * hold["XLP"]
                        hold["XLP"] -= move; hold["QQQ"] += move
                        step = move / total; n_rot += 1
                        action_txt = f"rotate XLP->QQQ frac_of_XLP={frac:.6f}"
                    elif direction < 0 and hold["QQQ"] > 0 and frac > 0:
                        move = frac * hold["QQQ"]
                        hold["QQQ"] -= move; hold["XLP"] += move
                        step = move / total; n_rot += 1
                        action_txt = f"rotate QQQ->XLP frac_of_QQQ={frac:.6f}"
                if dte in exec_to_idx and (k == 0 or act is not None):
                    rows[dte] = {"XLP": hold["XLP"] / total, "QQQ": hold["QQQ"] / total}
                elif act is not None:
                    rows[dte] = {"XLP": hold["XLP"] / total, "QQQ": hold["QQQ"] / total}
                records.append({
                    "decision_index": int(i),
                    "reference_month": str(dec["ref_month"].date()),
                    "knowable_date": str(dec["knowable"].date()),
                    "execution_date": str(dte.date()),
                    "feature_date": str(dec["feature_date"].date()),
                    "feature_stamp_ok": bool(dec["feature_date"] <= dec["knowable"]),
                    "cpi_yoy": dec["cpi_yoy"],
                    "cpi_trigger": bool(dec["cpi_yoy"] > 0.04),
                    "label": dec["label"],
                    "prob_logistic": pr["logistic"],
                    "prob_gbm": pr["gbm"],
                    "action": action_txt,
                    "step_fraction_of_nav": float(step),
                    "weight_xlp_after": float(hold["XLP"] / total),
                    "weight_qqq_after": float(hold["QQQ"] / total),
                })
        w = pd.DataFrame(rows).T.reindex(columns=["XLP", "QQQ"]).sort_index()
        w.index = pd.to_datetime(w.index)
        bt = run_backtest(px_oos, w, costs_bps=10)
        return bt, records, n_rot

    def stats_block(bt: pd.DataFrame, n_rot: int) -> dict:
        r = bt["strat"]
        nav = bt["nav"]
        n = len(r)
        total = float(nav.iloc[-1] - 1)
        cagr = float(nav.iloc[-1] ** (252.0 / n) - 1)
        sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if r.std(ddof=1) > 0 else None
        dd = eng_stats.max_drawdown(nav)
        return {
            "total_return_pct": total * 100,
            "cagr_pct": cagr * 100,
            "sharpe": sharpe,
            "max_drawdown_pct": float(dd["max_drawdown"]) * 100,
            "max_drawdown_peak": dd["dd_start"], "max_drawdown_trough": dd["dd_trough"],
            "n_rotations": int(n_rot),
            "total_turnover": float(bt["turnover"].sum()),
            "deflated_sharpe": eng_stats.deflated_sharpe(r, n_trials_dsr),
            "n_days": int(n),
        }

    arms: dict = {}
    arm_bts: dict = {}

    def add_arm(key: str, rule, model_for_metrics: str | None):
        bt, records, n_rot = run_arm(rule)
        blk = stats_block(bt, n_rot)
        if model_for_metrics:
            blk["classifier_auc"] = clf_metrics[model_for_metrics]["auc"]
            blk["classifier_brier"] = clf_metrics[model_for_metrics]["brier"]
        arms[key] = {"stats": blk, "decision_records": records}
        arm_bts[key] = bt
        print(key, {k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in blk.items() if k in
                    ("total_return_pct", "sharpe", "max_drawdown_pct", "n_rotations", "deflated_sharpe")})

    trig = lambda dec: dec["cpi_yoy"] > 0.04  # noqa: E731
    add_arm("arm_a_original", lambda dec, pr: (1, 0.10) if trig(dec) else None, None)
    for m in models:
        add_arm(f"arm_b_replace_{m}",
                lambda dec, pr, m=m: (1, 0.10) if pr[m] >= 0.5 else None, m)
        add_arm(f"arm_b_gate_{m}",
                lambda dec, pr, m=m: (1, 0.10) if (trig(dec) and pr[m] >= 0.5) else None, m)
        add_arm(f"arm_c_sizing_{m}",
                lambda dec, pr, m=m: (1, 0.20 * pr[m]) if trig(dec) else None, m)
        add_arm(f"arm_d_twoway_{m}",
                lambda dec, pr, m=m: (1, 0.10) if pr[m] >= 0.5 else (-1, 0.10), m)

    # benchmarks (no trials)
    bench_stats = {}
    for col, key in [("XLP", "bench_always_xlp"), ("QQQ", "bench_always_qqq")]:
        w = pd.DataFrame({col: [1.0]}, index=[px_oos.index[0]])
        bt = run_backtest(px_oos[[col]], w, costs_bps=10)
        bench_stats[key] = stats_block(bt, 0)
        arm_bts[key] = bt
        print(key, round(bench_stats[key]["total_return_pct"], 4), round(bench_stats[key]["sharpe"], 4))

    # frozen bar verdicts
    a = arms["arm_a_original"]["stats"]
    q = bench_stats["bench_always_qqq"]
    verdicts = {}
    for key, entry in arms.items():
        if key == "arm_a_original":
            continue
        s = entry["stats"]
        beats_a = s["sharpe"] > a["sharpe"] and s["total_return_pct"] > a["total_return_pct"]
        beats_q = s["sharpe"] > q["sharpe"] and s["total_return_pct"] > q["total_return_pct"]
        verdicts[key] = {
            "beats_arm_a_on_both": bool(beats_a), "beats_always_qqq_on_both": bool(beats_q),
            "verdict": "IMPROVEMENT" if (beats_a and beats_q) else "ML does not improve the strategy",
        }

    # ---- outputs ----
    results = {
        "experiment": "ML Target 7 — CPI hedge (S2) optimization test",
        "spec": "files/ml_target7_cpi_hedge_experiment_spec_2026-10-31.md (ML goal), frozen 2026-10-31 before fitting",
        "seed": SEED,
        "features": FEATURES,
        "n_decisions_total": len(decisions),
        "n_triggers_total": int(sum(1 for d in decisions if d["cpi_yoy"] > 0.04)),
        "first_test_decision_index": int(first_test),
        "oos_start_execution": str(oos_start.date()),
        "oos_window": [str(px_oos.index[0].date()), str(px_oos.index[-1].date())],
        "n_oos_decisions": len(oos_decisions),
        "n_labelled_oos_decisions": int(sum(1 for i in oos_decisions if decisions[i]["label"] is not None)),
        "ledger_max_row_before_booking": int(ledger_max),
        "n_trials_dsr": int(n_trials_dsr),
        "single_class_fallback_decisions": [int(i) for i in fallback_used],
        "classifier_metrics_by_model": clf_metrics,
        "logistic_selected_C_by_decision": {str(i): hypers["logistic"][i] for i in sorted(hypers["logistic"])},
        "arms": {k: {"stats": v["stats"], "decision_records": v["decision_records"]}
                 for k, v in arms.items()},
        "benchmarks": bench_stats,
        "verdicts": verdicts,
        "decisions_all": [{
            "decision_index": int(i), "reference_month": str(d["ref_month"].date()),
            "knowable_date": str(d["knowable"].date()), "execution_date": str(d["execution"].date()),
            "feature_date": str(d["feature_date"].date()), "cpi_yoy": d["cpi_yoy"],
            "cpi_trigger": bool(d["cpi_yoy"] > 0.04), "label": d["label"],
            "label_rel_ret": d["label_rel_ret"],
            "features": d["features"],
        } for i, d in enumerate(decisions)],
    }
    out_json = REPORTS / "ml_target7_cpi_hedge_results_2026-10-31.json"
    _write_retry(out_json, json.dumps(results, indent=1, default=str))
    print("wrote", out_json)
    for key, bt in arm_bts.items():
        df = pd.DataFrame({"net_return": bt["strat"], "nav_index": bt["nav"],
                           "turnover": bt["turnover"]})
        p = REPORTS / f"ml_target7_{key}_daily_2026-10-31.csv"
        _write_retry(p, df.to_csv(index_label="date"))
        print("wrote", p)


if __name__ == "__main__":
    main()
