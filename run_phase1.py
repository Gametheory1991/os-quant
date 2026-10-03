"""Phase 1 backtest runner: builds signals, runs both strategies (+ FI sleeve),
computes the institutional stats sheet, and saves equity-curve charts.

Usage:  .venv/bin/python run_phase1.py [--refresh]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.backtest import run_backtest
from engine.config import load_config
from engine.data import load_universe_prices
from engine.stats import format_report, report
from strategies.fi_rates_momentum import fi_rates_momentum_weights, treasury_benchmark
from strategies.value_momentum import value_momentum_weights

ROOT = Path(__file__).resolve().parent
CHART_DIR = ROOT / "reports" / "charts"


def load_prices(refresh: bool) -> pd.DataFrame:
    from engine.data import load_universe
    tickers = load_universe("etfs")["tickers"]
    return load_universe_prices(tickers, refresh=refresh)


def run_value_momentum(prices: pd.DataFrame, cfg: dict, sleeve: list[str] | None = None):
    sig = cfg["signal"]
    w = value_momentum_weights(
        prices,
        tickers=sleeve,
        value_window=sig["value_window_months"],
        value_skip=sig["value_skip_months"],
        mom_window=sig["momentum_window_months"],
        mom_skip=sig["momentum_skip_months"],
    )
    start = cfg["fi_sleeve_start"] if sleeve else cfg["start"]
    w = w[w.index >= start]
    px = prices[w.columns]
    bt = run_backtest(px, w, costs_bps=cfg["costs_bps"],
                      borrow_bps_annual=cfg["borrow_bps_annual"])
    bt = bt.loc[w.index[0]:]  # drop the pre-inception flat zeros
    bench = prices["SPY"].pct_change()
    stats = report(bt["strat"], benchmark=bench, turnover=bt["turnover"],
                   label=cfg["name"] + (" [FI sleeve]" if sleeve else ""))
    return bt, stats


def run_fi_momentum(prices: pd.DataFrame, cfg: dict):
    sig = cfg["signal"]
    w = fi_rates_momentum_weights(
        prices,
        long_ticker=sig["long_ticker"], cash_ticker=sig["cash_ticker"],
        target_vol=sig["target_vol_annual"], max_leverage=sig["max_leverage"],
    )
    w = w[w.index >= cfg["start"]]
    bt = run_backtest(prices[[sig["long_ticker"]]], w, costs_bps=cfg["costs_bps"])
    bt = bt.loc[w.index[0]:]  # drop the pre-inception flat zeros
    bench_px = treasury_benchmark(prices)
    bench = bench_px.pct_change().reindex(bt.index).fillna(0.0)
    stats = report(bt["strat"], benchmark=bench, turnover=bt["turnover"], label=cfg["name"])
    return bt, stats


def chart(nav_dict: dict[str, pd.Series], title: str, path: Path):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, nav in nav_dict.items():
        ax.plot(nav.index, nav.values, label=name, linewidth=1.5)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("NAV (log scale)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("chart:", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    print("loading prices...")
    prices = load_prices(args.refresh)
    print(f"panel: {prices.shape[0]} days x {prices.shape[1]} assets")

    results = {}

    cfg1 = load_config("value_momentum_everywhere")
    print("\n-- Strategy 1: full book --")
    bt1, s1 = run_value_momentum(prices, cfg1)
    print(format_report(s1))
    results["s1"] = (bt1, s1)

    print("\n-- Strategy 1: FI sleeve --")
    fi_tickers = cfg1["universe_def"]["sleeves"]["fixed_income"]
    bt1fi, s1fi = run_value_momentum(prices, cfg1, sleeve=fi_tickers)
    print(format_report(s1fi))
    results["s1fi"] = (bt1fi, s1fi)

    cfg2 = load_config("fi_rates_momentum")
    print("\n-- Strategy 2: FI rates momentum --")
    bt2, s2 = run_fi_momentum(prices, cfg2)
    print(format_report(s2))
    results["s2"] = (bt2, s2)

    chart({"V&M everywhere": bt1["nav"], "V&M FI sleeve": bt1fi["nav"],
           "SPY": (1 + prices["SPY"].pct_change().reindex(bt1.index).fillna(0)).cumprod()},
          "Strategy 1 — Value & Momentum Everywhere (net of costs)",
          CHART_DIR / "s1_equity_curve.png")
    chart({"FI rates momentum": bt2["nav"],
           "50/50 TLT/IEF": (1 + treasury_benchmark(prices).pct_change().reindex(bt2.index).fillna(0)).cumprod()},
          "Strategy 2 — FI Rates Time-Series Momentum, 10% vol target (net of costs)",
          CHART_DIR / "s2_equity_curve.png")

    # persist stats dicts (minus DataFrames) for the report writer
    import json
    summary = {}
    for k, (bt, s) in results.items():
        d = {kk: vv for kk, vv in s.items() if kk not in ("monthly_pnl", "yearly_returns")}
        d["monthly_pnl"] = s["monthly_pnl"].to_dict()
        d["yearly_returns"] = {str(kk): vv for kk, vv in s["yearly_returns"].items()}
        summary[k] = d
    (ROOT / "reports" / "phase1_stats.json").write_text(json.dumps(summary, indent=1, default=str))
    print("\nstats -> reports/phase1_stats.json")


if __name__ == "__main__":
    main()
