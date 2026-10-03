# os-quant

A systematic research engine: **academic papers → tradable implementation → backtest → institutional stats**.
Built so every strategy is a documented, reproducible artifact — the kind of thing you can walk a
quant fund through in an interview.

## What it does

- Ingests classic quant papers (starting with Asness–Moskowitz–Pedersen 2013 and MOP 2012)
- Implements them on free/keyless data (Yahoo Finance daily bars; FRED-ready)
- Backtests with honest accounting (turnover costs, stock borrow, execution lag)
- Produces the full institutional stats sheet (Sharpe, Sortino, max DD, VaR, beta/alpha, IR, win/loss, profit factor, monthly P&L)
- Ships a weekly-style report with equity curves and paper digests

Phase 1 result (honest): both flagship constructions are flat-to-negative on the
free-data ETF translation in these samples — documented with exact deviations from
the papers. The engine itself is validated on synthetic data with a planted edge
(Sharpe 3.93). Negative results, cleanly attributed, are the product working.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run_phase1.py            # downloads data (cached), runs both strategies
.venv/bin/python -m pytest tests/ -q     # 17 unit tests
```

## Adding a strategy (the 4-file pattern)

1. **`universes/<name>.yaml`** — ticker list + sleeves (see `universes/etfs.yaml`).
2. **`strategies/<name>.py`** — pure signal function returning a DataFrame of target
   weights indexed by rebalance date (month-end + 1 trading day execution lag).
3. **`configs/<name>.yaml`** — universe, signal params, costs, borrow, benchmark, start.
4. **`papers/<name>.md`** — the digest: citation, tradable idea, methodology with
   formulas, what was simplified, limitations, interview talking points.

Then wire it into `run_phase1.py` (or your own runner) and add the report section.

## Layout

```
engine/        data.py (Yahoo/FRED adapters + parquet cache),
               backtest.py (vectorized daily, costs/borrow/turnover),
               stats.py (institutional stats sheet), config.py
strategies/    signal construction (no I/O, pure functions of price panels)
universes/     ticker universes in YAML
configs/       per-strategy parameters
papers/        paper digests — interview-grade
reports/       dated reports + equity-curve charts
tests/         unit tests (hand-computed fixtures, fixture-based I/O, no network)
data/cache/    local parquet cache (gitignored)
```

## Conventions

- Prices are **adjusted closes** (total return). NaN = untradeable.
- Weights are fractions of NAV; positions held between rebalance dates.
- Costs in bps one-way per unit turnover; borrow in bps/yr on short notional.
- Free data only. Polite HTTP: 0.6s pacing, one 429 retry with backoff.
- Every deviation from a source paper is documented in its digest.
