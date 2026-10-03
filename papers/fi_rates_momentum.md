# Paper digest — FI rates time-series momentum

**Citations:**
- Moskowitz, Tobias J., Yao Hua Ooi, and Lasse Heje Pedersen (2012),
  "Time Series Momentum," *Journal of Financial Economics* 104(2), 228–250.
- The rates application is the standard FI-desk implementation of MOP's
  TSMOM, and the vol-targeting follows the CTA industry practice the paper
  discusses.

## The tradable idea (plain English)

Assets that outperformed cash over the past 12 months (skipping the most
recent month) tend to keep outperforming for the next month — and
underperformers keep underperforming. Unlike cross-sectional momentum (rank
assets against each other), **time-series momentum** trades each asset on the
sign of its *own* past excess return. MOP (2012) find it in 58 liquid
instruments, 1985–2009, with a diversified TSMOM portfolio Sharpe near 1.0.
Every rates desk runs a version of this on duration: long bonds when the
trend is up, short (or flat) when it's down.

## Data used (paper)

58 futures/forwards: 9 equity indices, 10 bond markets, 27 commodities,
12 currency pairs. Excess returns over T-bills.

## Methodology (the formulas)

For each asset, at month-end *t*:

- Signal: `mom_i(t) = sign( ∏_{k=t−12}^{t−2} (1 + r^{excess}_{i,k}) − 1 )`
- Position: `w_i(t) = sign × σ_target / σ_{i,60d}` — volatility-targeted so
  each asset contributes a stationary risk budget; the paper scales the
  *portfolio* to 40% vol for reporting, practitioners use 5–15% per asset.
- Hold one month, rebalance monthly.

## What we implemented (os-quant Strategy 2)

- Single-asset: 12−1m excess return of **TLT over SHY** (cash); sign → ±.
- Position = `sign × 10% / σ_60d(TLT−SHY excess)`, capped at ±1.5x notional.
  Monthly rebalance, 1-trading-day execution lag, 5 bps one-way costs.
- Benchmark: monthly-rebalanced 50/50 TLT/IEF (free-data proxy for the
  Bloomberg US Treasury index).

## What was simplified (honest deviations)

1. **One asset, not 58.** MOP's result is a *diversified* portfolio; single-asset
   TSMOM is far noisier. This is the single biggest deviation.
2. **ETF total returns**, not futures excess returns (no funding/roll mechanics,
   though TLT−SHY spread approximates the financed trade).
3. **10% vol target** is a choice, not the paper's number (paper reports at
   40% portfolio vol for the diversified book).
4. **No stop-loss / regime filter.** Practitioners often add them; the paper
   doesn't.

## Result in our sample

2003-10 → 2026-10, net of costs: **Sharpe −0.14, CAGR −2.20%, max DD −54.9%**,
45% monthly win rate. Gross ≈ same — not a cost story. Negative in *every*
subperiod (2003–10: −0.29; 2011–19: +0.03; 2020–26: −0.12). Flipping the
signal gives +0.12 — i.e. the signal is weak noise, not inverted (verified).

## Known limitations / interview talking points

- **Whipsaw in regime breaks.** The 12m lookback still holds the pre-crash rally
  when a hiking cycle starts (2022: the signal flipped long/short almost
  monthly). This is the known failure mode of 12−1 TSMOM at turning points.
- **Single-asset concentration.** MOP's Sharpe comes from diversification
  across 58 markets; one duration trade can't replicate that.
- **The 2022 problem.** Vol-targeting sizes *down* into the crash but can't
  fix a signal pointing the wrong way; realized vol lags the regime break.
- **What I'd test next:** (a) TSMOM across the full 11-ETF panel (diversified,
  like the paper), (b) shorter lookbacks (3m/6m) blended with 12m, (c) a
  "don't fight the Fed" regime filter using the hiking-cycle dummy,
  (d) carry (term-structure slope) as a confirming signal for rates.
