# Paper digest — "Value and Momentum Everywhere"

**Citation:** Asness, Clifford S., Tobias J. Moskowitz, and Lasse Heje Pedersen (2013),
"Value and Momentum Everywhere," *Journal of Finance* 68(3), 929–985.
(AQR authors; won the Smith Breeden Prize.)

## The tradable idea (plain English)

Two of the oldest anomalies in finance — **value** (buy what's cheap, it mean-reverts)
and **momentum** (buy what's been going up, trends persist) — work **in every
asset class**, not just stocks. Better: value and momentum are *negatively
correlated* with each other (roughly −0.3 to −0.5 within each class), so a
50/50 combo earns a higher Sharpe than either leg alone. The paper documents
this across 8 asset classes and 48 instruments, 1972–2011, with a combined
long-short Sharpe around **1.0–1.45** after vol-scaling.

## Data used (paper)

48 liquid instruments: 9 country equity indices, 10 government bond indices,
6 currencies, 24 commodities (all futures/forward-based, excess returns).

## Methodology (the formulas)

For each asset *i* at month-end *t*:

- **Value** (non-equities): `V_i(t) = −(P_{t−12} / P_{t−60} − 1)` — the negative
  of the past 5-year return (De Bondt–Thaler long-term reversal). For equities:
  book-to-market.
- **Momentum**: `M_i(t) = P_{t−2} / P_{t−12} − 1` — cumulative return over months
  *t−12..t−2* ("12−1", Jegadeesh–Titman).
- Cross-sectionally standardize each: `z^V_i`, `z^M_i`; combo score
  `s_i = z^V_i + z^M_i` (50/50).
- Long the top tercile of *s_i*, short the bottom tercile, equal-weighted.
  Rebalance monthly. Portfolios are **volatility-scaled** to a common target
  before combining across classes.

## What we implemented (os-quant Strategy 1)

- Universe: 11 liquid US-listed ETFs (5 equity, 4 rates, 2 credit) — the
  free-data tradeable proxy for the paper's futures universe.
- Value = −(48m return ending 12m before formation); momentum = 11m return
  ending 1m before formation; z-scored combo; long top third / short bottom
  third; $0.50 long / $0.50 short per $1 NAV; monthly rebalance with a
  1-trading-day execution lag.
- FI sleeve: identical construction on TLT/IEF/SHY/TIP/LQD/HYG, reported separately.

## What was simplified (honest deviations)

1. **ETF proxies, not futures.** No leverage, no carry/roll yield, ETF fees
   (~5–15 bps/yr drag). The paper's returns are *excess* returns on futures.
2. **No book-to-market for equities.** No free fundamentals feed in phase 1, so
   the 5y-reversal value proxy is used for *all* assets (as the paper does for
   non-equity classes).
3. **11 instruments, not 48.** Breadth is the diversification engine of this
   strategy; 11 ETFs across 3 sleeves is much narrower than 8 asset classes.
4. **No vol-scaling across sleeves** (doesn't change Sharpe, but changes the
   combo's risk contribution).
5. **Shorting assumed at 50 bps/yr borrow.** Real HYG borrow can gap wider
   in stress; costs are 10 bps one-way (liquid-ETF assumption).
6. **Sample is 2012–2026**, vs the paper's 1972–2011. Regimes differ enormously
   (ZIRP, then the 2022 hiking shock).

## Result in our sample

2012-06 → 2026-10, net of costs: **Sharpe −0.55, CAGR −3.98%, max DD −51%**
(full book); FI sleeve **Sharpe −0.19, CAGR −0.85%**. Gross (pre-cost) Sharpe
≈ −0.02 — costs are not the story; the signals are flat-to-negative here.

## Known limitations / interview talking points

- **The value leg fights the mega-cap regime.** 2012–2026 value = short the
  strongest 5y performers (US large-cap growth) and long the weakest (EM,
  duration in drawdown). That's structurally painful in this sample.
- **Breadth deficit.** With 11 assets, terciles are 3 names per leg — one
  idiosyncratic blowup dominates. The paper's result leans on 48 instruments.
- **No carry.** In rates/FX, carry is often *the* value proxy that works;
  our price-only value misses it. Koijen et al. (2018) "Carry" is the natural
  next paper for the FI sleeve.
- **What I'd test next:** (a) vol-scaled combo, (b) dropping the equity sleeve
  (where we lack B/M), (c) carry-augmented value in rates, (d) the 1972–2011
  futures sample for an apples-to-apples replication check.
