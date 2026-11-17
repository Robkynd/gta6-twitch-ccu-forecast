# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Robkynd/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Robkynd/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecast GTA VI's launch-day Twitch viewers hour by hour — a tier-weighted
comparable-kernel ensemble. Built for the make.fun competition
**gta6-twitch-ccu-forecast-01** ($250, top 20 paid, closes
2026-11-19T01:00:00Z).

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only — nothing to install. Reads every
`<id>.in` in the input directory, writes `<id>.out` (23 rows, hours 2–24,
five quantiles each) to the output directory. Deterministic: two runs are
byte-identical. ~0.07 s per case, far inside the 8 s limit.

## Method

1. Anchor on hour 1's print `p1` (the case's `known` row).
2. For every comparable launch, take its release UTC day (`day_start` + 0–23h)
   and form log ratios `ln(viewers_h / viewers_hour1)`.
3. Filter out launches that cannot anchor on midnight the way the target does:
   - release-day peak / hour-1 > 20 — single global unlock later in the UTC
     day (Silksong, Black Myth: Wukong, Battlefield 6, …): their hour 1 is
     pre-unlock, so their ratios explode to 100–2000×;
   - `Tier: early-access-contaminated` in the case's own unlock note
     (Hogwarts Legacy, Diablo IV, Starfield, …): already playing for days,
     so their ratios are compressed toward 1;
   - `Tier: target-history` / the target game's own category row.
4. Center: per-hour weighted median of the log ratios, Gaussian kernel
   σ=0.8 on `|ln(hour1_comp) − ln(p1)|` — launches that opened at a similar
   scale behave alike. `q50_h = p1 · exp(median_h)`.
5. Spread: per-hour weighted IQR/1.349 of the same log ratios (floor 0.05,
   cap 1.0); `q_τ = q50 · exp(z_τ · s_h)`.

Any failure in the model path falls back to the house baseline (flat `p1`,
spread `0.15·√(h−1)`), so the program always emits a valid output.

## Backtest (practice launches)

Ground truth: SullyGnome hourly "Average viewers" for each launch's release
day (`answers/answers.json`; anchor verified against each case's
`history_hourly` tail). Score: mean pinball loss on the log scale, lower is
better.

| model | mean | p00 Wolverine | p01 RDR2 | p02 GoW | p03 Spider-Man 2 |
|---|---:|---:|---:|---:|---:|
| house baseline | 0.3871 | 0.7437 | 0.2868 | 0.2146 | 0.3033 |
| **this model** | **0.2004** | 0.3150 | 0.1418 | 0.1477 | 0.1969 |

90% interval coverage 0.78–0.96 (mean 0.89, nominal 0.90). Rejected after
backtesting: pre-launch-shape similarity kernel (0.23–0.34), LOO bias
correction (mixed per case), shrinkage toward persistence (worse), hour
smoothing (worse), hard top-K nearest (unstable across K).

Reproduce with `python3 backtest.py` and `python3 backtest2.py`
(`answers/answers.json` holds the reconstructed practice answers).

## Files

- `main.py` — the entry program (self-contained, stdlib only).
- `backtest.py`, `backtest2.py` — scorer + model-variant backtests.
- `answers/answers.json` — reconstructed practice answers with provenance.
- `muse-usage-gta6.jsonl` — token-usage records for the build session.
