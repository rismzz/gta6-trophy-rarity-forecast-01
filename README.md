# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rismzz/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rismzz/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast every Grand Theft Auto VI PlayStation 5 trophy's earn rate on
Exophase at 2026-11-26 00:00 UTC, as five quantiles (5%, 25%, 50%, 75%, 95%)
per trophy, scored by pinball loss on the logit scale. One file, `main.py`
(Python 3.13, standard library only, no network, deterministic).

## The model

Counts, not rates. A trophy's earn rate is E / N: E, the players who earned
it, over N, the players Exophase tracks. Both counts only grow, and the
launch flood grows N far faster than E, so rates mostly fall early even as
players earn more trophies. Modelling E and N separately, then dividing,
captures that dilution; fitting the rate directly cannot.

1. **Distinct states.** Exophase refreshes its counts hours apart, so
   consecutive identical reads collapse to one observation each.
2. **Windowed count fit.** log E and log N are each fit by least squares on
   ln(1 + t), t the hours since 2026-11-19T00:00Z, over the last 36 hours of
   distinct states (the early launch-night cohort is not representative of
   the flood that followed). Each trophy's E-slope is shrunk toward the
   pooled slope across trophies; both forecasts are clamped monotone.
3. **Honest spreads.** A random-walk term growing with the horizon, plus the
   fit's residual scale, plus a refresh-jump allowance.
4. **Flat history.** When the history shows a single state (launch night),
   no trend is visible but a refresh is due, so the median shifts down by
   the expected refresh shock with a wide spread.
5. **Universal trophies.** A trophy every tracked player holds (100.00%) is
   forecast at 100 throughout.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Local verification

- Practice cases p00/p01 (answers reconstructed from the house's own rules:
  the 04:30Z refresh, 3,868 players tracked, every rate but 100% lower):
  pinball loss **0.461**, typical miss 0.23 points, 90% coverage 1.000 —
  vs the house baseline's 1.519 / 0.69 points / 0.000.
- Synthetic final-like cases (launch-week trajectories with Exophase-style
  refresh quantization, cut at the real final checkpoint, target a week
  after launch): beats both house baselines in every scenario tried
  (e.g. 0.234 vs 0.248 curve / 0.374 persistence; 0.231 vs 0.243 / 0.357).
- Determinism: two runs byte-identical. Runtime: ~0.1 s per case (limit 8 s).
  Output: 55 rows, five non-decreasing quantiles each, ~2.8 KB (limit 65,536).

---

<sub>Entered on [make.fun](https://make.fun/?ref=aa4mcy45&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
