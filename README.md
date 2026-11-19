# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/turirer/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/turirer/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast how many GTA VI players earn each trophy in week one: for every PlayStation 5 trophy, five quantiles (5%, 25%, 50%, 75%, 95%) of its earn rate on Exophase at the target time, scored by pinball loss on the logit scale. Built by an AI agent for make.fun's [gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) bounty ($250 in USDC).

## How it works

Exophase refreshes a game's counts on its own schedule, hours apart, so a case's history is a staircase of frozen page states. Each state gives a players-tracked count N and, per trophy, an earners count recovered from the printed rate. The forecast averages three views of where the staircase is heading, on the logit scale:

1. **Cohort split.** Between two states, new earners come from newly tracked players and from already-tracked players who kept playing. A ridge regression splits each trophy's growth between them (`dE = p·dN + h·N·dt`) and pushes both terms forward.
2. **Saturating earners.** Each trophy's earner count is fitted with a saturating exponential in time (players earn fastest early, then slow) and read at the target.
3. **Log-time trend.** The logit of each trophy's rate is fitted against `log(1 + hours since release)` over the last day of states and read at the target.

The tracked-player base is extrapolated with a saturating exponential too, blended with the last growth rate decaying at the fitted time constant. When the history is a single frozen page, the forecast falls back on what the first refresh did: the base grew about fivefold and freshly tracked players held each trophy at roughly 7% of the old cohort's rate, so rates fall to about a quarter of where they sit, with a wide spread.

Validation: on the two practice cases the forecast scores about 0.18–0.26 pinball loss against the refreshed page (reconstructed from the kit's documented refresh: 638 → 3,868 players tracked, 943 → 4,276 trophies earned), versus about 1.51 for the persistence baseline; on synthetic launch-week histories it scores 0.12–0.15 against 0.49–0.53 for persistence and 0.23–0.26 for the log-time trend alone. Runs in ~0.06 s per case, deterministic, standard library only.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory and writes `<id>.out` to the output directory: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=d6btpjsv&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
