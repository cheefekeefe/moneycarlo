# MoneyCarlo

A Monte Carlo simulation engine for NFL games. It simulates every game thousands of
times, drive by drive, and summarises the results into:

- **Game predictions:** win probability, projected final score, spread (projected margin), total,
  most likely final scores and full score distributions.
- **Player stat lines (offense):** completions, attempts, passing yards/TDs/INTs, carries,
  rushing yards/TDs, targets, receptions, receiving yards/TDs, each with a 10th–90th percentile
  range and the chance of at least one TD.
- **Memory:** every prediction is stored before kickoff, actual results are attached afterwards,
  and the stored track record is used to score and re-tune the model.

All data is free, from [nflverse](https://github.com/nflverse) via `nflreadpy`. No betting data is used.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .

moneycarlo predict          # downloads data (first run ~1 min), simulates the next week
moneycarlo dashboard        # opens the dashboard in your browser
```

Recommended once, to build a historical track record and tune the model:

```bash
moneycarlo backtest --seasons 2025     # walk-forward predictions for every 2025 week (~3 min)
moneycarlo calibrate                   # tune simulation spread against recent seasons (~8 min)
```

## Weekly routine

| When | Command | What it does |
|---|---|---|
| Before games (after injury reports) | `moneycarlo predict` | Refreshes data, retrains on everything played so far, simulates the upcoming week and stores it |
| After games | `moneycarlo update` | Pulls final scores and box scores and attaches them to stored predictions |
| Any time | `moneycarlo evaluate` | Prints the live track record (game and player accuracy) |
| Every few weeks | `moneycarlo calibrate` | Re-tunes the simulation's spread parameters from results |

Useful options: `predict --week 9 --sims 20000 --seed 1`, `predict --no-update` (skip download),
`evaluate --kind backtest`.

## How it works

```
nflverse play-by-play ──► drives + team box scores ──► pre-game team ratings (no look-ahead)
                                                            │
                     drive-outcome model (multinomial logistic regression)
                                                            │
     Monte Carlo: drives played in order ──► scores ──► team box score ──► player stat lines
                                                            │
                             DuckDB: predictions → results → evaluation → calibration
```

1. **Team ratings** (`moneycarlo/nfl/ratings.py`). For every team and game: offensive and defensive EPA/play,
   success rate, TD / FG / turnover rate per drive, plays per drive, sack rate, completion %, yards per
   completion, yards per carry, and a game-script-neutral pass rate. Ratings are opponent adjusted, weight
   recent games more (decay 0.90 per game), shrink toward league average, and regress 45% toward the mean
   between seasons. Each game only uses games played before it.
2. **Drive model** (`moneycarlo/nfl/model.py`). A multinomial logistic regression predicts P(touchdown,
   field goal, turnover, no score) for a drive from the offense's ratings, the defense's ratings, home
   field, dome and rest. It's trained on ~10 seasons of drives, with older seasons down-weighted (×0.75 per
   season) because the league changes. A pace model predicts drives per game.
3. **Simulation** (`moneycarlo/nfl/simulate.py`). For each of 10,000 runs: draw the number of drives, then play
   drives in order, alternating possession. Each drive outcome depends on the matchup *and the current score*:
   leading teams get more conservative and trailing teams press. Points include PATs, 2-point tries,
   defensive and return TDs, and overtime. Scoring by drive naturally produces the NFL's key numbers (3, 7).
   A team box score is built from the same simulated game: plays from drive outcomes, pass/run mix from
   team tendency and game script, and yards from matchup efficiency.
4. **Players** (`moneycarlo/nfl/players.py`). Recent target and carry shares, plus catch rate, yards per
   catch/carry and TD rates shrunk toward position averages. Players listed Out or Doubtful are removed and
   their usage goes to teammates. The starting QB comes from the nflverse depth chart. Each simulated team
   box score is split among players, so player lines always add up to the team's totals in the same
   simulated game. A QB having a big day means his receivers did too.
5. **Memory** (`moneycarlo/db.py`, `data/moneycarlo.duckdb`). Tables: `runs`, `game_predictions`,
   `score_distributions`, `team_predictions`, `player_predictions`, `game_results`, `team_results`,
   `player_results`, `model_params`. Every `predict` retrains on all completed games, and `calibrate` grid-searches
   the simulation's spread parameters against past results and saves the winner to `model_params` for future runs.

## 2025 backtest (walk-forward, all 285 games)

| | |
|---|---|
| Winners called | 59–61% (varies slightly by random seed) |
| Brier score | 0.226 (0.25 = coin flip) |
| Margin MAE | 10.2 points |
| Total MAE | 10.6 points |
| Actual margin / total inside predicted 80% range | 78% / 82% (target 80%) |
| Team stat bias (points, yards, attempts, carries…) | within ~1% of actuals |
| Player receiving / rushing bias | ≈ 0 (e.g. rec yds +0.2, carries −0.0) |

For reference, the Vegas closing line usually gets about 66% of winners and a margin MAE near 10. The model
doesn't use market data, so a gap is expected. Backtests use injury reports but not historical depth charts;
when the projected QB actually started, QB attempt projections were unbiased.

## Dashboard

- **Slate:** every game this week with projected score, winner and win %, spread, total, range and most
  likely score; finals and ✅/❌ once games are played.
- **Game detail:** margin and total distributions, a most-likely-scores table, a score heatmap, team box
  score projections, and passing / rushing / receiving projections per player (with actuals after the game).
- **Model performance:** winner accuracy, Brier score, calibration curve, accuracy by week, and player stat
  accuracy, for live and backtest predictions.

## Project layout

```
moneycarlo/
  config.py          paths + SimParams (all tunable knobs)
  db.py              DuckDB schema and storage
  evaluate.py        scoring predictions vs results
  cli.py             `moneycarlo` command
  nfl/
    data.py          nflreadpy download + parquet cache (data/cache/)
    ratings.py       pre-game team ratings
    model.py         drive-outcome and pace models
    simulate.py      vectorised drive-level game simulation
    players.py       usage profiles + allocation to players
    pipeline.py      predict / backtest / calibrate / results
dashboard/app.py     Streamlit dashboard
tests/               unit tests (`pip install -e .[dev] && pytest`)
```

## Known limitations / next steps

- **QB changes.** Ratings are team-level, so a backup QB starting doesn't lower the team's efficiency yet.
  A QB-adjusted offensive rating is the single biggest improvement available.
- **Weather.** Only dome vs. outdoors is used, because nflverse has no temperature or wind for future games. A
  free forecast (Open-Meteo) for outdoor stadiums would help totals and passing.
- **Injuries beyond the QB.** Out/Doubtful players are removed from player projections but don't change team
  strength (e.g. a missing left tackle or WR1).
- **Calibration scope.** `calibrate` tunes game-level spread. Player-level noise parameters in
  `SimParams` are set by hand from backtest coverage and could be tuned the same way.
- **NBA.** The storage, evaluation, calibration and dashboard layers are sport-agnostic. An NBA module would add
  a possession-level simulator under `moneycarlo/nba/`.
