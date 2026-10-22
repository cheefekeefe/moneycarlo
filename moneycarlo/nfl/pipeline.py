"""End-to-end: build features, fit, simulate, summarise, store; plus backtest and calibration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from moneycarlo import db
from moneycarlo.config import MODEL_VERSION, TRAIN_SEASONS, SimParams
from moneycarlo.nfl import data
from moneycarlo.nfl.model import NFLModel, matchup_frame
from moneycarlo.nfl.players import PLAYER_STATS, STAT_MAP, allocate, usage_profiles
from moneycarlo.nfl.ratings import compute_ratings, prepare_team_games
from moneycarlo.nfl.simulate import GameInputs, TeamInputs, simulate_game

log = logging.getLogger(__name__)

TEAM_STATS = ["points", "plays", "attempts", "completions", "sacks", "pass_yards", "carries",
              "rush_yards", "pass_td", "rush_td", "interceptions", "fumbles_lost"]


@dataclass
class Context:
    params: SimParams
    schedules: pd.DataFrame
    drives: pd.DataFrame
    team_games: pd.DataFrame
    ratings: pd.DataFrame
    matchups: pd.DataFrame
    script_beta: float
    player_stats: pd.DataFrame
    scoring: dict
    seasons: list[int]


def build_context(last_season: int, params: SimParams | None = None,
                  first_season: int | None = None) -> Context:
    params = params or SimParams()
    first = first_season if first_season is not None else last_season - TRAIN_SEASONS
    seasons = list(range(first, last_season + 1))
    schedules = data.load_schedules()
    sched = schedules[schedules["season"].isin(seasons)].reset_index(drop=True)
    drives, tg = data.load_history(seasons)
    log.info("Computing team ratings for %s-%s", seasons[0], seasons[-1])
    ratings, beta = compute_ratings(sched, tg, params)
    matchups = matchup_frame(sched, ratings)
    gameday = sched.set_index("game_id")["gameday"]
    drives = drives.assign(gameday=drives["game_id"].map(gameday))
    return Context(
        params=params, schedules=sched, drives=drives, team_games=tg, ratings=ratings,
        matchups=matchups, script_beta=beta,
        player_stats=data.load_player_stats(seasons[-4:]),
        scoring=data.load_scoring_constants(seasons), seasons=seasons,
    )


def fit_model(ctx: Context, cutoff: pd.Timestamp) -> NFLModel:
    """Fit on games played strictly before `cutoff`."""
    drives = ctx.drives[ctx.drives["gameday"] < cutoff]
    games = set(drives["game_id"])
    matchups = ctx.matchups[ctx.matchups["game_id"].isin(games)]
    return NFLModel().fit(drives, matchups, ctx.scoring, ctx.script_beta, ctx.params.train_season_decay)


def _team_inputs(off: pd.Series, probs: np.ndarray) -> TeamInputs:
    def combo(m):  # offense rating + opposing defense rating - league
        return off[f"off_{m}"] + off[f"def_{m}"] - off[f"lg_{m}"]

    return TeamInputs(
        drive_probs=probs,
        pass_rate_logit=off["off_pass_rate"],
        sack_rate=combo("sack_rate"),
        comp_pct=combo("comp_pct"),
        ypcomp=combo("ypcomp"),
        ypc=combo("ypc"),
        ppd_factor=combo("ppd") / off["lg_ppd"],
        int_share=off["off_int_share"],
        pass_td_share=off["off_pass_td_share"],
    )


def game_inputs(ctx: Context, model: NFLModel, games: pd.DataFrame) -> dict[str, GameInputs]:
    m = ctx.matchups[ctx.matchups["game_id"].isin(games["game_id"])].reset_index(drop=True)
    probs = model.drive_probs(m)
    out = {}
    for g in games.itertuples(index=False):
        rows = {}
        for side, team in (("home", g.home_team), ("away", g.away_team)):
            idx = m.index[(m["game_id"] == g.game_id) & (m["team"] == team)][0]
            rows[side] = (m.loc[idx], probs[idx])
        pace = sum(r["off_ppd"] + r["def_ppd"] - 2 * r["lg_ppd"] for r, _ in rows.values())
        out[g.game_id] = GameInputs(
            home=_team_inputs(*rows["home"]),
            away=_team_inputs(*rows["away"]),
            expected_drives=model.expected_drives(pace),
            drives_sd=model.drives_sd,
            ppd_by_outcome=model.ppd_by_outcome,
            p_def_td=model.p_def_td,
            p_st_td=model.p_st_td,
            scoring=model.scoring,
            script_beta=model.script_beta,
            playoff=g.game_type != "REG",
        )
    return out


# --------------------------------------------------------------------------- summaries


def _q(x, q):
    return float(np.quantile(x, q))


def summarise_game(g, sim: dict) -> tuple[dict, pd.DataFrame]:
    h, a = sim["home_pts"], sim["away_pts"]
    margin, total = h - a, h + a
    pairs = pd.Series(list(zip(h, a))).value_counts(normalize=True)
    mode = pairs.index[0]
    row = dict(
        game_id=g.game_id, season=g.season, week=g.week, game_type=g.game_type,
        gameday=g.gameday.date(), gametime=g.gametime, home_team=g.home_team, away_team=g.away_team,
        home_win_prob=float((margin > 0).mean()), away_win_prob=float((margin < 0).mean()),
        tie_prob=float((margin == 0).mean()),
        home_pts_mean=float(h.mean()), away_pts_mean=float(a.mean()),
        home_pts_median=float(np.median(h)), away_pts_median=float(np.median(a)),
        home_pts_p10=_q(h, .1), home_pts_p90=_q(h, .9), away_pts_p10=_q(a, .1), away_pts_p90=_q(a, .9),
        margin_mean=float(margin.mean()), margin_median=float(np.median(margin)),
        margin_sd=float(margin.std()), margin_p10=_q(margin, .1), margin_p90=_q(margin, .9),
        total_mean=float(total.mean()), total_median=float(np.median(total)),
        total_sd=float(total.std()), total_p10=_q(total, .1), total_p90=_q(total, .9),
        most_likely_score=f"{mode[0]}-{mode[1]}",
    )
    scores = pd.DataFrame(
        [(g.game_id, int(k[0]), int(k[1]), float(v)) for k, v in pairs.items()],
        columns=["game_id", "home_pts", "away_pts", "prob"],
    )
    return row, scores


def summarise_team(game_id: str, team: str, box: dict, points: np.ndarray) -> list[dict]:
    rows = []
    for stat in TEAM_STATS:
        x = points if stat == "points" else box[stat]
        rows.append(dict(game_id=game_id, team=team, stat=stat, mean=float(x.mean()), sd=float(x.std()),
                         p10=_q(x, .1), p50=_q(x, .5), p90=_q(x, .9)))
    return rows


def summarise_players(game_id: str, team: str, prof: pd.DataFrame, alloc: dict) -> list[dict]:
    rows = []
    info = prof.set_index("player_id")
    for pid, stats in alloc.items():
        for stat in PLAYER_STATS:
            if stat not in stats:
                continue
            x = stats[stat]
            qs = np.quantile(x, [.1, .25, .5, .75, .9])
            rows.append(dict(
                game_id=game_id, team=team, player_id=pid, player_name=info.at[pid, "name"],
                position=info.at[pid, "position"], stat=stat, mean=float(x.mean()), sd=float(x.std()),
                p10=qs[0], p25=qs[1], p50=qs[2], p75=qs[3], p90=qs[4], prob_pos=float((x > 0).mean()),
            ))
    return rows


# --------------------------------------------------------------------------- predict


def predict_week(
    ctx: Context, season: int, week: int, *, with_players: bool = True,
    injuries: pd.DataFrame | None = None, depth: pd.DataFrame | None = None, seed: int | None = None,
) -> dict[str, pd.DataFrame]:
    games = ctx.schedules[(ctx.schedules["season"] == season) & (ctx.schedules["week"] == week)]
    if games.empty:
        raise ValueError(f"No games found for {season} week {week}")
    model = fit_model(ctx, games["gameday"].min())
    inputs = game_inputs(ctx, model, games)
    rng = np.random.default_rng(seed)

    game_rows, score_rows, team_rows, player_rows = [], [], [], []
    for g in games.itertuples(index=False):
        sim = simulate_game(inputs[g.game_id], ctx.params, rng)
        row, scores = summarise_game(g, sim)
        game_rows.append(row)
        score_rows.append(scores)
        for side in ("home", "away"):
            team = getattr(g, f"{side}_team")
            box = sim[f"{side}_box"]
            if with_players:
                prof = usage_profiles(ctx.player_stats, team, season, week, injuries, depth)
                alloc = allocate(box, prof, ctx.params, rng)
                player_rows += summarise_players(g.game_id, team, prof, alloc)
            team_rows += summarise_team(g.game_id, team, box, sim[f"{side}_pts"])
    return {
        "games": pd.DataFrame(game_rows),
        "scores": pd.concat(score_rows, ignore_index=True),
        "teams": pd.DataFrame(team_rows),
        "players": pd.DataFrame(player_rows),
    }


def next_week(schedules: pd.DataFrame, season: int) -> int:
    s = schedules[(schedules["season"] == season) & ~schedules["completed"]]
    if s.empty:
        raise ValueError(f"No remaining games in {season}")
    return int(s["week"].min())


def run_week(con, ctx: Context, season: int, week: int, kind: str, with_players: bool = True,
             seed: int | None = None) -> str:
    injuries = data.load_injuries(season)
    depth = data.load_depth_charts(season) if kind == "live" else None
    out = predict_week(ctx, season, week, with_players=with_players, injuries=injuries, depth=depth, seed=seed)
    if kind == "backtest":
        db.delete_runs(con, "backtest", season, week)
    return db.save_run(
        con, kind=kind, season=season, week=week, model_version=MODEL_VERSION,
        params=ctx.params.to_dict(), games=out["games"], scores=out["scores"],
        teams=out["teams"], players=out["players"],
    )


# --------------------------------------------------------------------------- results


def sync_results(con, ctx: Context) -> int:
    """Store actual scores, team box scores and player stats for completed games."""
    done = ctx.schedules[ctx.schedules["completed"]]
    games = done[["game_id", "season", "week", "home_team", "away_team", "home_score", "away_score"]]
    tg = prepare_team_games(ctx.team_games, ctx.schedules)
    tg = tg[tg["game_id"].isin(done["game_id"])]
    teams = tg.melt(id_vars=["game_id", "team"], value_vars=TEAM_STATS, var_name="stat", value_name="value")
    ps = ctx.player_stats[ctx.player_stats["game_id"].isin(done["game_id"])]
    players = ps.rename(columns=STAT_MAP).melt(
        id_vars=["game_id", "team", "player_id"], value_vars=PLAYER_STATS, var_name="stat", value_name="value"
    )
    db.replace_results(con, games, teams.dropna(), players)
    return len(games)


# --------------------------------------------------------------------------- calibration


def crps(samples: np.ndarray, y: float) -> float:
    """Continuous ranked probability score of a sample forecast (lower is better)."""
    x = np.sort(samples)
    n = len(x)
    i = np.arange(1, n + 1)
    return float(np.abs(x - y).mean() - ((2 * i - n - 1) * x).sum() / n**2)


def calibrate(ctx: Context, weeks: list[tuple[int, int]], grid: list[dict], n_sims: int = 3000,
              seed: int = 7) -> tuple[dict, pd.DataFrame]:
    """Grid-search simulation dispersion parameters against completed games (game level only).

    Each grid entry is a dict of SimParams overrides, e.g. {"form_sigma": 0.2, "script_kappa": 0.1}.
    Returns the best overrides (lowest margin + total CRPS) and the full results table.
    """
    records = []
    for season, week in weeks:
        games = ctx.schedules[(ctx.schedules["season"] == season) & (ctx.schedules["week"] == week)
                              & ctx.schedules["completed"]]
        if games.empty:
            continue
        model = fit_model(ctx, games["gameday"].min())
        inputs = game_inputs(ctx, model, games)
        for k, overrides in enumerate(grid):
            p = replace(ctx.params, n_sims=n_sims, **overrides)
            rng = np.random.default_rng(seed)
            for g in games.itertuples(index=False):
                sim = simulate_game(inputs[g.game_id], p, rng)
                margin = sim["home_pts"] - sim["away_pts"]
                total = sim["home_pts"] + sim["away_pts"]
                y_m, y_t = g.home_score - g.away_score, g.home_score + g.away_score
                p_home = np.clip((margin > 0).mean() + 0.5 * (margin == 0).mean(), 0.01, 0.99)
                won = 1.0 if y_m > 0 else 0.0 if y_m < 0 else 0.5
                records.append(dict(
                    k=k, **overrides, game_id=g.game_id,
                    crps_margin=crps(margin, y_m), crps_total=crps(total, y_t),
                    log_loss=-(won * np.log(p_home) + (1 - won) * np.log(1 - p_home)),
                    brier=(p_home - won) ** 2,
                    margin_in_80=np.quantile(margin, 0.1) <= y_m <= np.quantile(margin, 0.9),
                    total_in_80=np.quantile(total, 0.1) <= y_t <= np.quantile(total, 0.9),
                ))
    keys = list(grid[0])
    res = (pd.DataFrame(records)
           .groupby(["k"] + keys)[["crps_margin", "crps_total", "log_loss", "brier", "margin_in_80", "total_in_80"]]
           .mean().reset_index())
    res["score"] = res["crps_margin"] + res["crps_total"]
    best = res.loc[res["score"].idxmin()]
    return {key: float(best[key]) for key in keys}, res.drop(columns="k")


def completed_weeks(schedules: pd.DataFrame, seasons: list[int]) -> list[tuple[int, int]]:
    s = schedules[schedules["season"].isin(seasons) & schedules["completed"]]
    return sorted(set(zip(s["season"].astype(int), s["week"].astype(int))))
