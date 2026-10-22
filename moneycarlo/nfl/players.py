"""Player usage profiles and allocation of simulated team stats to players.

Each simulated team box score (attempts, completions, yards, TDs) is split between
players, so player lines always add up to the team totals of the same simulated game.
Usage shares vary game to game (Dirichlet), and efficiency is shrunk toward
position averages so small samples don't produce wild projections.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from moneycarlo.config import SimParams

STAT_MAP = {
    "attempts": "pass_att",
    "completions": "pass_cmp",
    "passing_yards": "pass_yds",
    "passing_tds": "pass_td",
    "passing_interceptions": "pass_int",
    "carries": "rush_att",
    "rushing_yards": "rush_yds",
    "rushing_tds": "rush_td",
    "targets": "targets",
    "receptions": "receptions",
    "receiving_yards": "rec_yds",
    "receiving_tds": "rec_td",
}
PLAYER_STATS = list(STAT_MAP.values())

# Position priors: catch rate, yards/reception, yards/carry, TD per target, TD per carry.
_PRIORS = {
    "WR": dict(catch=0.63, ypr=12.8, ypc=6.5, td_tgt=0.045, td_car=0.03),
    "TE": dict(catch=0.70, ypr=10.6, ypc=4.0, td_tgt=0.05, td_car=0.03),
    "RB": dict(catch=0.77, ypr=7.8, ypc=4.3, td_tgt=0.03, td_car=0.03),
    "FB": dict(catch=0.75, ypr=7.0, ypc=3.5, td_tgt=0.06, td_car=0.06),
    "QB": dict(catch=0.6, ypr=10.0, ypc=5.2, td_tgt=0.05, td_car=0.035),
}
_DEFAULT_PRIOR = _PRIORS["WR"]
_OTHER = dict(catch=0.62, ypr=10.0, ypc=4.5, td_tgt=0.04, td_car=0.03)
_INACTIVE = {"Out", "Doubtful"}
SKILL_POSITIONS = {"QB", "RB", "WR", "TE", "FB"}


def _before(ps: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    return ps[(ps["season"] < season) | ((ps["season"] == season) & (ps["week"] < week))]


def usage_profiles(
    player_stats: pd.DataFrame,
    team: str,
    season: int,
    week: int,
    injuries: pd.DataFrame | None = None,
    depth: pd.DataFrame | None = None,
    window: int = 10,
    recency: float = 0.85,
) -> pd.DataFrame:
    """Usage and efficiency profile for each likely-active skill player on `team`."""
    hist = _before(player_stats, season, week)
    hist = hist[hist["position"].isin(SKILL_POSITIONS)]
    order = hist.sort_values(["season", "week"])
    latest_team = order.groupby("player_id")["team"].last()

    tgames = (
        hist[hist["team"] == team][["game_id", "season", "week"]]
        .drop_duplicates().sort_values(["season", "week"]).tail(window)
    )
    if tgames.empty:
        return pd.DataFrame()
    tgames["w"] = recency ** np.arange(len(tgames))[::-1]
    th = hist[hist["game_id"].isin(tgames["game_id"]) & (hist["team"] == team)]
    tot = th.groupby("game_id")[["targets", "carries"]].sum().rename(columns=lambda c: f"team_{c}")
    th = th.merge(tot, left_on="game_id", right_index=True).merge(tgames[["game_id", "w"]], on="game_id")
    th["tgt_sh"] = th["targets"] / th["team_targets"].replace(0, np.nan)
    th["car_sh"] = th["carries"] / th["team_carries"].replace(0, np.nan)
    th = th.fillna({"tgt_sh": 0.0, "car_sh": 0.0})
    th["wt"], th["wc"] = th["w"] * th["tgt_sh"], th["w"] * th["car_sh"]

    agg = th.groupby("player_id").agg(
        name=("player_display_name", "last"), position=("position", "last"),
        wsum=("w", "sum"), wt=("wt", "sum"), wc=("wc", "sum"),
        last_game=("game_id", "last"), att=("attempts", "sum"),
    )
    recent_games = set(tgames["game_id"].tail(3))
    recent_ids = set(th.loc[th["game_id"].isin(recent_games), "player_id"])
    # Shrink so a player who appeared once doesn't get a full share.
    agg["target_share"] = agg["wt"] / (agg["wsum"] + 0.5)
    agg["carry_share"] = agg["wc"] / (agg["wsum"] + 0.5)
    agg = agg[agg.index.isin(recent_ids) & (latest_team.reindex(agg.index) == team)]

    out_ids: set = set()
    if injuries is not None and not injuries.empty:
        inj = injuries[(injuries["season"] == season) & (injuries["week"] == week) & (injuries["team"] == team)]
        out_ids = set(inj.loc[inj["report_status"].isin(_INACTIVE), "gsis_id"])
    agg = agg[~agg.index.isin(out_ids)]

    # ---- starting QB: depth chart if available, else most recent passer who is active.
    # Most recent game last, and within a game the QB with the most attempts last.
    qbs = th[(th["position"] == "QB") & (th["attempts"] > 0)].sort_values(["season", "week", "attempts"])
    starter = None
    if depth is not None and not depth.empty:
        d = depth[(depth["team"] == team) & (depth["pos_abb"] == "QB")].sort_values("pos_rank")
        for gid in d["gsis_id"]:
            if gid not in out_ids:
                starter = gid
                break
    if starter is None:
        for gid in qbs["player_id"].iloc[::-1]:
            if gid not in out_ids and latest_team.get(gid) == team:
                starter = gid
                break
    agg = agg[(agg["position"] != "QB") | (agg.index == starter)]
    if starter is not None and starter not in agg.index:
        name = player_stats.loc[player_stats["player_id"] == starter, "player_display_name"]
        if name.empty and depth is not None:
            name = depth.loc[depth["gsis_id"] == starter, "player_name"]
        agg.loc[starter] = dict(name=name.iloc[-1] if len(name) else starter, position="QB",
                                target_share=0.0, carry_share=0.08)
    agg["is_starting_qb"] = agg.index == starter

    # ---- efficiency from each player's own recent games (any team), shrunk to priors
    eff_rows = []
    for pid, pos in agg["position"].items():
        p = hist[hist["player_id"] == pid].sort_values(["season", "week"]).tail(17)
        w = 0.93 ** np.arange(len(p))[::-1]
        s = lambda c: float((p[c] * w).sum()) if len(p) else 0.0
        pr = _PRIORS.get(pos, _DEFAULT_PRIOR)
        eff_rows.append(dict(
            player_id=pid,
            catch_rate=(s("receptions") + 20 * pr["catch"]) / (s("targets") + 20),
            ypr=(s("receiving_yards") + 15 * pr["ypr"]) / (s("receptions") + 15),
            ypc=(s("rushing_yards") + 30 * pr["ypc"]) / (s("carries") + 30),
            td_per_target=(s("receiving_tds") + 40 * pr["td_tgt"]) / (s("targets") + 40),
            td_per_carry=(s("rushing_tds") + 50 * pr["td_car"]) / (s("carries") + 50),
        ))
    eff = pd.DataFrame(eff_rows).set_index("player_id") if eff_rows else pd.DataFrame()
    prof = agg.join(eff)
    prof.index.name = "player_id"
    prof = prof[(prof["target_share"] > 0.005) | (prof["carry_share"] > 0.005) | prof["is_starting_qb"]]
    cols = ["name", "position", "is_starting_qb", "target_share", "carry_share", "catch_rate",
            "ypr", "ypc", "td_per_target", "td_per_carry"]
    return prof[cols].reset_index()


def _norm_shares(shares: np.ndarray, known: float) -> np.ndarray:
    """Scale known-player shares to sum to `known`, plus an 'other' bucket.

    Rescaling (rather than leaving gaps) hands an injured player's usage to his active
    teammates in proportion to their own roles.
    """
    total = shares.sum()
    if total > 0:
        shares = shares * known / total
    return np.append(shares, 1.0 - shares.sum())


def _multinomial_rows(rng, n: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Multinomial with a different probability row per simulation; all-zero rows go to the last column."""
    w = weights.astype(float).copy()
    empty = w.sum(axis=1) <= 0
    w[empty, -1] = 1.0
    p = w / w.sum(axis=1, keepdims=True)
    return rng.multinomial(n, p)


def allocate(box: dict, prof: pd.DataFrame, params: SimParams, rng: np.random.Generator) -> dict:
    """Split one team's simulated box score into per-player stat arrays.

    Updates `box` completions / pass_yards in place so team and player totals agree.
    Returns {player_id: {stat: np.ndarray(S)}}.
    """
    S = len(box["attempts"])
    if prof.empty:
        return {}
    n = len(prof)
    lognorm = lambda shape: np.exp(rng.normal(0, params.player_yards_noise, shape))

    # ---- receiving
    tshare = _norm_shares(prof["target_share"].to_numpy(), known=0.94)
    alpha = np.maximum(tshare * params.target_concentration, 1e-3)
    tgt_p = rng.dirichlet(alpha, size=S)
    targets = rng.multinomial(box["attempts"], tgt_p)
    catch = np.append(prof["catch_rate"].to_numpy(), _OTHER["catch"])
    # Scale catch rates per simulation so expected receptions match the team's completions.
    exp_rec = (targets * catch[None, :]).sum(axis=1, keepdims=True)
    k = np.where(exp_rec > 0, box["completions"][:, None] / np.where(exp_rec > 0, exp_rec, 1), 0.0)
    rec = rng.binomial(targets, np.clip(catch[None, :] * k, 0.0, 0.98))
    ypr = np.append(prof["ypr"].to_numpy(), _OTHER["ypr"])
    w = rec * ypr[None, :] * lognorm((S, n + 1))
    wsum = w.sum(axis=1, keepdims=True)
    rec_yds = np.where(wsum > 0, box["pass_yards"][:, None] * w / np.where(wsum > 0, wsum, 1), 0.0)
    td_tgt = np.append(prof["td_per_target"].to_numpy(), _OTHER["td_tgt"])
    rec_td = _multinomial_rows(rng, box["pass_td"], targets * td_tgt[None, :] * (rec > 0))

    # ---- rushing
    cshare = _norm_shares(prof["carry_share"].to_numpy(), known=0.97)
    alpha = np.maximum(cshare * params.carry_concentration, 1e-3)
    car_p = rng.dirichlet(alpha, size=S)
    carries = rng.multinomial(box["carries"], car_p)
    ypc = np.append(prof["ypc"].to_numpy(), _OTHER["ypc"])
    w = carries * ypc[None, :] * lognorm((S, n + 1))
    wsum = w.sum(axis=1, keepdims=True)
    rush_yds = np.where(wsum > 0, box["rush_yards"][:, None] * w / np.where(wsum > 0, wsum, 1), 0.0)
    td_car = np.append(prof["td_per_carry"].to_numpy(), _OTHER["td_car"])
    rush_td = _multinomial_rows(rng, box["rush_td"], carries * td_car[None, :])

    # Team completions / passing yards are the sum of what receivers caught.
    box["completions"] = rec.sum(axis=1)
    box["pass_yards"] = np.rint(rec_yds.sum(axis=1))

    out = {}
    for i, row in enumerate(prof.itertuples(index=False)):
        stats = {
            "rush_att": carries[:, i],
            "rush_yds": np.rint(rush_yds[:, i]),
            "rush_td": rush_td[:, i],
        }
        if row.is_starting_qb:
            stats.update({
                "pass_att": box["attempts"],
                "pass_cmp": box["completions"],
                "pass_yds": box["pass_yards"],
                "pass_td": box["pass_td"],
                "pass_int": box["interceptions"],
            })
        if row.position != "QB":
            stats.update({
                "targets": targets[:, i],
                "receptions": rec[:, i],
                "rec_yds": np.rint(rec_yds[:, i]),
                "rec_td": rec_td[:, i],
            })
        out[row.player_id] = stats
    return out
