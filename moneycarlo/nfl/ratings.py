"""Pre-game team ratings with no look-ahead.

Each rating is a decayed ratio of sums (e.g. touchdowns / drives), opponent adjusted,
shrunk toward the league average, and regressed toward the mean between seasons.
Walking the schedule in date order and recording each team's rating *before* every
game means features are only ever built from games that already happened.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from moneycarlo.config import SimParams


@dataclass(frozen=True)
class Metric:
    name: str
    num: str
    den: str
    prior_den: float  # pseudo-observations (in `den` units) of league-average prior
    adjust: bool = True  # opponent adjust?
    defense: bool = True  # also track the allowed version?


METRICS = [
    Metric("epa", "epa_sum", "plays", 180),
    Metric("sr", "success_sum", "plays", 180),
    Metric("td_rate", "d_td", "drives", 33),
    Metric("fg_rate", "d_fg", "drives", 33),
    Metric("to_rate", "d_to", "drives", 33),
    Metric("ppd", "drive_plays", "drives", 33),
    Metric("sack_rate", "sacks", "dropbacks", 100),
    Metric("comp_pct", "completions", "attempts", 100),
    Metric("ypcomp", "pass_yards", "completions", 60),
    Metric("ypc", "rush_yards", "carries", 75),
    Metric("pass_rate", "pass_rate_logit", "one", 3, adjust=False, defense=False),
    Metric("int_share", "interceptions", "d_to", 6, adjust=False, defense=False),
    Metric("pass_td_share", "pass_td", "off_tds", 8, adjust=False, defense=False),
]


def fit_script_beta(team_games: pd.DataFrame) -> float:
    """Slope of logit(dropback rate) on final margin: how game script moves pass rate."""
    tg = team_games.dropna(subset=["margin"])
    pr = (tg["dropbacks"] / (tg["dropbacks"] + tg["carries"])).clip(0.05, 0.95)
    y = np.log(pr / (1 - pr))
    x = tg["margin"].clip(-35, 35)
    return float(np.polyfit(x, y, 1)[0])


def prepare_team_games(team_games: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    tg = team_games.copy()
    sc = schedules.set_index("game_id")
    home = tg["game_id"].map(sc["home_team"])
    hs, as_ = tg["game_id"].map(sc["home_score"]), tg["game_id"].map(sc["away_score"])
    tg["points"] = np.where(tg["team"] == home, hs, as_)
    tg["margin"] = np.where(tg["team"] == home, hs - as_, as_ - hs)
    tg["dropbacks"] = tg["attempts"] + tg["sacks"]
    tg["off_tds"] = tg["pass_td"] + tg["rush_td"]
    tg["one"] = 1.0
    return tg


class _State:
    __slots__ = ("num", "den")

    def __init__(self):
        self.num = 0.0
        self.den = 0.0


def compute_ratings(
    schedules: pd.DataFrame, team_games: pd.DataFrame, params: SimParams | None = None
) -> tuple[pd.DataFrame, float]:
    """Return (pre-game ratings per game_id/team, script beta).

    Ratings columns: off_<metric>, def_<metric> (allowed), lg_<metric> (league mean).
    """
    params = params or SimParams()
    tg = prepare_team_games(team_games, schedules)
    beta = fit_script_beta(tg)
    pr = (tg["dropbacks"] / (tg["dropbacks"] + tg["carries"]).replace(0, np.nan)).clip(0.05, 0.95)
    tg["pass_rate_logit"] = np.log(pr / (1 - pr)) - beta * tg["margin"].clip(-35, 35)
    tg = tg.dropna(subset=["pass_rate_logit"])
    by_key = {(g, t): row for g, t, row in zip(tg["game_id"], tg["team"], tg.to_dict("records"))}

    decay = params.rating_decay
    carry = params.season_carryover
    lg_decay = 0.998
    # Seed league means from the first season in the data.
    first = tg[tg["season"] == tg["season"].min()]
    league = {}
    for m in METRICS:
        st = _State()
        st.num, st.den = first[m.num].sum(), first[m.den].sum()
        league[m.name] = st

    def lg(m: Metric) -> float:
        st = league[m.name]
        return st.num / st.den if st.den else 0.0

    states: dict[tuple[str, str, str], _State] = {}
    last_season: dict[str, int] = {}

    def value(team: str, side: str, m: Metric) -> float:
        st = states.get((team, side, m.name))
        mean = lg(m)
        if st is None:
            return mean
        return (st.num + m.prior_den * mean) / (st.den + m.prior_den)

    def new_season(team: str) -> None:
        for m in METRICS:
            for side in ("off", "def"):
                st = states.get((team, side, m.name))
                if st is None:
                    continue
                mean = lg(m)
                v = (st.num + m.prior_den * mean) / (st.den + m.prior_den)
                target = mean + carry * (v - mean)
                st.den = m.prior_den
                st.num = 2 * m.prior_den * target - m.prior_den * mean

    records = []
    games = schedules.sort_values(["gameday", "gametime", "game_id"])
    for g in games.itertuples(index=False):
        teams = (g.home_team, g.away_team)
        for t in teams:
            if last_season.get(t) is not None and last_season[t] != g.season:
                new_season(t)
            last_season[t] = g.season
        pre = {}
        for t in teams:
            rec = {"game_id": g.game_id, "team": t}
            for m in METRICS:
                rec[f"off_{m.name}"] = value(t, "off", m)
                if m.defense:
                    rec[f"def_{m.name}"] = value(t, "def", m)
                rec[f"lg_{m.name}"] = lg(m)
            pre[t] = rec
            records.append(rec)

        if not g.completed:
            continue
        rows = {t: by_key.get((g.game_id, t)) for t in teams}
        if any(r is None for r in rows.values()):
            continue
        for t, o in (teams, teams[::-1]):
            off_row = rows[t]
            for m in METRICS:
                num, den = float(off_row[m.num]), float(off_row[m.den])
                if den <= 0 or not np.isfinite(num):
                    continue
                mean = lg(m)
                # offense of t, adjusted for o's defense (and vice versa)
                adj_off = num - den * (pre[o][f"def_{m.name}"] - mean) if (m.adjust and m.defense) else num
                st = states.setdefault((t, "off", m.name), _State())
                st.num = st.num * decay + adj_off
                st.den = st.den * decay + den
                if m.defense:
                    adj_def = num - den * (pre[t][f"off_{m.name}"] - mean) if m.adjust else num
                    sd = states.setdefault((o, "def", m.name), _State())
                    sd.num = sd.num * decay + adj_def
                    sd.den = sd.den * decay + den
        for t in teams:
            for m in METRICS:
                num, den = float(rows[t][m.num]), float(rows[t][m.den])
                if den > 0 and np.isfinite(num):
                    st = league[m.name]
                    st.num = st.num * lg_decay + num
                    st.den = st.den * lg_decay + den

    return pd.DataFrame.from_records(records), beta
