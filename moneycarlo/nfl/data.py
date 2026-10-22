"""Download and cache NFL data from nflverse (via nflreadpy).

Raw play-by-play is large, so it is reduced to two compact tables per season and
cached as parquet:

* drives      - one row per offensive drive (outcome class, plays)
* team_games  - one row per team per game (offensive box score + efficiency)

Completed seasons are cached forever; the current season is refreshed by `update`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from moneycarlo.config import CACHE_DIR

log = logging.getLogger(__name__)

DRIVE_CLASSES = ["TD", "FG", "TO", "NS"]  # touchdown, field goal, turnover, no score

_DRIVE_RESULT_MAP = {
    "Touchdown": "TD",
    "Field goal": "FG",
    "Turnover": "TO",
    "Opp touchdown": "TO",
    "Punt": "NS",
    "Turnover on downs": "NS",
    "End of half": "NS",
    "Missed field goal": "NS",
    "Safety": "NS",
}

_PBP_COLUMNS = [
    "game_id", "season", "season_type", "week", "posteam", "defteam", "home_team",
    "qtr", "down", "fixed_drive", "fixed_drive_result", "play_type", "epa", "success",
    "pass_attempt", "rush_attempt", "sack", "complete_pass", "interception", "fumble_lost",
    "passing_yards", "rushing_yards", "pass_touchdown", "rush_touchdown",
    "two_point_attempt", "extra_point_result", "two_point_conv_result",
]


def _nfl():
    import nflreadpy

    return nflreadpy


def current_season() -> int:
    try:
        return int(_nfl().get_current_season())
    except Exception:  # pragma: no cover - fallback if helper is unavailable
        today = pd.Timestamp.today()
        return today.year if today.month >= 9 else today.year - 1


# Bump when the derived tables change shape so stale caches are rebuilt.
_CACHE_VERSION = 2


def _cache_path(name: str, season: int | None = None) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{season}" if season is not None else ""
    return CACHE_DIR / f"{name}{suffix}_v{_CACHE_VERSION}.parquet"


def _cached(name: str, season: int | None, builder, refresh: bool) -> pd.DataFrame:
    path = _cache_path(name, season)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = builder()
    df.to_parquet(path, index=False)
    return df


# --------------------------------------------------------------------------- schedules


def load_schedules(refresh: bool = False) -> pd.DataFrame:
    """All games 1999-present (one small file), with a few derived columns."""

    def build():
        s = _nfl().load_schedules().to_pandas()
        keep = [
            "game_id", "season", "game_type", "week", "gameday", "weekday", "gametime",
            "away_team", "away_score", "home_team", "home_score", "location", "result",
            "total", "overtime", "away_rest", "home_rest", "div_game", "roof", "surface",
            "temp", "wind", "stadium",
        ]
        return s[keep]

    s = _cached("schedules", None, build, refresh)
    s = s.copy()
    s["gameday"] = pd.to_datetime(s["gameday"])
    s["completed"] = s["home_score"].notna() & s["away_score"].notna()
    s["neutral"] = (s["location"] == "Neutral").astype(int)
    s["dome"] = s["roof"].isin(["dome", "closed"]).astype(int)
    return s.sort_values(["gameday", "gametime", "game_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------- play-by-play


def _summarise_pbp(pbp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pbp = pbp[pbp["posteam"].notna() & pbp["defteam"].notna()].copy()
    scrimmage = pbp[pbp["play_type"].isin(["pass", "run"]) & (pbp["two_point_attempt"] != 1)].copy()

    # ---- drives (regulation only; OT is simulated separately)
    reg = pbp[pbp["qtr"] <= 4]
    drives = (
        reg.groupby(["game_id", "fixed_drive", "posteam", "defteam"], as_index=False)
        .agg(
            season=("season", "first"),
            week=("week", "first"),
            qtr=("qtr", "min"),
            result=("fixed_drive_result", "first"),
            giveaway=("interception", "max"),
            fumble_lost=("fumble_lost", "max"),
        )
    )
    drive_plays = (
        scrimmage[scrimmage["qtr"] <= 4]
        .groupby(["game_id", "fixed_drive", "posteam"]).size().rename("plays").reset_index()
    )
    drives = drives.merge(drive_plays, on=["game_id", "fixed_drive", "posteam"], how="left")
    drives["plays"] = drives["plays"].fillna(0).astype(int)
    drives["outcome"] = drives["result"].map(_DRIVE_RESULT_MAP).fillna("NS")
    # "Opp touchdown" covers both turnovers returned for TDs and punt / kick-block return
    # TDs; only the former are turnovers.
    opp_td = drives["result"] == "Opp touchdown"
    had_giveaway = (drives["giveaway"].fillna(0) + drives["fumble_lost"].fillna(0)) > 0
    drives.loc[opp_td & ~had_giveaway, "outcome"] = "NS"
    drives["def_td"] = (opp_td & had_giveaway).astype(int)
    drives["st_td"] = (opp_td & ~had_giveaway).astype(int)
    drives = drives.drop(columns=["giveaway", "fumble_lost"])
    drives = drives.rename(columns={"posteam": "team", "defteam": "opp"})

    # ---- team-game offensive box score + efficiency
    s = scrimmage
    s["is_pass"] = (s["play_type"] == "pass").astype(int)
    s["att"] = ((s["play_type"] == "pass") & (s["sack"] != 1) & (s["pass_attempt"] == 1)).astype(int)
    s["sk"] = ((s["play_type"] == "pass") & (s["sack"] == 1)).astype(int)
    s["car"] = ((s["play_type"] == "run") & (s["rush_attempt"] == 1)).astype(int)
    tg = s.groupby(["game_id", "posteam", "defteam"], as_index=False).agg(
        season=("season", "first"),
        week=("week", "first"),
        plays=("epa", "size"),
        epa_sum=("epa", "sum"),
        success_sum=("success", "sum"),
        attempts=("att", "sum"),
        sacks=("sk", "sum"),
        carries=("car", "sum"),
        completions=("complete_pass", "sum"),
        pass_yards=("passing_yards", "sum"),
        rush_yards=("rushing_yards", "sum"),
        interceptions=("interception", "sum"),
        fumbles_lost=("fumble_lost", "sum"),
        pass_td=("pass_touchdown", "sum"),
        rush_td=("rush_touchdown", "sum"),
    )
    tg = tg.rename(columns={"posteam": "team", "defteam": "opp"})
    dcounts = (
        drives.assign(**{c: (drives["outcome"] == c).astype(int) for c in DRIVE_CLASSES})
        .groupby(["game_id", "team"], as_index=False)
        .agg(drives=("outcome", "size"), d_td=("TD", "sum"), d_fg=("FG", "sum"),
             d_to=("TO", "sum"), d_ns=("NS", "sum"), drive_plays=("plays", "sum"))
    )
    tg = tg.merge(dcounts, on=["game_id", "team"], how="left")

    # ---- scoring constants (PAT / 2pt rates) for the simulator
    xp = pbp["extra_point_result"].dropna()
    tp = pbp["two_point_conv_result"].dropna()
    consts = {
        "xp_att": float(len(xp)),
        "xp_good": float((xp == "good").sum()),
        "tp_att": float(len(tp)),
        "tp_good": float((tp == "success").sum()),
    }
    return drives, tg, consts


def load_season_tables(season: int, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(drives, team_games) for one season, built from play-by-play."""
    dpath, tpath, cpath = (_cache_path(n, season) for n in ("drives", "team_games", "consts"))
    if not refresh and dpath.exists() and tpath.exists():
        return pd.read_parquet(dpath), pd.read_parquet(tpath)
    log.info("Downloading play-by-play for %s", season)
    pbp = _nfl().load_pbp([season]).select(_PBP_COLUMNS).to_pandas()
    drives, tg, consts = _summarise_pbp(pbp)
    drives.to_parquet(dpath, index=False)
    tg.to_parquet(tpath, index=False)
    pd.DataFrame([consts]).to_parquet(cpath, index=False)
    return drives, tg


def load_scoring_constants(seasons: list[int]) -> dict:
    rows = [pd.read_parquet(_cache_path("consts", s)) for s in seasons if _cache_path("consts", s).exists()]
    if not rows:
        return {"p_xp": 0.94, "p_go2": 0.06, "p_2pt": 0.48}
    c = pd.concat(rows).sum()
    attempts = c["xp_att"] + c["tp_att"]
    return {
        "p_xp": c["xp_good"] / max(c["xp_att"], 1),
        "p_go2": c["tp_att"] / max(attempts, 1),
        "p_2pt": c["tp_good"] / max(c["tp_att"], 1),
    }


def load_history(seasons: list[int], refresh_current: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    cur = current_season()
    drives, tgs = [], []
    for s in seasons:
        d, t = load_season_tables(s, refresh=refresh_current and s == cur)
        drives.append(d)
        tgs.append(t)
    return pd.concat(drives, ignore_index=True), pd.concat(tgs, ignore_index=True)


# --------------------------------------------------------------------------- players


_PLAYER_COLS = [
    "player_id", "player_display_name", "position", "season", "week", "season_type", "game_id",
    "team", "opponent_team", "completions", "attempts", "passing_yards", "passing_tds",
    "passing_interceptions", "sacks_suffered", "carries", "rushing_yards", "rushing_tds",
    "receptions", "targets", "receiving_yards", "receiving_tds",
]


def load_player_stats(seasons: list[int], refresh_current: bool = False) -> pd.DataFrame:
    cur = current_season()
    frames = []
    for s in seasons:
        def build(s=s):
            df = _nfl().load_player_stats([s]).to_pandas()
            return df[[c for c in _PLAYER_COLS if c in df.columns]]

        frames.append(_cached("player_stats", s, build, refresh_current and s == cur))
    df = pd.concat(frames, ignore_index=True)
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].fillna(0)
    return df


def load_injuries(season: int, refresh: bool = False) -> pd.DataFrame:
    def build():
        try:
            df = _nfl().load_injuries([season]).to_pandas()
        except Exception as exc:  # injuries are optional; never block a prediction
            log.warning("Injury data unavailable for %s: %s", season, exc)
            return pd.DataFrame(columns=["season", "week", "team", "gsis_id", "full_name", "position", "report_status"])
        return df[["season", "week", "team", "gsis_id", "full_name", "position", "report_status"]]

    return _cached("injuries", season, build, refresh)


def load_depth_charts(season: int, refresh: bool = False) -> pd.DataFrame:
    """Latest snapshot of offensive skill-position depth charts."""

    def build():
        try:
            df = _nfl().load_depth_charts([season]).to_pandas()
        except Exception as exc:
            log.warning("Depth charts unavailable for %s: %s", season, exc)
            return pd.DataFrame(columns=["dt", "team", "gsis_id", "player_name", "pos_abb", "pos_rank"])
        if "dt" not in df.columns:  # older seasons use a different layout
            return pd.DataFrame(columns=["dt", "team", "gsis_id", "player_name", "pos_abb", "pos_rank"])
        df = df[df["pos_abb"].isin(["QB", "RB", "WR", "TE", "FB"])]
        df = df[df["dt"] == df["dt"].max()]
        return df[["dt", "team", "gsis_id", "player_name", "pos_abb", "pos_rank"]]

    return _cached("depth_charts", season, build, refresh)


def update(seasons: list[int]) -> None:
    """Refresh the current season's data (and fill any missing past seasons)."""
    cur = current_season()
    load_schedules(refresh=True)
    for s in seasons:
        load_season_tables(s, refresh=(s == cur))
    load_player_stats(seasons, refresh_current=True)
    load_injuries(cur, refresh=True)
    load_depth_charts(cur, refresh=True)
