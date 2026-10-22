"""DuckDB storage: every prediction run, its outputs, actual results and tuned parameters.

This is the engine's memory. Predictions are written *before* games are played and
results are attached afterwards, so the track record can be scored honestly and fed
back into calibration.
"""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from moneycarlo.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR PRIMARY KEY,
    created_at TIMESTAMP,
    kind VARCHAR,            -- 'live' or 'backtest'
    season INTEGER,
    week INTEGER,
    model_version VARCHAR,
    n_sims INTEGER,
    params JSON
);
CREATE TABLE IF NOT EXISTS game_predictions (
    run_id VARCHAR, game_id VARCHAR, season INTEGER, week INTEGER, game_type VARCHAR,
    gameday DATE, gametime VARCHAR, home_team VARCHAR, away_team VARCHAR,
    home_win_prob DOUBLE, away_win_prob DOUBLE, tie_prob DOUBLE,
    home_pts_mean DOUBLE, away_pts_mean DOUBLE, home_pts_median DOUBLE, away_pts_median DOUBLE,
    home_pts_p10 DOUBLE, home_pts_p90 DOUBLE, away_pts_p10 DOUBLE, away_pts_p90 DOUBLE,
    margin_mean DOUBLE, margin_median DOUBLE, margin_sd DOUBLE, margin_p10 DOUBLE, margin_p90 DOUBLE,
    total_mean DOUBLE, total_median DOUBLE, total_sd DOUBLE, total_p10 DOUBLE, total_p90 DOUBLE,
    most_likely_score VARCHAR
);
CREATE TABLE IF NOT EXISTS score_distributions (
    run_id VARCHAR, game_id VARCHAR, home_pts INTEGER, away_pts INTEGER, prob DOUBLE
);
CREATE TABLE IF NOT EXISTS team_predictions (
    run_id VARCHAR, game_id VARCHAR, team VARCHAR, stat VARCHAR,
    mean DOUBLE, sd DOUBLE, p10 DOUBLE, p50 DOUBLE, p90 DOUBLE
);
CREATE TABLE IF NOT EXISTS player_predictions (
    run_id VARCHAR, game_id VARCHAR, team VARCHAR, player_id VARCHAR, player_name VARCHAR,
    position VARCHAR, stat VARCHAR, mean DOUBLE, sd DOUBLE,
    p10 DOUBLE, p25 DOUBLE, p50 DOUBLE, p75 DOUBLE, p90 DOUBLE, prob_pos DOUBLE
);
CREATE TABLE IF NOT EXISTS game_results (
    game_id VARCHAR PRIMARY KEY, season INTEGER, week INTEGER,
    home_team VARCHAR, away_team VARCHAR, home_score INTEGER, away_score INTEGER
);
CREATE TABLE IF NOT EXISTS team_results (
    game_id VARCHAR, team VARCHAR, stat VARCHAR, value DOUBLE
);
CREATE TABLE IF NOT EXISTS player_results (
    game_id VARCHAR, team VARCHAR, player_id VARCHAR, stat VARCHAR, value DOUBLE
);
CREATE TABLE IF NOT EXISTS model_params (
    created_at TIMESTAMP, params JSON, metrics JSON
);
"""


@contextmanager
def connect(path: Path | str = DB_PATH, read_only: bool = False):
    path = Path(path)
    if not read_only:
        path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    try:
        if not read_only:
            for stmt in filter(str.strip, SCHEMA.split(";")):
                _exec(con, stmt)
        yield con
    finally:
        con.close()


def _exec(con, sql: str, params=None) -> None:
    """Run a statement to completion (DuckDB keeps results pending until fetched)."""
    con.execute(sql, params or []).fetchall()


def _insert(con, table: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    cols = [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]
    df = df[[c for c in cols if c in df.columns]]
    con.register("_df", df)
    _exec(con, f"INSERT INTO {table} ({', '.join(df.columns)}) SELECT * FROM _df")
    con.unregister("_df")


def save_run(
    con,
    *,
    kind: str,
    season: int,
    week: int,
    model_version: str,
    params: dict,
    games: pd.DataFrame,
    scores: pd.DataFrame,
    teams: pd.DataFrame,
    players: pd.DataFrame,
) -> str:
    run_id = uuid.uuid4().hex[:12]
    created = datetime.now(timezone.utc).replace(tzinfo=None)
    _exec(
        con,
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [run_id, created, kind, season, week, model_version, params.get("n_sims"), json.dumps(params)],
    )
    for table, df in (
        ("game_predictions", games),
        ("score_distributions", scores),
        ("team_predictions", teams),
        ("player_predictions", players),
    ):
        if df is not None and not df.empty:
            _insert(con, table, df.assign(run_id=run_id))
    return run_id


def delete_runs(con, kind: str, season: int, week: int) -> None:
    ids = [r[0] for r in con.execute(
        "SELECT run_id FROM runs WHERE kind = ? AND season = ? AND week = ?", [kind, season, week]
    ).fetchall()]
    for table in ("game_predictions", "score_distributions", "team_predictions", "player_predictions", "runs"):
        for rid in ids:
            _exec(con, f"DELETE FROM {table} WHERE run_id = ?", [rid])


def replace_results(con, games: pd.DataFrame, teams: pd.DataFrame, players: pd.DataFrame) -> None:
    for table, df in (("game_results", games), ("team_results", teams), ("player_results", players)):
        if df.empty:
            continue
        ids = df["game_id"].unique().tolist()
        con.register("_ids", pd.DataFrame({"game_id": ids}))
        _exec(con, f"DELETE FROM {table} WHERE game_id IN (SELECT game_id FROM _ids)")
        con.unregister("_ids")
        _insert(con, table, df)


def save_params(con, params: dict, metrics: dict) -> None:
    _exec(
        con,
        "INSERT INTO model_params VALUES (?, ?, ?)",
        [datetime.now(timezone.utc).replace(tzinfo=None), json.dumps(params), json.dumps(metrics)],
    )


def latest_params(con) -> dict | None:
    rows = con.execute("SELECT params FROM model_params ORDER BY created_at DESC LIMIT 1").fetchall()
    return json.loads(rows[0][0]) if rows else None
