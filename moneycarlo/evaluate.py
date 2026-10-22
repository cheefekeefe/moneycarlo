"""Score stored predictions against actual results."""

from __future__ import annotations

import numpy as np
import pandas as pd

LATEST_GAME_PRED = """
WITH ranked AS (
    SELECT gp.*, r.kind, r.created_at,
           ROW_NUMBER() OVER (PARTITION BY gp.game_id, r.kind ORDER BY r.created_at DESC) AS rn
    FROM game_predictions gp JOIN runs r USING (run_id)
    WHERE r.kind = ?
)
SELECT * FROM ranked WHERE rn = 1
"""


def game_scorecard(con, kind: str = "live") -> pd.DataFrame:
    """Latest prediction of each completed game joined with the final score."""
    preds = con.execute(LATEST_GAME_PRED, [kind]).df()
    res = con.execute("SELECT game_id, home_score, away_score FROM game_results").df()
    df = preds.merge(res, on="game_id", how="inner")
    if df.empty:
        return df
    df["actual_margin"] = df["home_score"] - df["away_score"]
    df["actual_total"] = df["home_score"] + df["away_score"]
    won = np.select([df["actual_margin"] > 0, df["actual_margin"] < 0], [1.0, 0.0], 0.5)
    p = (df["home_win_prob"] + 0.5 * df["tie_prob"]).clip(0.01, 0.99)
    df["home_won"] = won
    df["p_home"] = p
    df["correct"] = np.where(won == 0.5, np.nan, ((p > 0.5) == (won == 1.0)).astype(float))
    df["brier"] = (p - won) ** 2
    df["log_loss"] = -(won * np.log(p) + (1 - won) * np.log(1 - p))
    df["margin_err"] = df["margin_mean"] - df["actual_margin"]
    df["total_err"] = df["total_mean"] - df["actual_total"]
    df["margin_in_80"] = df["actual_margin"].between(df["margin_p10"], df["margin_p90"])
    df["total_in_80"] = df["actual_total"].between(df["total_p10"], df["total_p90"])
    return df


def summary(card: pd.DataFrame) -> dict:
    if card.empty:
        return {}
    return {
        "games": int(len(card)),
        "winner_accuracy": float(card["correct"].mean()),
        "brier": float(card["brier"].mean()),
        "log_loss": float(card["log_loss"].mean()),
        "margin_mae": float(card["margin_err"].abs().mean()),
        "total_mae": float(card["total_err"].abs().mean()),
        "margin_80_coverage": float(card["margin_in_80"].mean()),
        "total_80_coverage": float(card["total_in_80"].mean()),
    }


def player_scorecard(con, kind: str = "live") -> pd.DataFrame:
    q = """
    WITH latest AS (
        SELECT gp.game_id, gp.run_id,
               ROW_NUMBER() OVER (PARTITION BY gp.game_id ORDER BY r.created_at DESC) AS rn
        FROM game_predictions gp JOIN runs r USING (run_id) WHERE r.kind = ?
    )
    -- nflverse only lists players who recorded a stat, so a projected player missing from a
    -- completed game's results is scored as zero rather than silently dropped.
    SELECT pp.*, COALESCE(pr.value, 0) AS actual
    FROM player_predictions pp
    JOIN latest l ON pp.run_id = l.run_id AND pp.game_id = l.game_id AND l.rn = 1
    JOIN game_results gr ON gr.game_id = pp.game_id
    LEFT JOIN player_results pr
        ON pr.game_id = pp.game_id AND pr.player_id = pp.player_id AND pr.stat = pp.stat
    """
    df = con.execute(q, [kind]).df()
    if df.empty:
        return df
    df["err"] = df["mean"] - df["actual"]
    df["in_80"] = df["actual"].between(df["p10"], df["p90"])
    return df


def player_summary(card: pd.DataFrame) -> pd.DataFrame:
    if card.empty:
        return card
    return (
        card.groupby("stat")
        .agg(n=("err", "size"), mae=("err", lambda e: e.abs().mean()), bias=("err", "mean"),
             coverage_80=("in_80", "mean"))
        .reset_index()
    )
