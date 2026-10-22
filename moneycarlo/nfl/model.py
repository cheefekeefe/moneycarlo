"""Statistical models that feed the simulator.

* Drive-outcome model: multinomial logistic regression giving P(TD, FG, turnover,
  no score) for a drive, from the offense's and defense's pre-game ratings plus
  home field, roof and rest.
* Drive-count model: linear regression of regulation drives per game on both
  teams' plays-per-drive (pace).
* League constants: plays per drive by outcome, defensive-TD rate on turnovers,
  PAT / 2pt rates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from moneycarlo.nfl.data import DRIVE_CLASSES

OFF_FEATURES = ["epa", "sr", "td_rate", "fg_rate", "to_rate", "ppd"]
DEF_FEATURES = ["epa", "sr", "td_rate", "fg_rate", "to_rate", "ppd"]
FEATURE_NAMES = (
    [f"off_{m}" for m in OFF_FEATURES]
    + [f"def_{m}" for m in DEF_FEATURES]
    + ["home", "dome", "rest_diff"]
)


def matchup_frame(schedules: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, offense) with offense ratings, opposing defense ratings and context."""
    r = ratings.set_index(["game_id", "team"])
    rows = []
    for side, opp_side in (("home", "away"), ("away", "home")):
        df = schedules[["game_id", "season", "week", f"{side}_team", f"{opp_side}_team",
                        f"{side}_rest", f"{opp_side}_rest", "neutral", "dome"]].copy()
        df.columns = ["game_id", "season", "week", "team", "opp", "rest", "opp_rest", "neutral", "dome"]
        df["is_home"] = side == "home"
        df["home"] = np.where(df["neutral"] == 1, 0.0, 1.0 if side == "home" else -1.0)
        rows.append(df)
    m = pd.concat(rows, ignore_index=True)
    off = r.loc[list(zip(m["game_id"], m["team"]))].reset_index(drop=True)
    dfn = r.loc[list(zip(m["game_id"], m["opp"]))].reset_index(drop=True)
    for col in off.columns:
        if col.startswith("off_") or col.startswith("lg_"):
            m[col] = off[col].to_numpy()
    for col in dfn.columns:
        if col.startswith("def_"):
            m[col] = dfn[col].to_numpy()
    m["rest_diff"] = ((m["rest"].fillna(7) - m["opp_rest"].fillna(7)).clip(-7, 7)) / 7.0
    return m


def design_matrix(m: pd.DataFrame) -> np.ndarray:
    X = pd.DataFrame(index=m.index)
    for f in OFF_FEATURES:
        X[f"off_{f}"] = m[f"off_{f}"] - m[f"lg_{f}"]
    for f in DEF_FEATURES:
        X[f"def_{f}"] = m[f"def_{f}"] - m[f"lg_{f}"]
    X["home"] = m["home"]
    X["dome"] = m["dome"]
    X["rest_diff"] = m["rest_diff"]
    return X[FEATURE_NAMES].to_numpy(dtype=float)


@dataclass
class NFLModel:
    drive_model: object = None
    drives_intercept: float = 22.0
    drives_slope: float = -1.0
    drives_sd: float = 2.0
    ppd_by_outcome: dict = field(default_factory=dict)
    p_def_td: float = 0.1
    p_st_td: float = 0.01
    scoring: dict = field(default_factory=dict)
    script_beta: float = -0.02

    def fit(self, drives: pd.DataFrame, matchups: pd.DataFrame, scoring: dict, script_beta: float,
            season_decay: float = 0.75) -> "NFLModel":
        """`drives` and `matchups` must only contain games before the prediction date.

        Older seasons are down-weighted by `season_decay` per season so the model tracks
        league-wide trends (scoring environment, pace) instead of a 10-year average.
        """
        d = drives.merge(matchups, on=["game_id", "team"], how="inner", suffixes=("", "_m"))
        latest = d["season"].max()
        X = design_matrix(d)
        y = d["outcome"].to_numpy()
        self.drive_model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
        self.drive_model.fit(X, y, logisticregression__sample_weight=season_decay ** (latest - d["season"]))

        # Pace: regulation drives per game vs. both offenses' plays-per-drive.
        g = drives.groupby("game_id").agg(
            n=("outcome", "size"), season=("season", "first"), gameday=("gameday", "first")
        ).reset_index()
        pace = matchups.groupby("game_id").apply(
            lambda x: (x["off_ppd"] + x["def_ppd"] - 2 * x["lg_ppd"]).sum(), include_groups=False
        ).rename("pace").reset_index()
        g = g.merge(pace, on="game_id")
        w = season_decay ** (latest - g["season"])
        lr = LinearRegression().fit(g[["pace"]], g["n"], sample_weight=w)
        self.drives_slope = float(lr.coef_[0])
        # Drives per game trend down over time, so anchor the baseline on the last 12 months.
        recent_games = g[g["gameday"] >= g["gameday"].max() - pd.Timedelta(days=365)]
        self.drives_intercept = float((recent_games["n"] - self.drives_slope * recent_games["pace"]).mean())
        resid = recent_games["n"] - self.expected_drives(recent_games["pace"])
        self.drives_sd = float(resid.std())

        recent = drives[drives["season"] >= latest - 2]
        self.ppd_by_outcome = recent.groupby("outcome")["plays"].mean().to_dict()
        to = recent[recent["outcome"] == "TO"]
        ns = recent[recent["outcome"] == "NS"]
        self.p_def_td = float(to["def_td"].mean()) if len(to) else 0.08
        self.p_st_td = float(ns["st_td"].mean()) if len(ns) else 0.01
        self.scoring = scoring
        self.script_beta = script_beta
        return self

    def drive_probs(self, matchups: pd.DataFrame) -> np.ndarray:
        """(n, 4) probabilities ordered as DRIVE_CLASSES."""
        proba = self.drive_model.predict_proba(design_matrix(matchups))
        classes = list(self.drive_model.classes_)
        return proba[:, [classes.index(c) for c in DRIVE_CLASSES]]

    def expected_drives(self, pace: float) -> float:
        return self.drives_intercept + self.drives_slope * pace
