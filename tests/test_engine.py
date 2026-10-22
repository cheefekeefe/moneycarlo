import numpy as np
import pandas as pd

from moneycarlo.config import SimParams
from moneycarlo.nfl.pipeline import crps
from moneycarlo.nfl.players import allocate
from moneycarlo.nfl.ratings import compute_ratings
from moneycarlo.nfl.simulate import GameInputs, TeamInputs, simulate_game


def _team(td=0.22):
    return TeamInputs(drive_probs=np.array([td, 0.15, 0.12, 1 - td - 0.27]), pass_rate_logit=0.3,
                      sack_rate=0.065, comp_pct=0.65, ypcomp=11.0, ypc=4.3, ppd_factor=1.0,
                      int_share=0.6, pass_td_share=0.62)


def _game(home_td=0.22, away_td=0.22):
    return GameInputs(home=_team(home_td), away=_team(away_td), expected_drives=21.5, drives_sd=2.5,
                      ppd_by_outcome={"TD": 7.5, "FG": 7.9, "TO": 4.1, "NS": 4.3}, p_def_td=0.08,
                      p_st_td=0.005, scoring={"p_xp": 0.94, "p_go2": 0.08, "p_2pt": 0.48},
                      script_beta=-0.02)


def test_simulated_scores_are_realistic():
    sim = simulate_game(_game(), SimParams(n_sims=20_000), np.random.default_rng(0))
    total = sim["home_pts"] + sim["away_pts"]
    assert 40 < total.mean() < 52
    assert (sim["home_pts"] >= 0).all()
    # Key numbers: margins of 3 and 7 should be far more common than 5.
    margin = np.abs(sim["home_pts"] - sim["away_pts"])
    assert (margin == 3).mean() > 1.5 * (margin == 5).mean()
    assert (margin == 7).mean() > 1.5 * (margin == 5).mean()


def test_better_offense_wins_more():
    sim = simulate_game(_game(home_td=0.30, away_td=0.18), SimParams(n_sims=20_000), np.random.default_rng(1))
    assert (sim["home_pts"] > sim["away_pts"]).mean() > 0.65


def test_player_lines_add_up_to_team():
    rng = np.random.default_rng(2)
    sim = simulate_game(_game(), SimParams(n_sims=2_000), rng)
    box = sim["home_box"]
    prof = pd.DataFrame({
        "player_id": ["qb", "wr1", "wr2", "te", "rb"],
        "name": ["QB", "WR1", "WR2", "TE", "RB"],
        "position": ["QB", "WR", "WR", "TE", "RB"],
        "is_starting_qb": [True, False, False, False, False],
        "target_share": [0.0, 0.28, 0.2, 0.18, 0.12],
        "carry_share": [0.12, 0.0, 0.02, 0.0, 0.7],
        "catch_rate": [0.6, 0.65, 0.62, 0.72, 0.78],
        "ypr": [10, 13, 12, 10, 8], "ypc": [5.5, 6, 6, 4, 4.3],
        "td_per_target": [0.05] * 5, "td_per_carry": [0.03] * 5,
    })
    out = allocate(box, prof, SimParams(), rng)
    rec_yds = sum(out[p]["rec_yds"] for p in out if "rec_yds" in out[p])
    assert (rec_yds <= out["qb"]["pass_yds"] + 5).all()  # remainder goes to unlisted players
    targets = sum(out[p]["targets"] for p in out if "targets" in out[p])
    assert (targets <= box["attempts"]).all()
    rush_att = sum(out[p]["rush_att"] for p in out)
    assert (rush_att <= box["carries"]).all()
    assert (out["qb"]["pass_att"] == box["attempts"]).all()
    for p in out:
        if "receptions" in out[p]:
            assert (out[p]["receptions"] <= out[p]["targets"]).all()


def test_ratings_have_no_lookahead():
    sched = pd.DataFrame({
        "game_id": ["g1", "g2"], "season": [2024, 2024], "week": [1, 2],
        "gameday": pd.to_datetime(["2024-09-08", "2024-09-15"]), "gametime": ["13:00", "13:00"],
        "home_team": ["A", "A"], "away_team": ["B", "B"], "home_score": [30, 10], "away_score": [10, 30],
        "completed": [True, True],
    })
    row = dict(plays=60, epa_sum=0.0, success_sum=27, attempts=32, sacks=2, carries=26, completions=21,
               pass_yards=230, rush_yards=110, interceptions=1, fumbles_lost=0, pass_td=1, rush_td=1,
               drives=11, d_td=2, d_fg=2, d_to=1, d_ns=6, drive_plays=60, season=2024, week=1)
    tg = pd.DataFrame([
        {**row, "game_id": "g1", "team": "A", "opp": "B", "epa_sum": 12.0},
        {**row, "game_id": "g1", "team": "B", "opp": "A", "epa_sum": -6.0},
        {**row, "game_id": "g2", "team": "A", "opp": "B", "week": 2},
        {**row, "game_id": "g2", "team": "B", "opp": "A", "week": 2},
    ])
    ratings, _ = compute_ratings(sched, tg)
    r = ratings.set_index(["game_id", "team"])
    # Before game 1 both teams are league average; game 1 then separates them.
    assert r.loc[("g1", "A"), "off_epa"] == r.loc[("g1", "B"), "off_epa"]
    assert r.loc[("g2", "A"), "off_epa"] > r.loc[("g2", "B"), "off_epa"]


def test_crps_prefers_sharp_correct_forecast():
    rng = np.random.default_rng(3)
    assert crps(rng.normal(3, 2, 5000), 3) < crps(rng.normal(3, 10, 5000), 3)
    assert crps(rng.normal(3, 2, 5000), 3) < crps(rng.normal(10, 2, 5000), 3)
