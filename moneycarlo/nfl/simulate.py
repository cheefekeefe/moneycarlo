"""Vectorised drive-level Monte Carlo simulation of one NFL game.

Per simulated game:
1. Each offense draws a latent "form" shock (how well it plays today).
2. The number of regulation drives is drawn from the pace model and split between teams.
3. Drives are played in order, alternating possession. Each ends in TD / FG / turnover /
   no score, with matchup probabilities nudged by form and by the current score.
4. Points come from those outcomes (PATs, 2pt tries, defensive TDs); ties go to overtime.
5. A team box score (plays, attempts, completions, yards, TDs, INTs) is generated
   consistent with the drives, the form shock and the final game script.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from moneycarlo.config import SimParams

# How the form shock moves each drive-outcome logit (TD, FG, TO, NS).
_FORM_LOADINGS = np.array([1.0, 0.4, -0.6, 0.0])
# How the current lead moves them (scaled by script_kappa per touchdown of lead).
_SCRIPT_LOADINGS = np.array([1.0, 0.5, 0.0, 0.0])


@dataclass
class TeamInputs:
    drive_probs: np.ndarray  # (4,) base P(TD, FG, TO, NS)
    pass_rate_logit: float   # script-neutral pass rate (logit)
    sack_rate: float
    comp_pct: float
    ypcomp: float
    ypc: float
    ppd_factor: float        # team plays-per-drive relative to league
    int_share: float
    pass_td_share: float


@dataclass
class GameInputs:
    home: TeamInputs
    away: TeamInputs
    expected_drives: float
    drives_sd: float
    ppd_by_outcome: dict
    p_def_td: float
    p_st_td: float
    scoring: dict
    script_beta: float
    playoff: bool = False


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _td_points(rng, n_td: np.ndarray, scoring: dict) -> np.ndarray:
    go2 = rng.binomial(n_td, scoring["p_go2"])
    made2 = rng.binomial(go2, scoring["p_2pt"])
    xp = rng.binomial(n_td - go2, scoring["p_xp"])
    return 6 * n_td + 2 * made2 + xp


def _mean_one_lognormal(rng, shift: np.ndarray, noise_sd: float, shift_sd: float) -> np.ndarray:
    """exp(shift + noise) rescaled to have mean one, so noise widens but doesn't bias yardage."""
    var = noise_sd**2 + shift_sd**2
    return np.exp(shift + rng.normal(0, noise_sd, len(shift)) - var / 2)


def simulate_game(g: GameInputs, params: SimParams, rng: np.random.Generator) -> dict:
    S = params.n_sims
    sides = {"home": g.home, "away": g.away}

    z = {s: rng.normal(0.0, params.form_sigma, S) for s in sides}
    probs = {}
    for s, t in sides.items():
        base = np.log(np.clip(t.drive_probs, 1e-6, 1))
        # Lognormal correction keeps the average drive-outcome rates equal to the model's.
        shift = z[s][:, None] * _FORM_LOADINGS[None, :] - 0.5 * (params.form_sigma * _FORM_LOADINGS[None, :]) ** 2
        probs[s] = _softmax_rows(base[None, :] + shift)

    total = np.clip(np.rint(rng.normal(g.expected_drives, g.drives_sd * params.drive_count_sd_scale, S)), 14, 32).astype(int)
    home_first = rng.integers(0, 2, S).astype(bool)
    drives = {"home": total // 2 + (total % 2) * home_first, "away": total // 2 + (total % 2) * ~home_first}

    # Play drives in order, alternating possession, so the score can feed back into each
    # drive: the leading offense gets more conservative, the trailing one presses.
    pts = {s: np.zeros(S, dtype=int) for s in sides}
    counts = {s: np.zeros((S, 4), dtype=int) for s in sides}
    base = {s: np.log(probs[s]) for s in sides}
    rows = np.arange(S)
    for i in range(int(total.max())):
        active = i < total
        home_off = (i % 2 == 0) == home_first
        for s, o, is_off in (("home", "away", home_off), ("away", "home", ~home_off)):
            m = active & is_off
            if not m.any():
                continue
            lead = np.clip(pts[s][m] - pts[o][m], -21, 21) / 7.0
            logits = base[s][m] - params.script_kappa * lead[:, None] * _SCRIPT_LOADINGS[None, :]
            p = _softmax_rows(logits)
            outcome = (rng.random(m.sum())[:, None] > np.cumsum(p, axis=1)[:, :-1]).sum(axis=1)
            idx = rows[m]
            counts[s][idx, outcome] += 1
            pts[s][idx] += np.where(outcome == 0, _td_points(rng, np.ones(len(idx), int), g.scoring), 0)
            pts[s][idx] += np.where(outcome == 1, 3, 0)
            # Return TDs: turnovers returned for scores, and punt / kick-block returns.
            ret = np.where(outcome == 2, rng.random(len(idx)) < g.p_def_td,
                           np.where(outcome == 3, rng.random(len(idx)) < g.p_st_td, False))
            pts[o][idx] += np.where(ret, _td_points(rng, np.ones(len(idx), int), g.scoring), 0)

    # Overtime: both teams get a possession per round until one is ahead after a round
    # (regular season OT is short, so at most two rounds before it ends in a tie).
    tied = pts["home"] == pts["away"]
    rounds = 10 if g.playoff else 2
    for _ in range(rounds):
        if not tied.any():
            break
        idx = np.flatnonzero(tied)
        add = {}
        for s in sides:
            p = probs[s][idx]
            u = rng.random(len(idx))[:, None]
            outcome = (u > np.cumsum(p, axis=1)).sum(axis=1)
            add[s] = np.select([outcome == 0, outcome == 1], [7, 3], 0)
        pts["home"][idx] += add["home"]
        pts["away"][idx] += add["away"]
        tied = pts["home"] == pts["away"]

    out = {"home_pts": pts["home"], "away_pts": pts["away"], "drives": drives, "counts": counts}
    for s, o in (("home", "away"), ("away", "home")):
        margin = pts[s] - pts[o]
        out[f"{s}_box"] = _team_box(rng, sides[s], counts[s], z[s], margin, g, params)
    return out


def _team_box(rng, t: TeamInputs, counts: np.ndarray, z: np.ndarray, margin: np.ndarray,
              g: GameInputs, params: SimParams) -> dict:
    S = len(z)
    ppd = np.array([g.ppd_by_outcome.get(c, 6.0) for c in ("TD", "FG", "TO", "NS")])
    exp_plays = (counts * ppd[None, :]).sum(axis=1) * t.ppd_factor
    plays = np.maximum(np.rint(exp_plays + rng.normal(0, params.plays_noise, S)), counts.sum(axis=1) * 2).astype(int)

    logit = (t.pass_rate_logit + g.script_beta * np.clip(margin, -35, 35)
             + rng.normal(0, params.pass_rate_noise, S))
    pass_rate = 1.0 / (1.0 + np.exp(-logit))
    dropbacks = rng.binomial(plays, pass_rate)
    sacks = rng.binomial(dropbacks, np.clip(t.sack_rate, 0.01, 0.2))
    attempts = dropbacks - sacks
    carries = plays - dropbacks
    completions = rng.binomial(attempts, np.clip(t.comp_pct, 0.35, 0.9))

    pass_yards = completions * t.ypcomp * _mean_one_lognormal(rng, params.form_yards_gamma * z,
                                                                params.pass_yards_noise, params.form_sigma * params.form_yards_gamma)
    rush_yards = carries * t.ypc * _mean_one_lognormal(rng, params.form_yards_gamma * z,
                                                       params.rush_yards_noise, params.form_sigma * params.form_yards_gamma)

    td, to = counts[:, 0], counts[:, 2]
    pass_td = rng.binomial(td, np.clip(t.pass_td_share, 0.05, 0.95))
    rush_td = td - pass_td
    ints = rng.binomial(to, np.clip(t.int_share, 0.05, 0.95))
    return {
        "plays": plays,
        "attempts": attempts,
        "completions": completions,
        "sacks": sacks,
        "pass_yards": np.rint(pass_yards),
        "carries": carries,
        "rush_yards": np.rint(rush_yards),
        "pass_td": pass_td,
        "rush_td": rush_td,
        "interceptions": ints,
        "fumbles_lost": to - ints,
    }
