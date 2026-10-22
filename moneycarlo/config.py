"""Paths and default model parameters."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(os.environ.get("MONEYCARLO_HOME", Path(__file__).resolve().parent.parent))
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = Path(os.environ.get("MONEYCARLO_DB", DATA_DIR / "moneycarlo.duckdb"))

MODEL_VERSION = "nfl-drive-sim-0.1"

# Number of past seasons used to train the drive-outcome model.
TRAIN_SEASONS = 10


@dataclass
class SimParams:
    """Tunable knobs of the simulation. `calibrate` searches over the dispersion ones."""

    n_sims: int = 10_000
    # Std-dev of the per-game "form" shock applied to each offense (logit scale).
    # Larger -> fatter score distributions. Tuned by `moneycarlo calibrate`.
    form_sigma: float = 0.0
    # Game-script feedback: logit shift of a drive's TD/FG chances per 7 points of lead
    # (positive -> leaders score less, trailers more). Tuned by `moneycarlo calibrate`.
    script_kappa: float = 0.10
    # Scale on the drive-count spread. Extra drives tend to be shorter and less productive,
    # so the raw spread of drive counts overstates the spread of game totals.
    drive_count_sd_scale: float = 0.6
    # How strongly the form shock moves yards per play (multiplicative, log scale).
    form_yards_gamma: float = 0.45
    # Game-to-game noise on offensive snaps and on pass/run mix (logit scale).
    plays_noise: float = 5.0
    pass_rate_noise: float = 0.3
    # Game-to-game noise on yardage efficiency (log scale).
    pass_yards_noise: float = 0.16
    rush_yards_noise: float = 0.22
    # Dirichlet concentration for game-to-game usage variation among players.
    target_concentration: float = 35.0
    carry_concentration: float = 25.0
    # Player-level yardage efficiency noise (log scale).
    player_yards_noise: float = 0.35
    # Rating decay: weight of a game N games ago is decay**N.
    rating_decay: float = 0.90
    # Share of last season's rating (relative to league average) carried into a new season.
    season_carryover: float = 0.55
    # Weight of a training season N seasons old is train_season_decay**N.
    train_season_decay: float = 0.75
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SimParams":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)
