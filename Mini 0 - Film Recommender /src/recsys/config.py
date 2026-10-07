"""Every data decision lives here, so changing one is a one-line diff.

Each constant is explained in DECISIONS.md. If you change one, rerun from the
step that uses it (noted beside each) and log the change there.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(os.environ.get("RECSYS_ROOT", Path.cwd()))
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
CACHE = DATA / "cache"
RESULTS = ROOT / "results"

DATASETS = {
    # Develop against the small set; switch to 32M for real numbers.
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "ml-25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
    "ml-32m": "https://files.grouplens.org/datasets/movielens/ml-32m.zip",
}


@dataclass(frozen=True)
class Settings:
    # --- prepare ---------------------------------------------------------
    # Gap between two ratings (seconds) that starts a new session.
    session_gap_s: int = 30 * 60
    # Optional seeded subsample of users, for fast iteration. None = all.
    max_users: int | None = None

    # --- split -----------------------------------------------------------
    # A rating at or above this counts as a positive ("liked").
    positive_threshold: float = 4.0
    # Users need at least this many positives in the validation history
    # to be evaluated (everyone's data is still used for training).
    min_history_positives: int = 5
    # Head = most-rated films covering this share of training interactions.
    head_share: float = 0.8

    # --- tmdb ------------------------------------------------------------
    # Only fetch overviews for films with at least this many ratings.
    tmdb_min_ratings: int = 1
    tmdb_concurrency: int = 8

    seed: int = 42


SETTINGS = Settings()


def ensure_dirs() -> None:
    for p in (RAW, PROCESSED, CACHE, RESULTS, RESULTS / "eda"):
        p.mkdir(parents=True, exist_ok=True)
