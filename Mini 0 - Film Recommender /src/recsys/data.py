"""The interface for everything you build from Step 5 onward.

    from recsys.data import load_eval_set, load_films

    ev = load_eval_set("val")   # tune on this
    ev.history   # interactions models may train on / condition on
    ev.targets   # one held-out positive per evaluated user
    ev.buckets   # movie_id, n, bucket ("head"/"tail"/"cold"), from ev.history only
    films = load_films()        # raw fields for Step 7's text features
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Literal

import polars as pl

from .config import PROCESSED
from .split import SPLIT_DIR


@dataclass(frozen=True)
class EvalSet:
    stage: str
    history: pl.DataFrame
    targets: pl.DataFrame
    buckets: pl.DataFrame


def load_eval_set(stage: Literal["val", "test"]) -> EvalSet:
    if stage == "val":
        hist, tgt, n, b = "train", "val_targets", "n_train", "bucket_val"
    elif stage == "test":
        print("note: loading the TEST set. Run this once, at the end (Step 11).", file=sys.stderr)
        hist, tgt, n, b = "train_full", "test_targets", "n_train_full", "bucket_test"
    else:
        raise ValueError("stage must be 'val' or 'test'")
    stats = pl.read_parquet(SPLIT_DIR / "item_stats.parquet")
    return EvalSet(
        stage=stage,
        history=pl.read_parquet(SPLIT_DIR / f"{hist}.parquet"),
        targets=pl.read_parquet(SPLIT_DIR / f"{tgt}.parquet"),
        buckets=stats.select("movie_id", pl.col(n).alias("n"), pl.col(b).alias("bucket")),
    )


def load_films() -> pl.DataFrame:
    return pl.read_parquet(PROCESSED / "films.parquet")
