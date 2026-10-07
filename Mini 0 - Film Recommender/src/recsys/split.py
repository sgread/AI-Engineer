"""Session-aware leave-last-out split. Run once, then never regenerate.

For each *eligible* user (see DECISIONS.md):
  test target  = one random positive from their last session containing a positive
  val target   = one random positive from the session containing a positive before that
  train        = everything before the val session         -> history for VALIDATION
  train_full   = everything before the test session        -> history for TEST
Other test-session ratings are dropped, so same-session co-ratings can't leak.

Ineligible users contribute all their ratings to train and train_full: they
still teach CF models, they just aren't scored.

Outputs in data/processed/split/:
  train.parquet, train_full.parquet       user_id, movie_id, rating, ts, session
  val_targets.parquet, test_targets.parquet   user_id, movie_id, rating, ts
  item_stats.parquet   movie_id, n_train, bucket_val, n_train_full, bucket_test
  meta.json            settings + counts, for the write-up
"""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import polars as pl

from .config import PROCESSED, SETTINGS, Settings

SPLIT_DIR = PROCESSED / "split"


def _pick_one(df: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """One seeded-random row per user (row order made deterministic first)."""
    df = df.sort(["user_id", "movie_id"])
    return (
        df.with_columns(pl.Series("_r", rng.random(df.height)))
        .sort(["user_id", "_r"])
        .group_by("user_id", maintain_order=True)
        .first()
        .select("user_id", "movie_id", "rating", "ts")
    )


def buckets(history: pl.DataFrame, movie_ids: pl.Series, head_share: float) -> pl.DataFrame:
    """head / tail / cold per film, from training interactions only."""
    counts = (
        pl.DataFrame({"movie_id": movie_ids.unique()})
        .join(history.group_by("movie_id").agg(pl.len().alias("n")), on="movie_id", how="left")
        .with_columns(pl.col("n").fill_null(0).cast(pl.Int64))
        .sort(["n", "movie_id"], descending=[True, False])
    )
    total = max(int(counts["n"].sum()), 1)
    before = (pl.col("n").cum_sum() - pl.col("n")) / total  # share covered before this film
    return counts.with_columns(
        pl.when(pl.col("n") == 0).then(pl.lit("cold"))
        .when(before < head_share).then(pl.lit("head"))
        .otherwise(pl.lit("tail"))
        .alias("bucket")
    )


def make_split(inter: pl.DataFrame, s: Settings = SETTINGS,
               all_movie_ids: pl.Series | None = None) -> dict[str, pl.DataFrame]:
    rng = np.random.default_rng(s.seed)
    pos = inter.filter(pl.col("rating") >= s.positive_threshold)

    last_two = (
        pos.select("user_id", "session").unique()
        .group_by("user_id")
        .agg(pl.col("session").sort().tail(2).alias("s"))
        .filter(pl.col("s").list.len() == 2)
        .select("user_id",
                pl.col("s").list.get(0).alias("v_sess"),
                pl.col("s").list.get(1).alias("t_sess"))
    )
    hist_pos = (
        pos.join(last_two, on="user_id")
        .filter(pl.col("session") < pl.col("v_sess"))
        .group_by("user_id").agg(pl.len().alias("n_hist_pos"))
    )
    eligible = last_two.join(hist_pos, on="user_id").filter(
        pl.col("n_hist_pos") >= s.min_history_positives
    ).select("user_id", "v_sess", "t_sess")

    cand = pos.join(eligible, on="user_id")
    val_t = _pick_one(cand.filter(pl.col("session") == pl.col("v_sess")), rng)
    test_t = _pick_one(cand.filter(pl.col("session") == pl.col("t_sess")), rng)

    marked = inter.join(eligible, on="user_id", how="left")
    cols = ["user_id", "movie_id", "rating", "ts", "session"]
    train = marked.filter(pl.col("v_sess").is_null() | (pl.col("session") < pl.col("v_sess"))).select(cols)
    train_full = marked.filter(pl.col("t_sess").is_null() | (pl.col("session") < pl.col("t_sess"))).select(cols)

    ids = all_movie_ids if all_movie_ids is not None else inter["movie_id"]
    b_val = buckets(train, ids, s.head_share).rename({"n": "n_train", "bucket": "bucket_val"})
    b_test = buckets(train_full, ids, s.head_share).rename({"n": "n_train_full", "bucket": "bucket_test"})
    item_stats = b_val.join(b_test, on="movie_id").sort("movie_id")

    return {"train": train, "train_full": train_full, "val_targets": val_t,
            "test_targets": test_t, "item_stats": item_stats}


def run(s: Settings = SETTINGS) -> None:
    inter = pl.read_parquet(PROCESSED / "interactions.parquet")
    films_path = PROCESSED / "films_base.parquet"
    ids = pl.read_parquet(films_path)["movie_id"] if films_path.exists() else None
    out = make_split(inter, s, ids)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df.write_parquet(SPLIT_DIR / f"{name}.parquet")

    stats = out["item_stats"]
    def target_buckets(t: pl.DataFrame, col: str) -> dict[str, int]:
        j = t.join(stats.select("movie_id", col), on="movie_id")
        return dict(j.group_by(col).len().sort(col).iter_rows())

    meta = {
        "settings": asdict(s),
        "users_total": inter["user_id"].n_unique(),
        "users_eligible": out["test_targets"].height,
        "train_rows": out["train"].height,
        "train_full_rows": out["train_full"].height,
        "val_target_buckets": target_buckets(out["val_targets"], "bucket_val"),
        "test_target_buckets": target_buckets(out["test_targets"], "bucket_test"),
    }
    (SPLIT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "settings"}, indent=2))
