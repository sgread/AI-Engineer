"""films_base + TMDB cache -> data/processed/films.parquet.

This is the input to Step 7. It holds raw fields only. Deciding which of them
become the text you embed (and how) is deliberately left to you.
"""

from __future__ import annotations

import polars as pl

from .config import PROCESSED
from .tmdb import read_cache

TMDB_SCHEMA = {
    "tmdb_id": pl.Int64,
    "tmdb_title": pl.Utf8,
    "overview": pl.Utf8,
    "tagline": pl.Utf8,
    "release_date": pl.Utf8,
    "runtime": pl.Int32,
    "original_language": pl.Utf8,
    "tmdb_genres": pl.List(pl.Utf8),
    "keywords": pl.List(pl.Utf8),
    "directors": pl.List(pl.Utf8),
    "top_cast": pl.List(pl.Utf8),
    "vote_count": pl.Int64,
    "vote_average": pl.Float64,
    "popularity": pl.Float64,
}


def build_films() -> pl.DataFrame:
    base = pl.read_parquet(PROCESSED / "films_base.parquet")
    rows = [{"tmdb_id": k, **v["data"]} for k, v in read_cache().items() if v["status"] == 200]
    tmdb = pl.DataFrame(rows, schema=TMDB_SCHEMA) if rows else pl.DataFrame(schema=TMDB_SCHEMA)
    films = base.join(tmdb, on="tmdb_id", how="left").with_columns(
        pl.col("overview").is_not_null().alias("has_overview")
    )
    films.write_parquet(PROCESSED / "films.parquet")

    rated = films.filter(pl.col("n_ratings") > 0)
    share_films = rated["has_overview"].mean() or 0.0
    total = rated["n_ratings"].sum() or 1
    share_ratings = rated.filter(pl.col("has_overview"))["n_ratings"].sum() / total
    print(
        f"films: {films.height:,} total | rated films with an overview: {share_films:.1%} "
        f"| share of ratings those films cover: {share_ratings:.1%}"
    )
    return films
