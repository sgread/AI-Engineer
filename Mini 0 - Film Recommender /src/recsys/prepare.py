"""Raw MovieLens CSVs -> data/processed/{interactions,films_base}.parquet.

interactions: user_id, movie_id, rating, ts (unix seconds), session (0-based per user)
films_base:   movie_id, title, clean_title, year, genres (list), imdb_id, tmdb_id, n_ratings
"""

from __future__ import annotations

import polars as pl

from .config import PROCESSED, SETTINGS, Settings, ensure_dirs
from .download import dataset_dir

YEAR_RE = r"\((\d{4})\)\s*$"


def load_ratings(name: str) -> pl.DataFrame:
    return pl.read_csv(
        dataset_dir(name) / "ratings.csv",
        schema_overrides={"userId": pl.Int32, "movieId": pl.Int32, "rating": pl.Float32, "timestamp": pl.Int64},
    ).rename({"userId": "user_id", "movieId": "movie_id", "timestamp": "ts"})


def subsample_users(df: pl.DataFrame, n: int | None, seed: int) -> pl.DataFrame:
    if n is None:
        return df
    users = df.select("user_id").unique().sort("user_id")
    if n >= users.height:
        return df
    keep = users.sample(n=n, seed=seed)
    return df.join(keep, on="user_id", how="semi")


def add_sessions(df: pl.DataFrame, gap_s: int) -> pl.DataFrame:
    """Number each user's sessions: a new session starts after a gap > gap_s.

    Ties (same timestamp) stay in the same session, so bulk-rated signup
    batches become one session and never get split into fake "next items".
    """
    df = df.sort(["user_id", "ts", "movie_id"])
    gap = pl.col("ts") - pl.col("ts").shift(1).over("user_id")
    new = (gap.is_null() | (gap > gap_s)).cast(pl.Int32)
    return df.with_columns((new.cum_sum().over("user_id") - 1).alias("session"))


def build_interactions(name: str, s: Settings = SETTINGS) -> pl.DataFrame:
    df = load_ratings(name)
    df = subsample_users(df, s.max_users, s.seed)
    return add_sessions(df, s.session_gap_s)


def build_films_base(name: str, interactions: pl.DataFrame) -> pl.DataFrame:
    d = dataset_dir(name)
    movies = pl.read_csv(d / "movies.csv", schema_overrides={"movieId": pl.Int32})
    links = pl.read_csv(
        d / "links.csv",
        schema_overrides={"movieId": pl.Int32, "imdbId": pl.Utf8, "tmdbId": pl.Int64},
    )
    counts = interactions.group_by("movie_id").agg(pl.len().alias("n_ratings"))

    return (
        movies.rename({"movieId": "movie_id"})
        .with_columns(
            pl.col("title").str.extract(YEAR_RE, 1).cast(pl.Int32, strict=False).alias("year"),
            pl.col("title").str.replace(YEAR_RE, "").str.strip_chars().alias("clean_title"),
            pl.when(pl.col("genres") == "(no genres listed)")
            .then(pl.lit(""))
            .otherwise(pl.col("genres"))
            .str.split("|")
            .list.eval(pl.element().filter(pl.element() != ""))
            .alias("genres"),
        )
        .join(
            links.rename({"movieId": "movie_id"}).select(
                "movie_id",
                ("tt" + pl.col("imdbId").str.zfill(7)).alias("imdb_id"),
                pl.col("tmdbId").alias("tmdb_id"),
            ),
            on="movie_id",
            how="left",
        )
        .join(counts, on="movie_id", how="left")
        .with_columns(pl.col("n_ratings").fill_null(0).cast(pl.Int32))
        .select("movie_id", "title", "clean_title", "year", "genres", "imdb_id", "tmdb_id", "n_ratings")
        .sort("movie_id")
    )


def prepare(name: str, s: Settings = SETTINGS) -> None:
    ensure_dirs()
    inter = build_interactions(name, s)
    inter.write_parquet(PROCESSED / "interactions.parquet")
    films = build_films_base(name, inter)
    films.write_parquet(PROCESSED / "films_base.parquet")
    print(
        f"interactions: {inter.height:,} rows, {inter['user_id'].n_unique():,} users, "
        f"{inter['movie_id'].n_unique():,} films rated\n"
        f"films_base:   {films.height:,} films ({films['tmdb_id'].is_null().sum():,} without a TMDB id)"
    )
