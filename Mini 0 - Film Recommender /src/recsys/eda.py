"""Write results/eda.md and a few plots. Reads whatever pipeline outputs exist.

Not a substitute for poking around yourself, but it answers the questions the
build guide asks in Steps 2-4, so you can get to the RAG work quickly.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from .config import PROCESSED, RESULTS, SETTINGS
from .split import SPLIT_DIR

QS = [0.0, 0.25, 0.5, 0.75, 0.95, 1.0]
QNAMES = ["min", "p25", "median", "p75", "p95", "max"]


def _quantiles(s: pl.Series) -> str:
    vals = [s.quantile(q, "nearest") for q in QS]
    head = "| " + " | ".join(QNAMES) + " |\n|" + "---|" * len(QS) + "\n"
    return head + "| " + " | ".join(f"{v:,.0f}" for v in vals) + " |\n"


def _date(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def report() -> None:
    out_dir = RESULTS / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    inter = pl.read_parquet(PROCESSED / "interactions.parquet")
    lines: list[str] = ["# EDA report", ""]

    per_user = inter.group_by("user_id").len()["len"]
    per_film = inter.group_by("movie_id").len()["len"]
    lines += [
        "## Size", "",
        f"- Users: {inter['user_id'].n_unique():,}",
        f"- Films with at least one rating: {inter['movie_id'].n_unique():,}",
        f"- Ratings: {inter.height:,}",
        f"- Date range: {_date(inter['ts'].min())} to {_date(inter['ts'].max())}",
        "", "## Ratings per user", "", _quantiles(per_user),
        "## Ratings per film", "", _quantiles(per_film),
    ]

    top_share = per_film.sort(descending=True).head(max(1, per_film.len() // 100)).sum() / inter.height
    lines.append(f"The most-rated 1% of films receive {top_share:.0%} of all ratings.\n")

    dist = inter.group_by("rating").len().sort("rating")
    lines += ["## Rating values", "", "| rating | share |", "|---|---|"]
    lines += [f"| {r:g} | {n / inter.height:.1%} |" for r, n in dist.iter_rows()]
    pos_share = (inter["rating"] >= SETTINGS.positive_threshold).mean()
    lines += ["", f"At threshold {SETTINGS.positive_threshold:g}, {pos_share:.0%} of ratings count as positive.", ""]

    sessions = inter.group_by("user_id").agg(pl.col("session").max() + 1)["session"]
    first_sess = (inter["session"] == 0).mean()
    lines += [
        f"## Sessions (gap > {SETTINGS.session_gap_s // 60} min starts a new one)", "",
        _quantiles(sessions),
        f"- Users with only one session: {(sessions == 1).mean():.0%}",
        f"- Ratings made in a user's first session: {first_sess:.0%} (signup bulk-rating)",
        "",
    ]

    films_path = PROCESSED / "films.parquet"
    if films_path.exists():
        films = pl.read_parquet(films_path).filter(pl.col("n_ratings") > 0)
        cov = films.filter(pl.col("has_overview"))["n_ratings"].sum() / max(films["n_ratings"].sum(), 1)
        lines += [
            "## Text coverage", "",
            f"- Rated films with a TMDB overview: {films['has_overview'].mean():.1%}",
            f"- Share of ratings those films cover: {cov:.1%}",
            f"- Median overview length (words): "
            f"{films['overview'].drop_nulls().str.split(' ').list.len().median():.0f}",
            f"- Films with keywords: {(films['keywords'].list.len().fill_null(0) > 0).mean():.1%}",
            "",
        ]

    meta_path = SPLIT_DIR / "meta.json"
    if meta_path.exists():
        m = json.loads(meta_path.read_text())
        lines += [
            "## Split", "",
            f"- Eligible (evaluated) users: {m['users_eligible']:,} of {m['users_total']:,}",
            f"- Train rows: {m['train_rows']:,} | train_full rows: {m['train_full_rows']:,}",
            f"- Validation targets by bucket: {m['val_target_buckets']}",
            f"- Test targets by bucket: {m['test_target_buckets']}",
            "",
            "Cold targets can't be scored by pure CF at all. Tail targets are where",
            "the hybrid should earn its keep.",
            "",
        ]

    # Plots
    fig, ax = plt.subplots(figsize=(6, 4))
    ranked = per_film.sort(descending=True).to_numpy()
    ax.loglog(range(1, len(ranked) + 1), ranked)
    ax.set(xlabel="film rank", ylabel="ratings", title="Long tail: ratings per film")
    fig.tight_layout(); fig.savefig(out_dir / "long_tail.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(per_user.to_numpy(), bins=50, log=True)
    ax.set(xlabel="ratings per user", ylabel="users (log)", title="Ratings per user")
    fig.tight_layout(); fig.savefig(out_dir / "ratings_per_user.png", dpi=120); plt.close(fig)

    years = inter.select(pl.from_epoch("ts").dt.year().alias("y")).group_by("y").len().sort("y")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(years["y"].to_numpy(), years["len"].to_numpy())
    ax.set(xlabel="year", ylabel="ratings", title="Ratings over time")
    fig.tight_layout(); fig.savefig(out_dir / "ratings_over_time.png", dpi=120); plt.close(fig)

    lines += ["## Plots", "", "![](eda/long_tail.png)", "![](eda/ratings_per_user.png)",
              "![](eda/ratings_over_time.png)", ""]
    (RESULTS / "eda.md").write_text("\n".join(lines))
    print(f"eda: wrote {RESULTS / 'eda.md'}")
