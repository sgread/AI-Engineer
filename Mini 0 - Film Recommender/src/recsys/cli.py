"""recsys <command> [--dataset NAME] [--max-users N]

  download   fetch and unzip MovieLens
  prepare    CSV -> interactions.parquet + films_base.parquet
  tmdb       fetch TMDB details into the resumable cache
  films      join TMDB fields -> films.parquet
  split      frozen train / val / test
  eda        results/eda.md + plots
  all        everything above, in order
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from .config import SETTINGS

STEPS = ["download", "prepare", "tmdb", "films", "split", "eda"]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="recsys", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=STEPS + ["all"])
    p.add_argument("--dataset", default="ml-latest-small")
    p.add_argument("--max-users", type=int, default=None, help="seeded user subsample, for speed")
    p.add_argument("--tmdb-min-ratings", type=int, default=None,
                   help="only fetch TMDB data for films with at least this many ratings")
    a = p.parse_args(argv)

    s = SETTINGS
    if a.max_users is not None:
        s = replace(s, max_users=a.max_users)
    if a.tmdb_min_ratings is not None:
        s = replace(s, tmdb_min_ratings=a.tmdb_min_ratings)

    todo = STEPS if a.command == "all" else [a.command]
    for step in todo:
        print(f"\n== {step} ==")
        if step == "download":
            from .download import download
            download(a.dataset)
        elif step == "prepare":
            from .prepare import prepare
            prepare(a.dataset, s)
        elif step == "tmdb":
            from .tmdb import fetch
            fetch(s)
        elif step == "films":
            from .films import build_films
            build_films()
        elif step == "split":
            from .split import run
            run(s)
        elif step == "eda":
            from .eda import report
            report()


if __name__ == "__main__":
    main()
