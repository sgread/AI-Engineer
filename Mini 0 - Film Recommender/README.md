# Mini 0: film recommender — starter kit

This kit does Steps 1–4 of the build guide (setup, data, cleaning, split) so
your time goes into the parts worth learning: text features, embeddings, the
vector store, the eval harness and the hybrid. Every data decision it makes is
explained in `DECISIONS.md` and set in `src/recsys/config.py`.

## Quickstart

```bash
uv sync --extra dev                 # or: python -m venv .venv && pip install -e ".[dev]"
cp .env.example .env                # add your TMDB token (free account)
uv run pytest                       # 11 tests, a few seconds

uv run recsys all                   # small dataset, end to end, a few minutes
open results/eda.md                 # read this before anything else
```

For real numbers, switch to the 32M release. The TMDB fetch is the slow part
(tens of thousands of films); it's resumable, so Ctrl-C and rerun freely:

```bash
uv run recsys all --dataset ml-32m --tmdb-min-ratings 5 --max-users 20000
```

`--tmdb-min-ratings` skips barely-rated films; `--max-users` takes a seeded
subsample to keep iteration fast. Drop both for the final run if time allows.
Each step can also run on its own: `download`, `prepare`, `tmdb`, `films`,
`split`, `eda`.

## What you get

```
data/processed/
  interactions.parquet    user_id, movie_id, rating, ts, session
  films.parquet           movie_id, title, clean_title, year, genres, imdb_id, tmdb_id,
                          n_ratings, overview, tagline, keywords, directors, top_cast,
                          runtime, original_language, release_date, has_overview, ...
  split/                  train, train_full, val_targets, test_targets, item_stats, meta.json
data/cache/tmdb.jsonl     raw TMDB cache (never refetched)
results/eda.md            the numbers Steps 2–4 ask for, plus plots
```

## Where you take over

```python
from recsys.data import load_eval_set, load_films

ev = load_eval_set("val")    # always "val" until Step 11
ev.history                   # what a model may learn from
ev.targets                   # one held-out liked film per evaluated user
ev.buckets                   # head / tail / cold per film (from ev.history only)

films = load_films()         # raw fields; turning them into embedding text is Step 7
```

Suggested layout for your own code: `src/recsys/eval.py` (Step 5),
`src/recsys/models/` (one file per model), `src/recsys/index.py` (Step 7).

## Attribution

Uses MovieLens data from GroupLens Research (see the dataset README for terms).
This product uses the TMDB API but is not endorsed or certified by TMDB.
