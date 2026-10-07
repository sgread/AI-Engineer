"""Fetch TMDB details (overview, keywords, credits) for every film, resumably.

Each response is reduced to the fields we might want and appended as one JSON
line to data/cache/tmdb.jsonl. Rerunning skips ids already in the cache, so a
crash or Ctrl-C costs nothing. 404s are cached too, so they aren't retried.

This product uses the TMDB API but is not endorsed or certified by TMDB.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import httpx
import polars as pl
from dotenv import load_dotenv
from tqdm import tqdm

from .config import CACHE, PROCESSED, SETTINGS, Settings, ensure_dirs

BASE = "https://api.themoviedb.org/3"
CACHE_FILE = CACHE / "tmdb.jsonl"
MAX_ATTEMPTS = 6


def auth() -> tuple[dict[str, str], dict[str, str]]:
    """Return (headers, params) for whichever credential is configured."""
    load_dotenv()
    token = os.environ.get("TMDB_READ_TOKEN", "").strip()
    key = os.environ.get("TMDB_API_KEY", "").strip()
    if token:
        return {"Authorization": f"Bearer {token}"}, {}
    if key:
        return {}, {"api_key": key}
    raise RuntimeError("Set TMDB_READ_TOKEN or TMDB_API_KEY in .env (see .env.example)")


def compact(j: dict[str, Any]) -> dict[str, Any]:
    """Keep only what could plausibly become a text feature or a filter."""
    credits = j.get("credits") or {}
    cast = sorted(credits.get("cast") or [], key=lambda c: c.get("order", 999))
    return {
        "tmdb_title": j.get("title"),
        "overview": (j.get("overview") or "").strip() or None,
        "tagline": (j.get("tagline") or "").strip() or None,
        "release_date": j.get("release_date") or None,
        "runtime": j.get("runtime") or None,
        "original_language": j.get("original_language"),
        "tmdb_genres": [g["name"] for g in j.get("genres") or []],
        "keywords": [k["name"] for k in (j.get("keywords") or {}).get("keywords") or []],
        "directors": [c["name"] for c in credits.get("crew") or [] if c.get("job") == "Director"],
        "top_cast": [c["name"] for c in cast[:5]],
        # Present-day popularity: handy, but see DECISIONS.md before using as a feature.
        "vote_count": j.get("vote_count"),
        "vote_average": j.get("vote_average"),
        "popularity": j.get("popularity"),
    }


def read_cache(path: Path = CACHE_FILE) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a half-written last line from a crash
            out[rec["tmdb_id"]] = rec
    return out


async def fetch_one(client: httpx.AsyncClient, tmdb_id: int, sem: asyncio.Semaphore,
                    params: dict[str, str]) -> dict[str, Any] | None:
    q = {**params, "append_to_response": "keywords,credits", "language": "en-US"}
    for attempt in range(MAX_ATTEMPTS):
        try:
            async with sem:
                r = await client.get(f"/movie/{tmdb_id}", params=q)
        except httpx.TransportError:
            await asyncio.sleep(2 ** attempt + random.random())
            continue
        if r.status_code == 200:
            return {"tmdb_id": tmdb_id, "status": 200, "data": compact(r.json())}
        if r.status_code == 404:
            return {"tmdb_id": tmdb_id, "status": 404, "data": None}
        if r.status_code == 401:
            raise RuntimeError("TMDB rejected the credentials (401). Check .env.")
        if r.status_code == 429 or r.status_code >= 500:
            wait = float(r.headers.get("retry-after", 2 ** attempt))
            await asyncio.sleep(wait + random.random())
            continue
        return {"tmdb_id": tmdb_id, "status": r.status_code, "data": None}
    return None  # gave up; not cached, so the next run retries it


def _chunks(xs: list[int], n: int) -> Iterable[list[int]]:
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


async def _fetch_all(ids: list[int], concurrency: int, transport: httpx.AsyncBaseTransport | None = None,
                     cache_file: Path = CACHE_FILE) -> int:
    headers, params = auth() if transport is None else ({}, {})
    sem = asyncio.Semaphore(concurrency)
    written = 0
    async with httpx.AsyncClient(base_url=BASE, headers=headers, timeout=30, transport=transport) as client:
        with open(cache_file, "a") as f, tqdm(total=len(ids), desc="tmdb", unit="film") as bar:
            for batch in _chunks(ids, 500):
                for coro in asyncio.as_completed([fetch_one(client, i, sem, params) for i in batch]):
                    rec = await coro
                    if rec is not None:
                        f.write(json.dumps(rec) + "\n")
                        written += 1
                    bar.update(1)
                f.flush()
    return written


def ids_to_fetch(s: Settings = SETTINGS) -> list[int]:
    films = pl.read_parquet(PROCESSED / "films_base.parquet")
    wanted = (
        films.filter(pl.col("tmdb_id").is_not_null() & (pl.col("n_ratings") >= s.tmdb_min_ratings))
        .sort("n_ratings", descending=True)  # most-rated first: useful even if you stop early
        ["tmdb_id"].unique(maintain_order=True).to_list()
    )
    done = read_cache()
    return [i for i in wanted if i not in done]


def fetch(s: Settings = SETTINGS) -> None:
    ensure_dirs()
    todo = ids_to_fetch(s)
    if not todo:
        print("tmdb: cache is complete")
        return
    print(f"tmdb: {len(todo):,} films to fetch (most-rated first; safe to Ctrl-C and rerun)")
    n = asyncio.run(_fetch_all(todo, s.tmdb_concurrency))
    print(f"tmdb: cached {n:,} new responses in {CACHE_FILE}")
