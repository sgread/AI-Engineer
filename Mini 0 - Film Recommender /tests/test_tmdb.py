import asyncio
import json

import httpx

from recsys import tmdb


def test_fetch_handles_200_404_and_429(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        tid = int(request.url.path.rsplit("/", 1)[1])
        if tid == 1:
            return httpx.Response(200, json={"title": "A", "overview": "Plot.", "keywords": {"keywords": [{"name": "k"}]},
                                             "credits": {"crew": [{"job": "Director", "name": "D"}],
                                                         "cast": [{"name": "X", "order": 1}, {"name": "Y", "order": 0}]}})
        if tid == 2:
            return httpx.Response(404)
        calls["n"] += 1
        return httpx.Response(429, headers={"retry-after": "0"}) if calls["n"] == 1 else httpx.Response(200, json={"title": "C"})

    cache = tmp_path / "tmdb.jsonl"
    n = asyncio.run(tmdb._fetch_all([1, 2, 3], 2, transport=httpx.MockTransport(handler), cache_file=cache))
    assert n == 3
    recs = tmdb.read_cache(cache)
    assert recs[1]["data"]["directors"] == ["D"]
    assert recs[1]["data"]["top_cast"] == ["Y", "X"]
    assert recs[2]["status"] == 404
    assert recs[3]["status"] == 200 and recs[3]["data"]["overview"] is None


def test_read_cache_skips_half_written_line(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps({"tmdb_id": 5, "status": 404, "data": None}) + "\n{\"tmdb_id\": 6, \"sta")
    assert list(tmdb.read_cache(p)) == [5]
