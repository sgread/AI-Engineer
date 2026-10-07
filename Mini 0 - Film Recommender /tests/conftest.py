import numpy as np
import polars as pl
import pytest

from recsys.prepare import add_sessions


@pytest.fixture
def interactions() -> pl.DataFrame:
    """Synthetic users with signup bursts and later sessions, plus a few one-session users."""
    rng = np.random.default_rng(0)
    rows = []
    for u in range(1, 61):
        t = 1_000_000 + u * 10_000_000
        n_sessions = 1 if u % 10 == 0 else int(rng.integers(3, 7))
        films = rng.permutation(400)[:120]
        i = 0
        for sess in range(n_sessions):
            size = 25 if sess == 0 else int(rng.integers(1, 5))
            for _ in range(size):
                # signup burst: many ratings in the same second
                ts = t if sess == 0 else t + int(rng.integers(0, 600))
                rows.append((u, int(films[i]) + 1, float(rng.choice([2.0, 3.0, 4.0, 4.5, 5.0])), ts))
                i += 1
            t += 7 * 24 * 3600
    df = pl.DataFrame(rows, schema={"user_id": pl.Int32, "movie_id": pl.Int32,
                                    "rating": pl.Float32, "ts": pl.Int64}, orient="row")
    return add_sessions(df, 30 * 60)
