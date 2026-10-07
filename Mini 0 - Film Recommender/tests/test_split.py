from dataclasses import replace

import polars as pl

from recsys.config import SETTINGS
from recsys.prepare import add_sessions
from recsys.split import buckets, make_split

S = replace(SETTINGS, min_history_positives=3)


def test_sessions_keep_ties_together_and_split_on_gaps():
    df = pl.DataFrame({"user_id": [1, 1, 1, 1], "movie_id": [1, 2, 3, 4],
                       "rating": [4.0] * 4, "ts": [0, 0, 100, 10_000]})
    out = add_sessions(df, 1800).sort("movie_id")
    assert out["session"].to_list() == [0, 0, 0, 1]


def test_targets_are_not_in_own_history(interactions):
    out = make_split(interactions, S)
    for hist, tgt in (("train", "val_targets"), ("train_full", "test_targets")):
        leak = out[tgt].join(out[hist], on=["user_id", "movie_id"], how="inner")
        assert leak.height == 0, f"{tgt} items appear in {hist}"


def test_val_comes_before_test(interactions):
    out = make_split(interactions, S)
    j = out["val_targets"].join(out["test_targets"], on="user_id", suffix="_t")
    assert j.height == out["test_targets"].height > 0
    assert (j["ts"] < j["ts_t"]).all()


def test_history_is_strictly_before_target(interactions):
    out = make_split(interactions, S)
    for hist, tgt in (("train", "val_targets"), ("train_full", "test_targets")):
        last = out[hist].group_by("user_id").agg(pl.col("ts").max().alias("last"))
        j = out[tgt].join(last, on="user_id")
        assert (j["last"] < j["ts"]).all()


def test_train_is_subset_of_train_full(interactions):
    out = make_split(interactions, S)
    extra = out["train"].join(out["train_full"], on=["user_id", "movie_id"], how="anti")
    assert extra.height == 0


def test_ineligible_users_kept_whole_for_training(interactions):
    out = make_split(interactions, S)
    ineligible = interactions.join(out["test_targets"].select("user_id"), on="user_id", how="anti")
    assert ineligible.height > 0
    kept = ineligible.join(out["train"], on=["user_id", "movie_id"], how="semi")
    assert kept.height == ineligible.height


def test_split_is_deterministic(interactions):
    a, b = make_split(interactions, S), make_split(interactions, S)
    assert a["test_targets"].equals(b["test_targets"])
    assert a["val_targets"].equals(b["val_targets"])


def test_buckets_use_history_only():
    hist = pl.DataFrame({"user_id": [1, 2, 3, 1], "movie_id": [10, 10, 10, 20]})
    b = buckets(hist, pl.Series([10, 20, 30]), head_share=0.5).sort("movie_id")
    assert b["bucket"].to_list() == ["head", "tail", "cold"]
