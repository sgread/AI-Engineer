import polars as pl

from recsys import prepare


def test_films_base_parses_titles_genres_and_ids(tmp_path, monkeypatch):
    (tmp_path / "movies.csv").write_text(
        'movieId,title,genres\n1,Toy Story (1995),Adventure|Animation\n'
        '2,"American President, The (1995)",Comedy\n3,Untitled,(no genres listed)\n')
    (tmp_path / "links.csv").write_text("movieId,imdbId,tmdbId\n1,0114709,862\n2,0112346,9087\n3,1234567,\n")
    monkeypatch.setattr(prepare, "dataset_dir", lambda name: tmp_path)
    inter = pl.DataFrame({"user_id": [1, 2], "movie_id": [1, 1]}, schema={"user_id": pl.Int32, "movie_id": pl.Int32})
    f = prepare.build_films_base("x", inter).sort("movie_id")
    assert f["year"].to_list() == [1995, 1995, None]
    assert f["clean_title"].to_list()[1] == "American President, The"
    assert f["genres"].to_list() == [["Adventure", "Animation"], ["Comedy"], []]
    assert f["imdb_id"][0] == "tt0114709" and f["tmdb_id"][2] is None
    assert f["n_ratings"].to_list() == [2, 0, 0]
