import pytest

from stratarena.rating import ELO_MEAN, PairStats, bradley_terry


def test_stronger_player_gets_higher_rating() -> None:
    stats = PairStats()
    stats.record("a", "b", 9.0, 10)
    stats.record("b", "c", 8.0, 10)
    stats.record("a", "c", 10.0, 10)
    r = stats.ratings(["a", "b", "c"])
    assert r["a"] > r["b"] > r["c"]
    assert sum(r.values()) / 3 == pytest.approx(ELO_MEAN)


def test_equal_results_give_equal_ratings() -> None:
    stats = PairStats()
    stats.record("a", "b", 5.0, 10)
    r = stats.ratings(["a", "b"])
    assert r["a"] == pytest.approx(r["b"])


def test_prior_keeps_perfect_records_finite() -> None:
    stats = PairStats()
    stats.record("a", "b", 4.0, 4)
    r = stats.ratings(["a", "b"])
    assert 0 < r["a"] - r["b"] < 1000


def test_record_is_symmetric_and_merge_adds() -> None:
    s1 = PairStats()
    s1.record("a", "b", 1.5, 2)
    assert s1.points[("b", "a")] == 0.5
    assert s1.games[("a", "b")] == s1.games[("b", "a")] == 2
    s2 = PairStats()
    s2.record("a", "b", 2.0, 2)
    s1.merge(s2)
    assert s1.points[("a", "b")] == 3.5
    assert s1.games[("a", "b")] == 4


def test_ratings_ignore_games_outside_ids_and_players_without_games() -> None:
    stats = PairStats()
    stats.record("a", "b", 2.0, 2)
    stats.record("a", "c", 0.0, 2)
    r = stats.ratings(["b", "c"])
    assert r == {"b": ELO_MEAN, "c": ELO_MEAN}
    assert bradley_terry([], stats) == {}
