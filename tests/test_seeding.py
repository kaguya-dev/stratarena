from stratarena.seeding import Stream, derive_seed, make_rng


def test_same_path_gives_same_seed() -> None:
    assert derive_seed(7, Stream.TRAIN, 0, 1, 2) == derive_seed(7, Stream.TRAIN, 0, 1, 2)


def test_paths_and_roots_are_independent() -> None:
    base = derive_seed(7, Stream.TRAIN, 0, 1, 2)
    assert base != derive_seed(7, Stream.TRAIN, 0, 1, 3)
    assert base != derive_seed(8, Stream.TRAIN, 0, 1, 2)


def test_train_and_holdout_streams_do_not_overlap() -> None:
    train = {
        derive_seed(1, Stream.TRAIN, g, r, i) for g in range(3) for r in range(5) for i in range(8)
    }
    holdout = {derive_seed(1, Stream.HOLDOUT, i) for i in range(200)}
    assert train.isdisjoint(holdout)


def test_rng_is_reproducible() -> None:
    a = make_rng(3, Stream.INIT, 0).random(5)
    b = make_rng(3, Stream.INIT, 0).random(5)
    assert (a == b).all()


def test_seed_fits_in_signed_64_bits() -> None:
    assert 0 <= derive_seed(2**63, Stream.TRAIN, 0) < 2**63
