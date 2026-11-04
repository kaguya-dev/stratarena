"""Deterministic seed derivation.

Every random choice in a run comes from a ``SeedSequence`` derived from the root seed,
a stream and a path of integers (generation, round, index, ...). Results therefore never
depend on execution order or on the number of worker processes, and the training and
held-out streams can never produce the same instance seeds by construction.
"""

from enum import IntEnum

import numpy as np


class Stream(IntEnum):
    """Independent namespaces of randomness inside one run."""

    TRAIN = 0
    HOLDOUT = 1
    INIT = 2
    BOOTSTRAP = 3
    VARIATION = 4


def seed_sequence(root: int, stream: Stream, *path: int) -> np.random.SeedSequence:
    """Return the ``SeedSequence`` at ``(stream, *path)`` under ``root``."""
    return np.random.SeedSequence(root, spawn_key=(int(stream), *path))


def make_rng(root: int, stream: Stream, *path: int) -> np.random.Generator:
    """Return a generator seeded at ``(stream, *path)`` under ``root``."""
    return np.random.default_rng(seed_sequence(root, stream, *path))


def derive_seed(root: int, stream: Stream, *path: int) -> int:
    """Return a 63-bit integer seed at ``(stream, *path)``, e.g. to generate an instance."""
    state = seed_sequence(root, stream, *path).generate_state(1, np.uint64)
    return int(state[0] >> np.uint64(1))
