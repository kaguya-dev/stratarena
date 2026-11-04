"""Bootstrap statistics shared by the arena and the held-out validation."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

IndexMatrix = npt.NDArray[np.int64]


def resample_indices(rng: np.random.Generator, n: int, samples: int) -> IndexMatrix:
    """Bootstrap index matrix of shape ``(samples, n)``.

    Reusing one matrix for every strategy evaluated on the same instances makes the
    intervals paired, so differences between strategies are not inflated by the
    variance between instances.
    """
    return rng.integers(0, n, size=(samples, n))


def mean_ci(values: Sequence[float], idx: IndexMatrix) -> tuple[float, float, float]:
    """Mean and 95% percentile bootstrap interval of ``values``."""
    arr = np.asarray(values, dtype=float)
    means = arr[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(arr.mean()), float(low), float(high)
