"""Check a bin packing code strategy on fresh instances, outside the search.

    uv run python scripts/check_champion.py <run_dir>/strategies/<id>.py

Uses seeds 10000+ (never used by any search stream) and compares with best-fit decreasing
on the triplets profile and on the three other profiles.
"""

import sys
from pathlib import Path

import numpy as np

from stratarena.examples.binpacking import BinPacking
from stratarena.strategy import CodeStrategy


def main(path: str) -> None:
    strategy = CodeStrategy(Path(path).stem, Path(path).read_text(encoding="utf-8"), "pack")
    print("| Profiles | Instances | Strategy | best-fit decreasing | Better | Worse | Errors |")
    print("|---|---|---|---|---|---|---|")
    for profiles, n in ((["triplets"], 150), (["small", "large", "bimodal"], 60)):
        problem = BinPacking(profiles=profiles)
        bfd = next(b for b in problem.baselines() if b.id == "best-fit-decreasing")
        ours, theirs, errors = [], [], 0
        for seed in range(10_000, 10_000 + n):
            inst = problem.generate_instance(seed)
            result = problem.evaluate(strategy, inst)
            errors += result.error is not None
            ours.append(result.score)
            theirs.append(problem.evaluate(bfd, inst).score)
        a, b = np.array(ours), np.array(theirs)
        print(
            f"| {', '.join(profiles)} | {n} | {a.mean():.4f} | {b.mean():.4f} "
            f"| {(a > b).sum()} | {(a < b).sum()} | {errors} |"
        )


if __name__ == "__main__":
    main(sys.argv[1])
