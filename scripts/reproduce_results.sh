#!/usr/bin/env bash
# Reproduce every result reported in the README.
#
#   scripts/reproduce_results.sh            # all runs without an LLM (a few minutes)
#   WITH_OLLAMA=1 scripts/reproduce_results.sh   # also the agents run (needs `ollama serve`
#                                                # and `ollama pull gemma4:e4b`; ~10 min,
#                                                # instant when replayed from the LLM cache)
#
# Results go to results/runs/; `stratarena summary` prints the tables.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=results/runs

run() { uv run stratarena run --config "configs/$1.yaml" --seed "$2" --out "$OUT/$1" --quiet; }

for seed in 42 7 123; do run binpacking "$seed"; run binpacking-triplets "$seed"; done
for seed in 42 1 2 3; do run gridbots "$seed"; run gridbots-sparring "$seed"; done
if [[ "${WITH_OLLAMA:-0}" == "1" ]]; then run binpacking-agents 42; fi

for group in "$OUT"/*/; do
  echo "## $(basename "$group")"
  uv run stratarena summary "$group"
  echo
done
