#!/usr/bin/env bash
# Reproduce every result reported in the README.
#
#   scripts/reproduce_results.sh
#
# The two agents experiments replay their LLM answers from results/llm-cache/, so no model,
# key or network is needed. Time-limited code strategies depend on CPU speed: run on an
# otherwise idle machine to reproduce them exactly.
#
# Results go to results/runs/; `stratarena summary` prints the tables.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=results/runs

run() { uv run stratarena run --config "configs/$1.yaml" --seed "$2" --out "$OUT/$1" --quiet; }

for seed in 42 7 123; do run binpacking "$seed"; run binpacking-triplets "$seed"; done
for seed in 42 1 2 3; do run gridbots "$seed"; run gridbots-sparring "$seed"; done
run binpacking-agents 42
run binpacking-agents-subagent 42

for group in "$OUT"/*/; do
  echo "## $(basename "$group")"
  uv run stratarena summary "$group"
  echo
done
