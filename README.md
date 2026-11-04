# stratarena

Choose strategies by simulation. stratarena generates candidate strategies for a problem,
puts them through battle-royale arenas on fresh random instances, eliminates the weakest,
hybridizes the best and validates the winners on held-out instances.

> Status: early development (milestone 4 of 5). The full README, with architecture,
> results and roadmap, comes with the later milestones.

## Quick start

```bash
uv sync
uv run stratarena run --problem binpacking --config configs/binpacking.yaml
uv run stratarena report runs/<run_dir>   # regenerate report.md and evolution.png
```

Each run writes `config.yaml`, `run.json`, `report.md` and `evolution.png` to
`runs/<problem>-seed<seed>-<config hash>/`. The same seed and configuration produce a
byte-identical `run.json`.

## Development

```bash
uv sync
uv run pytest
uv run ruff check . && uv run mypy src
```

## Built with AI

This project is developed with AI assistance. Built with Claude Code.

## License

MIT
