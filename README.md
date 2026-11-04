# stratarena

Choose strategies by simulation. stratarena generates candidate strategies for a problem,
puts them through battle-royale arenas on fresh random instances, eliminates the weakest,
hybridizes the best and validates the winners on held-out instances.

> Status: early development (milestone 1 of 5). The full README, with architecture,
> results and roadmap, comes with the later milestones.

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
