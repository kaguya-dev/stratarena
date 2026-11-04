"""Command-line interface: ``stratarena run`` and ``stratarena report``."""

import logging
from pathlib import Path
from typing import Annotated, Any

import typer

from stratarena.config import RunConfig, load_config
from stratarena.examples import PROBLEMS, make_problem
from stratarena.log import configure_logging
from stratarena.report import (
    REPORT_MD,
    RUN_JSON,
    render_report,
    run_name,
    summarize_runs,
    write_run,
)
from stratarena.search import run_search
from stratarena.strategy import ParamStrategy

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Choose strategies by simulation: battle-royale arenas and hybridization.",
)


@app.command()
def run(
    problem: Annotated[
        str | None,
        typer.Option(
            help=f"Problem to solve ({', '.join(sorted(PROBLEMS))}). Overrides the config."
        ),
    ] = None,
    config: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False, help="YAML run configuration.")
    ] = None,
    seed: Annotated[int | None, typer.Option(help="Root seed. Overrides the config.")] = None,
    workers: Annotated[
        int | None, typer.Option(help="Worker processes. Overrides the config.")
    ] = None,
    out: Annotated[Path, typer.Option(help="Directory where run folders are created.")] = Path(
        "runs"
    ),
    log_json: Annotated[bool, typer.Option(help="Emit logs as JSON lines.")] = False,
    quiet: Annotated[bool, typer.Option(help="Only print warnings and the summary.")] = False,
) -> None:
    """Run a search and write its artifacts to OUT/<problem>-seed<seed>-<hash>/."""
    if config is None and problem is None:
        raise typer.BadParameter("give --problem, --config or both")
    data: dict[str, Any] = load_config(config).model_dump() if config else {"problem": problem}
    if problem is not None:
        data["problem"] = problem
    if seed is not None:
        data["seed"] = seed
    if workers is not None:
        data["arena"] = {**data.get("arena", {}), "workers": workers}
    cfg = RunConfig.model_validate(data)

    configure_logging(level=logging.WARNING if quiet else logging.INFO, json=log_json)
    result = run_search(make_problem(cfg.problem, cfg.problem_options), cfg)
    run_dir = write_run(result, out / run_name(result))

    v = result.validation
    champ = v.champion
    champion = result.champion
    what = (
        champion.as_dict() if isinstance(champion, ParamStrategy) else f"code ({champion.origin})"
    )
    typer.echo(f"champion   {champion.id}  {what}")
    typer.echo(
        f"held-out   {champ.holdout.mean:.4f} [{champ.holdout.ci_low:.4f}, "
        f"{champ.holdout.ci_high:.4f}]  gap {v.generalization_gap:+.4f}"
    )
    for e in v.entries:
        if e.kind == "baseline":
            typer.echo(f"baseline   {e.strategy_id:<22} {e.holdout.mean:.4f}  Δ {e.diff_mean:+.4f}")
    typer.echo(f"report     {run_dir / REPORT_MD}")


@app.command()
def report(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="A run directory.")],
) -> None:
    """Regenerate report.md and evolution.png from a run's run.json."""
    typer.echo(render_report(run_dir))


@app.command()
def summary(
    run_dirs: Annotated[
        list[Path], typer.Argument(exists=True, help="Run directories, or folders of runs.")
    ],
) -> None:
    """Print a Markdown table of champions vs. their best baseline on held-out instances."""
    found: list[Path] = []
    for path in run_dirs:
        if (path / RUN_JSON).is_file():
            found.append(path)
        else:
            found += sorted(p.parent for p in path.glob(f"*/{RUN_JSON}"))
    if not found:
        raise typer.BadParameter(f"no {RUN_JSON} found")
    typer.echo(summarize_runs(found), nl=False)


def main() -> None:
    app()
