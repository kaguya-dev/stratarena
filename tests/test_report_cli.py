import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from stratarena.cli import app
from stratarena.config import RunConfig
from stratarena.report import render_report, run_name, write_run
from stratarena.search import run_search

TINY = {
    "seed": 3,
    "problem": "binpacking",
    "problem_options": {"n_items": 25},
    "population": 6,
    "generations": 2,
    "holdout_instances": 8,
    "arena": {"instances_per_round": 2, "finalists": 2, "bootstrap_samples": 100},
    "hybrid": {"immigrants": 1},
}

runner = CliRunner()


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(TINY))
    return path


def run_dirs(out: Path) -> list[Path]:
    return [p for p in out.iterdir() if p.is_dir()]


def test_write_run_creates_artifacts(tmp_path: Path) -> None:
    from stratarena.examples import make_problem

    cfg = RunConfig.model_validate(TINY)
    result = run_search(make_problem(cfg.problem, cfg.problem_options), cfg)
    run_dir = write_run(result, tmp_path / run_name(result))
    for name in ["config.yaml", "run.json", "report.md", "evolution.png"]:
        assert (run_dir / name).is_file()

    data = json.loads((run_dir / "run.json").read_text())
    assert data["champion"]["id"] == result.champion.id
    assert len(data["generations"]) == 2
    assert RunConfig.model_validate(yaml.safe_load((run_dir / "config.yaml").read_text())) == cfg

    md = (run_dir / "report.md").read_text()
    assert result.champion.id in md
    assert "Held-out validation" in md
    assert "Problem: `binpacking (n_items=25)`" in md


def test_report_is_regenerated_from_json(tmp_path: Path, config_file: Path) -> None:
    assert (
        runner.invoke(
            app, ["run", "--config", str(config_file), "--out", str(tmp_path), "--quiet"]
        ).exit_code
        == 0
    )
    (run_dir,) = run_dirs(tmp_path)
    original = (run_dir / "report.md").read_text()
    (run_dir / "report.md").unlink()
    (run_dir / "evolution.png").unlink()
    res = runner.invoke(app, ["report", str(run_dir)])
    assert res.exit_code == 0, res.output
    assert (run_dir / "report.md").read_text() == original
    assert (run_dir / "evolution.png").is_file()


def test_cli_runs_are_reproducible(tmp_path: Path, config_file: Path) -> None:
    outputs = []
    for name in ["a", "b"]:
        out = tmp_path / name
        res = runner.invoke(
            app, ["run", "--config", str(config_file), "--out", str(out), "--quiet"]
        )
        assert res.exit_code == 0, res.output
        (run_dir,) = run_dirs(out)
        outputs.append((run_dir.name, (run_dir / "run.json").read_bytes()))
    assert outputs[0] == outputs[1]


def test_cli_overrides(tmp_path: Path, config_file: Path) -> None:
    res = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(config_file),
            "--seed",
            "9",
            "--workers",
            "2",
            "--out",
            str(tmp_path),
            "--quiet",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "champion" in res.output
    (run_dir,) = run_dirs(tmp_path)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert cfg["seed"] == 9
    assert cfg["arena"]["workers"] == 2
    assert run_dir.name.startswith("binpacking-seed9-")


def test_cli_needs_problem_or_config() -> None:
    res = runner.invoke(app, ["run"])
    assert res.exit_code != 0


def test_cli_rejects_unknown_problem(tmp_path: Path) -> None:
    res = runner.invoke(app, ["run", "--problem", "nope", "--out", str(tmp_path)])
    assert res.exit_code != 0


def test_render_report_needs_run_json(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        render_report(tmp_path)


def test_problem_label() -> None:
    from stratarena.report import problem_label

    assert problem_label({"problem": "x", "problem_options": {}}) == "x"
    label = problem_label({"problem": "x", "problem_options": {"profiles": ["a", "b"], "n": 3}})
    assert label == "x (n=3, profiles=a,b)"
