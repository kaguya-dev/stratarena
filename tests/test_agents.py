import json
from pathlib import Path

import pytest

from stratarena.agents import (
    AgentContext,
    CodeGeneratorAgent,
    CriticAgent,
    Orchestrator,
    ParamTunerAgent,
    extract_code,
)
from stratarena.arena import Arena
from stratarena.config import AgentsConfig, ArenaConfig, HybridConfig, LLMConfig, RunConfig
from stratarena.examples.binpacking import CODE_EXAMPLE, BinPacking
from stratarena.examples.gridbots import CODE_EXAMPLE as BOT_EXAMPLE
from stratarena.examples.gridbots import GridBots
from stratarena.llm import CachedLLM, ScriptedLLM
from stratarena.report import write_run
from stratarena.sandbox import SandboxLimits
from stratarena.search import random_strategies, run_search
from stratarena.strategy import CodeStrategy

LIMITS = SandboxLimits(call_timeout_s=1.0, cpu_seconds=10)
GOOD = f"Here you go:\n```python\n{CODE_EXAMPLE}```\n"
BROKEN = "```python\ndef pack(items, capacity):\n    return [0] * len(items)\n```"


@pytest.fixture(scope="module")
def ctx() -> AgentContext:
    problem = BinPacking(n_items=30, profiles=["triplets"])
    cfg = ArenaConfig(instances_per_round=2, finalists=2, bootstrap_samples=100)
    result = Arena(problem, cfg, seed=0).run(random_strategies(problem, 0, 0, 4))
    return AgentContext(
        problem=problem,
        generation=0,
        finalists=result.finalists,
        standings=result.standings,
        baselines=tuple(problem.baselines()),
        seed=0,
    )


def test_extract_code() -> None:
    assert extract_code(GOOD, "pack") == CODE_EXAMPLE
    assert extract_code("```\ndef pack(a, b):\n    pass\n```", "pack") is not None
    assert extract_code("def pack(a, b):\n    pass\n", "pack") is not None
    assert extract_code("```python\ndef other():\n    pass\n```", "pack") is None


def test_generator_accepts_valid_code(ctx: AgentContext) -> None:
    llm = ScriptedLLM([GOOD])
    proposal = CodeGeneratorAgent(llm, LIMITS).propose_one(ctx, 0, 1)
    (strategy,) = proposal.strategies
    assert isinstance(strategy, CodeStrategy)
    assert strategy.id == "g1-llm00" and strategy.origin == "generator:scripted"
    assert [e.action for e in proposal.events] == ["accepted"]
    prompt = llm.calls[0][1]
    assert "pack(items: list[int], capacity: int)" in prompt
    assert "How to read the parametric heuristics" in prompt and "open_penalty" in prompt
    assert ctx.finalists[0].id in prompt


def test_generator_retries_with_the_error(ctx: AgentContext) -> None:
    llm = ScriptedLLM([BROKEN, GOOD])
    proposal = CodeGeneratorAgent(llm, LIMITS, max_attempts=2).propose_one(ctx, 1, 2)
    assert len(proposal.strategies) == 1
    assert [e.action for e in proposal.events] == ["rejected", "accepted"]
    assert "exceeds the capacity" in llm.calls[1][1]


def test_generator_gives_up(ctx: AgentContext) -> None:
    llm = ScriptedLLM(["no code here", "```python\ndef pack(:\n```"])
    proposal = CodeGeneratorAgent(llm, LIMITS, max_attempts=2).propose_one(ctx, 0, 1)
    assert proposal.strategies == ()
    assert [e.action for e in proposal.events] == ["rejected", "rejected"]
    out = CodeGeneratorAgent(ScriptedLLM([]), LIMITS).propose_one(ctx, 0, 1)
    assert [e.action for e in out.events] == ["failed"]


def test_critic_note_becomes_feedback(ctx: AgentContext) -> None:
    llm = ScriptedLLM(["Too greedy.\n Try   pairing items."])
    strategy = CodeStrategy("g1-llm00", CODE_EXAMPLE, "pack", limits=LIMITS)
    proposal = CriticAgent(llm).review(ctx, strategy)
    assert proposal.notes == ("On `g1-llm00`: Too greedy. Try pairing items.",)
    assert "g1-llm00" in llm.calls[0][1]


def test_tuner_is_deterministic_and_improves(ctx: AgentContext) -> None:
    tuner = ParamTunerAgent(iterations=6, instances=3)
    a = tuner.propose(ctx, 2)
    assert a == tuner.propose(ctx, 2)
    for s in a.strategies:
        assert s.id.startswith("g1-t") and s.parents == (ctx.finalists[0].id,)
    assert a.events


def test_orchestrator_merges_in_task_order(ctx: AgentContext) -> None:
    def answer(system: str, prompt: str) -> str:
        return "Fine." if "weaknesses" in prompt else GOOD

    cfg = AgentsConfig(enabled=True, code_candidates=3, tuner_candidates=1, tuner_iterations=3)
    proposal = Orchestrator(cfg, ScriptedLLM(answer)).propose(ctx)
    ids = [s.id for s in proposal.strategies]
    assert ids[:3] == ["g1-llm00", "g1-llm01", "g1-llm02"]
    assert sum(e.action == "reviewed" for e in proposal.events) == 3
    assert len(proposal.notes) == 3


def test_orchestrator_without_llm_only_tunes(ctx: AgentContext) -> None:
    cfg = AgentsConfig(enabled=True, tuner_iterations=3)
    orchestrator = Orchestrator(cfg, None)
    assert orchestrator.generator is None and orchestrator.critic is None
    assert all(s.id.startswith("g1-t") for s in orchestrator.propose(ctx).strategies)


def test_game_smoke_test_rejects_bots_that_never_score() -> None:
    game = GridBots(size=7, coin_pairs=3, turns=20)
    ctx = AgentContext(game, 0, tuple(game.baselines()[:2]), (), tuple(game.baselines()), 0)
    idle = "```python\ndef choose_move(state):\n    return 4\n```"
    good = f"```python\n{BOT_EXAMPLE}```"
    gen = CodeGeneratorAgent(ScriptedLLM([idle, good]), LIMITS, max_attempts=2)
    proposal = gen.propose_one(ctx, 0, 1)
    assert [e.action for e in proposal.events] == ["rejected", "accepted"]


def test_search_with_agents_end_to_end(tmp_path: Path) -> None:
    def answer(system: str, prompt: str) -> str:
        return "Fine." if "weaknesses" in prompt else GOOD

    cfg = RunConfig(
        seed=2,
        problem="binpacking",
        problem_options={"n_items": 30},
        population=8,
        generations=2,
        holdout_instances=6,
        tracking_instances=2,
        arena=ArenaConfig(instances_per_round=2, finalists=2, bootstrap_samples=100),
        hybrid=HybridConfig(immigrants=1),
        agents=AgentsConfig(
            enabled=True,
            code_candidates=2,
            tuner_candidates=1,
            tuner_iterations=2,
            llm=LLMConfig(cache_dir=str(tmp_path / "cache")),
        ),
    )
    problem = BinPacking(n_items=30)
    llm = CachedLLM(ScriptedLLM(answer), tmp_path / "cache")
    result = run_search(problem, cfg, llm=llm)
    ids = {s.id for s in result.populations[1]}
    assert {"g1-llm00", "g1-llm01"} <= ids
    assert any(e.agent == "tuner" for e in result.agent_events)
    assert len(result.populations[1]) == 8
    assert result.llm == "scripted"

    replay = run_search(problem, cfg, llm=CachedLLM(ScriptedLLM([]), tmp_path / "cache"))
    assert replay.generations == result.generations
    assert replay.validation == result.validation

    run_dir = write_run(result, tmp_path / "run")
    assert (run_dir / "strategies" / "g1-llm00.py").read_text().endswith(CODE_EXAMPLE)
    data = json.loads((run_dir / "run.json").read_text())
    assert data["agents"]["llm_requests"] == len(result.exchanges) > 0
    assert (run_dir / "agents" / "llm_exchanges.jsonl").is_file()
    assert "## Agents" in (run_dir / "report.md").read_text()
