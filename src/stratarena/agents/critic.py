"""Critic: reviews accepted code candidates; its notes steer the next generation."""

from stratarena.agents.base import AgentContext, AgentEvent, Proposal, smoke_test
from stratarena.llm import LLM, LLMError
from stratarena.problem import HasCodeSpec
from stratarena.strategy import CodeStrategy

SYSTEM = (
    "You review strategy code for an optimization framework. Be concrete and brief: "
    "plain text, no code blocks, at most 120 words."
)


class CriticAgent:
    name = "critic"

    def __init__(self, llm: LLM, smoke_instances: int = 2) -> None:
        self.llm = llm
        self.smoke_instances = smoke_instances

    def review(self, ctx: AgentContext, strategy: CodeStrategy) -> Proposal:
        if not isinstance(ctx.problem, HasCodeSpec):
            return Proposal()
        spec = ctx.problem.code_spec()
        smoke = smoke_test(ctx, strategy, self.smoke_instances)
        best = ctx.standings[0] if ctx.standings else None
        reference = (
            f"The current best strategy `{best.strategy_id}` averages {best.mean:.4f}."
            if best
            else ""
        )
        prompt = (
            f"# Problem: {ctx.problem.name}\n\n{spec.description}\n"
            f"## Candidate `{strategy.id}`\n\n```python\n{strategy.source}```\n\n"
            f"On {len(smoke.scores)} test instances it scored "
            f"{', '.join(f'{x:.4f}' for x in smoke.scores)} (higher is better). {reference}\n\n"
            "List up to three weaknesses and one concrete improvement."
        )
        try:
            note = " ".join(self.llm.complete(SYSTEM, prompt).split())
        except LLMError as e:
            return Proposal(
                events=(AgentEvent(self.name, ctx.generation, "failed", strategy.id, str(e)),)
            )
        event = AgentEvent(self.name, ctx.generation, "reviewed", strategy.id, note)
        return Proposal(events=(event,), notes=(f"On `{strategy.id}`: {note}",))
