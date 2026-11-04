"""Code generator: asks an LLM for new strategies and admits only those that pass checks."""

import re

from stratarena.agents.base import (
    AgentContext,
    AgentEvent,
    Proposal,
    describe_strategy,
    smoke_test,
)
from stratarena.llm import LLM, LLMError
from stratarena.problem import CodeSpec, HasCodeSpec
from stratarena.sandbox import SandboxLimits, check_source
from stratarena.strategy import CodeStrategy

SYSTEM = (
    "You write Python strategies for an optimization framework. Reply with exactly one "
    "```python code block containing a complete module that defines the requested "
    "function. Use the standard library only. No explanations outside the code block."
)

# Each parallel candidate gets a different direction, so the batch is diverse.
DIRECTIONS = (
    "Improve on the reference implementation with a better rule or ordering.",
    "Use look-ahead or a short local search on top of a greedy construction.",
    "Try a clearly different approach from the reference and from the strategies above.",
    "Combine the strongest ideas of the strategies above into one.",
)

_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str, entrypoint: str) -> str | None:
    """The first code block that defines ``entrypoint`` (or the whole text if it does)."""
    pattern = re.compile(rf"^\s*def\s+{re.escape(entrypoint)}\s*\(", re.MULTILINE)
    for block in _CODE_BLOCK.findall(text):
        if pattern.search(block):
            return str(block).strip() + "\n"
    if pattern.search(text) and "```" not in text:
        return text.strip() + "\n"
    return None


class CodeGeneratorAgent:
    name = "generator"

    def __init__(
        self, llm: LLM, limits: SandboxLimits, max_attempts: int = 2, smoke_instances: int = 2
    ) -> None:
        self.llm = llm
        self.limits = limits
        self.max_attempts = max_attempts
        self.smoke_instances = smoke_instances

    def prompt(self, ctx: AgentContext, spec: CodeSpec, index: int, total: int) -> str:
        means = {s.strategy_id: s.mean for s in ctx.standings}
        best = "\n\n".join(
            f"- `{f.id}` (mean score {means.get(f.id, float('nan')):.4f}): {describe_strategy(f)}"
            for f in ctx.finalists
        )
        notes = "\n".join(f"- {n}" for n in ctx.feedback[-6:]) or "- (none yet)"
        guide = (
            f"## How to read the parametric heuristics below\n\n{spec.param_guide}\n"
            if spec.param_guide
            else ""
        )
        return (
            f"# Problem: {ctx.problem.name}\n\n{spec.description}\n"
            f"## Reference implementation (valid)\n\n```python\n{spec.example}```\n\n"
            f"{guide}"
            f"## Best strategies after generation {ctx.generation}\n\n{best}\n\n"
            f"## Reviewer notes on earlier candidates\n\n{notes}\n\n"
            f"## Your task\n\nCandidate {index + 1} of {total}. "
            f"{DIRECTIONS[index % len(DIRECTIONS)]} Write a new `{spec.entrypoint}` that "
            "should score higher than every strategy above."
        )

    def propose_one(self, ctx: AgentContext, index: int, total: int) -> Proposal:
        if not isinstance(ctx.problem, HasCodeSpec):
            return Proposal()
        spec = ctx.problem.code_spec()
        sid = f"g{ctx.generation + 1}-llm{index:02d}"
        base_prompt = self.prompt(ctx, spec, index, total)
        events = []
        error = ""
        for attempt in range(1, self.max_attempts + 1):
            prompt = base_prompt
            if error:
                prompt += f"\n\n## Your previous attempt was rejected\n\n{error}\nFix it."
            try:
                reply = self.llm.complete(SYSTEM, prompt)
            except LLMError as e:
                events.append(AgentEvent(self.name, ctx.generation, "failed", sid, str(e)))
                return Proposal(events=tuple(events))
            source = extract_code(reply, spec.entrypoint)
            if source is None:
                error = f"No ```python block defining `{spec.entrypoint}` was found."
            elif (problem := check_source(source, spec.entrypoint, self.limits)) is not None:
                error = f"Loading the code failed: {problem}"
            else:
                candidate = CodeStrategy(
                    id=sid,
                    source=source,
                    entrypoint=spec.entrypoint,
                    parents=tuple(f.id for f in ctx.finalists[:2]),
                    origin=f"{self.name}:{self.llm.name}",
                    limits=self.limits,
                )
                smoke = smoke_test(ctx, candidate, self.smoke_instances)
                if smoke.error is None:
                    detail = f"attempt {attempt}, smoke mean {smoke.mean:.4f}"
                    events.append(AgentEvent(self.name, ctx.generation, "accepted", sid, detail))
                    return Proposal((candidate,), tuple(events))
                error = f"The code ran but failed the smoke test: {smoke.error}"
            events.append(
                AgentEvent(
                    self.name, ctx.generation, "rejected", sid, f"attempt {attempt}: {error}"
                )
            )
        return Proposal(events=tuple(events))
