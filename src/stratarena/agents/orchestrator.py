"""Orchestrator: runs the specialized agents in parallel and merges their proposals."""

from concurrent.futures import Future, ThreadPoolExecutor

from stratarena.agents.base import AgentContext, Proposal
from stratarena.agents.critic import CriticAgent
from stratarena.agents.generator import CodeGeneratorAgent
from stratarena.agents.tuner import ParamTunerAgent
from stratarena.config import AgentsConfig
from stratarena.llm import LLM
from stratarena.log import get_logger
from stratarena.sandbox import SandboxLimits
from stratarena.strategy import CodeStrategy

log = get_logger(__name__)


class Orchestrator:
    """Splits the work of proposing the next generation's candidates between agents.

    Phase 1 runs in parallel threads: each code candidate is its own generator task (LLM
    calls are I/O-bound), next to the parameter tuner. Phase 2 has the critic review every
    accepted code candidate, also in parallel. Results are merged in task order, never in
    completion order, so the output does not depend on timing.
    """

    def __init__(
        self,
        config: AgentsConfig,
        llm: LLM | None,
    ) -> None:
        self.config = config
        limits = SandboxLimits(**config.sandbox.model_dump())
        self.generator = (
            CodeGeneratorAgent(llm, limits, config.max_attempts, config.smoke_instances)
            if llm is not None and config.code_candidates > 0
            else None
        )
        self.critic = (
            CriticAgent(llm, config.smoke_instances)
            if llm is not None and config.critic and self.generator is not None
            else None
        )
        self.tuner = (
            ParamTunerAgent(config.tuner_iterations, config.tuner_instances)
            if config.tuner_candidates > 0
            else None
        )

    def propose(self, ctx: AgentContext) -> Proposal:
        cfg = self.config
        with ThreadPoolExecutor(max_workers=cfg.max_parallel) as pool:
            tasks: list[Future[Proposal]] = []
            if self.generator is not None:
                total = cfg.code_candidates
                tasks += [
                    pool.submit(self.generator.propose_one, ctx, i, total) for i in range(total)
                ]
            if self.tuner is not None:
                tasks.append(pool.submit(self.tuner.propose, ctx, cfg.tuner_candidates))
            proposed = Proposal.merge([t.result() for t in tasks])

            reviews: list[Future[Proposal]] = []
            if self.critic is not None:
                reviews = [
                    pool.submit(self.critic.review, ctx, s)
                    for s in proposed.strategies
                    if isinstance(s, CodeStrategy)
                ]
            reviewed = Proposal.merge([r.result() for r in reviews])

        result = Proposal.merge([proposed, reviewed])
        log.info(
            "agents.proposal",
            generation=ctx.generation,
            candidates=[s.id for s in result.strategies],
            events=len(result.events),
        )
        return result
