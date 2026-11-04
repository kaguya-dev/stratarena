"""stratarena: choose strategies by simulation."""

from stratarena.arena import Arena, ArenaResult, RoundRecord, Standing
from stratarena.config import ArenaConfig, RunConfig, load_config
from stratarena.problem import EvalResult, Problem
from stratarena.strategy import ParamSpace, ParamSpec, ParamStrategy

__all__ = [
    "Arena",
    "ArenaConfig",
    "ArenaResult",
    "EvalResult",
    "ParamSpace",
    "ParamSpec",
    "ParamStrategy",
    "Problem",
    "RoundRecord",
    "RunConfig",
    "Standing",
    "load_config",
]

__version__ = "0.1.0"
