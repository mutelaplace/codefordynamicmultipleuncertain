from dynamic_intermodal.data.augment import augment_requests
from dynamic_intermodal.data.loader import build_problem_instance
from dynamic_intermodal.data.models import ProblemInstance, Request, Terminal, Vehicle


def load_problem(config) -> ProblemInstance:
    """Load the instance of a top-level Config, optionally densified with resampled orders."""
    problem = build_problem_instance(config.data, config.cost)
    if config.data.augment_to:
        problem = augment_requests(problem, config.data.augment_to, config.data.augment_seed)
    return problem


__all__ = [
    "ProblemInstance", "Request", "Terminal", "Vehicle",
    "augment_requests", "build_problem_instance", "load_problem",
]
