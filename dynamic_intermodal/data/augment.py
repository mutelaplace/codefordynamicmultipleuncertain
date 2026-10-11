"""Denser order sets for the same network (more interaction on shared services and yards).

Extra orders are resampled from the existing ones: same origin / destination and time-window
length, release time jittered and volume perturbed.
"""

from __future__ import annotations

import numpy as np

from dynamic_intermodal.data.models import ProblemInstance, Request


def augment_requests(
    problem: ProblemInstance,
    n_total: int,
    seed: int = 0,
    release_jitter: float = 12.0,
    volume_range: tuple[float, float] = (10.0, 30.0),
) -> ProblemInstance:
    rng = np.random.default_rng(seed)
    base = [problem.requests[k] for k in sorted(problem.requests)]
    if n_total <= len(base):
        requests = {r.id: r for r in base[:n_total]}
    else:
        requests = {r.id: r for r in base}
        latest = max(r.pickup_time_window[0] for r in base)
        next_id = max(requests) + 1
        while len(requests) < n_total:
            src = base[rng.integers(len(base))]
            shift = float(rng.uniform(-release_jitter, release_jitter))
            release = float(np.clip(round(src.pickup_time_window[0] + shift), 1.0, latest))
            length = src.delivery_time_window[1] - src.pickup_time_window[0]
            volume = float(rng.integers(int(volume_range[0]), int(volume_range[1]) + 1))
            requests[next_id] = Request(
                id=next_id,
                pickup_node=src.pickup_node,
                delivery_node=src.delivery_node,
                pickup_time_window=(release, release + length),
                delivery_time_window=(release, release + length),
                demand=volume,
                delay_cost=src.delay_cost,
            )
            next_id += 1
    return ProblemInstance(
        terminals=problem.terminals,
        vehicles=problem.vehicles,
        requests=requests,
        distances=problem.distances,
        transfer_nodes=problem.transfer_nodes,
        loading_time=problem.loading_time,
        agent_mapping=problem.agent_mapping,
    )
