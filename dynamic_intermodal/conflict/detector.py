"""Shared-resource conflict check: node-slot-order (yard) and service-order (capacity) records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from dynamic_intermodal.network import ServiceNetwork
from dynamic_intermodal.risk.propagation import OrderOutcome, ScenarioOutcome


@dataclass(frozen=True)
class ConflictRecord:
    resource: str  # "yard" | "service"
    node: int
    slot: int
    slot_start: float
    slot_end: float
    service_id: int | None
    order_ids: frozenset[int]
    probability: float
    expected_overflow: float
    max_load: float
    capacity: float
    scenario_ids: tuple[int, ...]

    @property
    def key(self) -> tuple:
        return (self.resource, self.node, self.slot, self.service_id)


def order_slot_occupancy(
    outcome: OrderOutcome, network: ServiceNetwork, t_from: float = -np.inf
) -> dict[tuple[int, int], float]:
    """(node, slot) -> TEU of this order on the yard during the slot (slots ending after t_from)."""
    result: dict[tuple[int, int], float] = {}
    for node, start, end in outcome.occupancy:
        if end <= start:
            continue
        first, last = network.slot_of(start), network.slot_of(end - 1e-9)
        for slot in range(first, last + 1):
            if network.slot_bounds(slot)[1] <= t_from:
                continue
            result[(node, slot)] = outcome.volume
    return result


def yard_profile(
    outcome: ScenarioOutcome,
    network: ServiceNetwork,
    t_from: float = -np.inf,
    order_ids: Iterable[int] | None = None,
) -> dict[tuple[int, int], tuple[float, set[int]]]:
    profile: dict[tuple[int, int], tuple[float, set[int]]] = {}
    ids = order_ids if order_ids is not None else outcome.orders
    for oid in ids:
        for key, volume in order_slot_occupancy(outcome.orders[oid], network, t_from).items():
            load, members = profile.get(key, (0.0, set()))
            members.add(oid)
            profile[key] = (load + volume, members)
    return profile


class ConflictDetector:
    def __init__(self, network: ServiceNetwork, min_prob: float = 0.05):
        self.network = network
        self.min_prob = min_prob

    def detect(
        self, outcomes: list[ScenarioOutcome], weights: np.ndarray, t_now: float = 0.0
    ) -> list[ConflictRecord]:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
        acc: dict[tuple, dict] = {}

        for s, outcome in enumerate(outcomes):
            for (node, slot), (load, members) in yard_profile(outcome, self.network, t_now).items():
                capacity = self.network.yard_capacity[node]
                if load <= capacity + 1e-9:
                    continue
                entry = acc.setdefault(("yard", node, slot, None), self._empty(capacity))
                self._add(entry, s, weights[s], load, capacity, members)
            for sid, bumped in outcome.service_bumped.items():
                service = self.network.services[sid]
                if service.cutoff <= t_now:
                    continue
                slot = self.network.slot_of(service.cutoff)
                entry = acc.setdefault(
                    ("service", service.origin, slot, sid), self._empty(service.capacity)
                )
                load = outcome.service_load.get(sid, 0.0) + bumped
                self._add(entry, s, weights[s], load, load - bumped, outcome.service_orders.get(sid, set()))

        records = []
        for (resource, node, slot, sid), entry in acc.items():
            if entry["prob"] < self.min_prob:
                continue
            start, end = self.network.slot_bounds(slot)
            records.append(
                ConflictRecord(
                    resource=resource,
                    node=node,
                    slot=slot,
                    slot_start=start,
                    slot_end=end,
                    service_id=sid,
                    order_ids=frozenset(entry["orders"]),
                    probability=entry["prob"],
                    expected_overflow=entry["overflow"],
                    max_load=entry["max_load"],
                    capacity=entry["capacity"],
                    scenario_ids=tuple(entry["scenarios"]),
                )
            )
        records.sort(key=lambda r: (-r.probability, r.slot_start, r.node))
        return records

    @staticmethod
    def _empty(capacity: float) -> dict:
        return {"prob": 0.0, "overflow": 0.0, "max_load": 0.0, "capacity": capacity,
                "orders": set(), "scenarios": []}

    @staticmethod
    def _add(entry: dict, s: int, w: float, load: float, capacity: float, members: set[int]) -> None:
        entry["prob"] += w
        entry["overflow"] += w * max(0.0, load - capacity)
        entry["max_load"] = max(entry["max_load"], load)
        entry["capacity"] = min(entry["capacity"], capacity)
        entry["orders"] |= set(members)
        entry["scenarios"].append(s)
