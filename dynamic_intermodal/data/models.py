"""Core input data structures.

Copied from ``intermodal_refactor`` (intermodal/data/models.py). Only the input-side
models are kept; the ALNS route representation (TaskPoint / Route / Solution) is not
needed because plans here are chains of services (see ``dynamic_intermodal.plan``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Terminal:
    id: int
    name: str
    is_transfer: bool


@dataclass
class Vehicle:
    id: int
    name: str
    type: int
    capacity: int
    speed: float
    origin_depot: int
    destination_depot: int

    km_cost: float
    h_cost: float
    carbon_cost: float
    load_cost: float
    storage_cost: float
    waiting_cost: float
    carbon_emission: float

    timetable: dict[int, tuple[float, float]] | None = None
    agent_id: int | None = None
    fixed_transportation: bool = True

    @property
    def type_name(self) -> str:
        return {1: "Barge", 2: "Train", 3: "Truck"}.get(self.type, "Unknown")


@dataclass(frozen=True)
class Request:
    id: int
    pickup_node: int
    delivery_node: int
    pickup_time_window: tuple[float, float]
    delivery_time_window: tuple[float, float]
    demand: float
    delay_cost: float
    allow_transshipment: bool = True


class ProblemInstance:
    def __init__(
        self,
        terminals: dict[int, Terminal] | None = None,
        vehicles: dict[int, Vehicle] | None = None,
        requests: dict[int, Request] | None = None,
        distances: dict[int, np.ndarray] | None = None,
        transfer_nodes: set[int] | None = None,
        loading_time: float = 0.0,
        agent_mapping: dict[tuple[int, int], int] | None = None,
    ):
        self.terminals = terminals or {}
        self.vehicles = vehicles or {}
        self.requests = requests or {}
        self.distances = distances or {}
        self.transfer_nodes = transfer_nodes or set()
        self.loading_time = loading_time
        self.agent_mapping = agent_mapping or self._derive_agent_mapping()

    def _derive_agent_mapping(self) -> dict[tuple[int, int], int]:
        mapping: dict[tuple[int, int], int] = {}
        for vehicle in self.vehicles.values():
            if vehicle.agent_id is not None:
                mapping[(vehicle.origin_depot, vehicle.destination_depot)] = vehicle.agent_id
        return mapping

    def get_distance(self, from_node: int, to_node: int, vehicle_type: int) -> float:
        return float(self.distances[vehicle_type][from_node, to_node])

    def is_reachable(self, from_node: int, to_node: int, vehicle_type: int) -> bool:
        return self.get_distance(from_node, to_node, vehicle_type) < 1e6
