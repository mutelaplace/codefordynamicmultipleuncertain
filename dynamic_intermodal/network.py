"""Service-network view of a problem instance.

Every vehicle of the original data model is one *service* on a single origin-destination
leg: barge and rail services run on a fixed timetable (cut-off, departure, arrival), trucks
are flexible and leave as soon as the cargo is ready. Orders are chains of services joined
at transfer terminals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable

from dynamic_intermodal.config import CostConfig, NetworkConfig, StakeholderConfig
from dynamic_intermodal.data.models import ProblemInstance, Terminal

MODE_BY_TYPE = {1: "barge", 2: "rail", 3: "truck"}


@dataclass(frozen=True)
class Service:
    id: int
    name: str
    mode: str
    origin: int
    destination: int
    carrier_id: str
    capacity: float
    scheduled: bool
    cutoff: float  # scheduled: loading starts, cargo must be on the terminal; truck: nan
    departure: float  # scheduled departure; truck: nan
    arrival: float  # scheduled arrival at destination; truck: nan
    load_time: float
    unload_time: float
    distance: float
    travel_time: float  # nominal running time
    cost_per_teu: float  # transport + loading/unloading + carbon per TEU

    @property
    def leg(self) -> tuple[int, int]:
        return (self.origin, self.destination)


@dataclass(frozen=True)
class Order:
    id: int
    origin: int
    destination: int
    release: float
    due: float
    volume: float
    delay_cost: float
    shipper_id: str


@dataclass(frozen=True)
class NominalLeg:
    service_id: int
    load_start: float
    departure: float
    arrival: float
    ready: float
    wait: float


class ServiceNetwork:
    def __init__(
        self,
        problem: ProblemInstance,
        cost_config: CostConfig | None = None,
        network_config: NetworkConfig | None = None,
        stakeholder_config: StakeholderConfig | None = None,
    ):
        self.problem = problem
        self.cost_config = cost_config or CostConfig()
        self.config = network_config or NetworkConfig()
        shippers = (stakeholder_config or StakeholderConfig()).shippers

        self.terminals: dict[int, Terminal] = dict(problem.terminals)
        self.transfer_nodes: set[int] = set(problem.transfer_nodes)
        self.services: dict[int, Service] = {
            vid: self._build_service(vid) for vid in sorted(problem.vehicles)
        }
        self.orders: dict[int, Order] = {}
        for index, rid in enumerate(sorted(problem.requests)):
            req = problem.requests[rid]
            self.orders[rid] = Order(
                id=rid,
                origin=req.pickup_node,
                destination=req.delivery_node,
                release=float(req.pickup_time_window[0]),
                due=float(req.delivery_time_window[1]),
                volume=float(req.demand),
                delay_cost=float(req.delay_cost),
                shipper_id=shippers[index % len(shippers)],
            )

        self._from_node: dict[int, list[Service]] = {}
        self._scheduled_on_leg: dict[tuple[int, int], list[Service]] = {}
        self._truck_on_leg: dict[tuple[int, int], Service] = {}
        for service in self.services.values():
            self._from_node.setdefault(service.origin, []).append(service)
            if service.scheduled:
                self._scheduled_on_leg.setdefault(service.leg, []).append(service)
            else:
                self._truck_on_leg.setdefault(service.leg, service)
        for services in self._scheduled_on_leg.values():
            services.sort(key=lambda s: (s.cutoff, s.arrival, s.id))
        for services in self._from_node.values():
            services.sort(key=lambda s: (s.cutoff if s.scheduled else -1.0, s.id))

        name_to_id = {t.name: t.id for t in self.terminals.values()}
        self.yard_capacity: dict[int, float] = {
            tid: float(self.config.yard_capacity.get(t.name, self.config.default_yard_capacity))
            for tid, t in self.terminals.items()
        }
        self.name_to_id = name_to_id

    # ------------------------------------------------------------------ build

    def _build_service(self, vid: int) -> Service:
        vehicle = self.problem.vehicles[vid]
        mode = MODE_BY_TYPE.get(vehicle.type, "unknown")
        o, d = vehicle.origin_depot, vehicle.destination_depot
        distance = self.problem.get_distance(o, d, vehicle.type)
        cost_travel_time = distance / vehicle.speed if vehicle.speed > 0 else 0.0
        load_cost = self.cost_config.load_cost.get(vehicle.type, 0.0)
        cost_per_teu = (
            vehicle.km_cost * distance
            + vehicle.h_cost * cost_travel_time
            + load_cost * 2.0
            + vehicle.carbon_cost * distance / self.cost_config.carbon_emission
        )
        carrier_id = f"{mode}-{vehicle.agent_id or 0}"
        timetable = vehicle.timetable if vehicle.type != 3 else None
        if timetable and o in timetable and d in timetable:
            tw_o, tw_d = timetable[o], timetable[d]
            return Service(
                id=vid,
                name=vehicle.name,
                mode=mode,
                origin=o,
                destination=d,
                carrier_id=carrier_id,
                capacity=float(vehicle.capacity),
                scheduled=True,
                cutoff=float(tw_o[0]),
                departure=float(tw_o[1]),
                arrival=float(tw_d[0]),
                load_time=float(tw_o[1] - tw_o[0]),
                unload_time=float(tw_d[1] - tw_d[0]),
                distance=distance,
                travel_time=float(tw_d[0] - tw_o[1]),
                cost_per_teu=cost_per_teu,
            )
        handling = self.config.truck_handling_time
        return Service(
            id=vid,
            name=vehicle.name,
            mode=mode,
            origin=o,
            destination=d,
            carrier_id=carrier_id,
            capacity=float(vehicle.capacity),
            scheduled=False,
            cutoff=math.nan,
            departure=math.nan,
            arrival=math.nan,
            load_time=handling,
            unload_time=handling,
            distance=distance,
            travel_time=cost_travel_time,
            cost_per_teu=cost_per_teu,
        )

    # ---------------------------------------------------------------- queries

    def services_from(self, node: int) -> list[Service]:
        return self._from_node.get(node, [])

    def scheduled_on_leg(self, origin: int, destination: int) -> list[Service]:
        return self._scheduled_on_leg.get((origin, destination), [])

    def truck_on_leg(self, origin: int, destination: int) -> Service | None:
        return self._truck_on_leg.get((origin, destination))

    @property
    def scheduled_services(self) -> list[Service]:
        return [s for s in self.services.values() if s.scheduled]

    @property
    def truck_legs(self) -> list[tuple[int, int]]:
        return sorted(self._truck_on_leg)

    def node_name(self, node: int) -> str:
        return self.terminals[node].name

    def slot_of(self, t: float) -> int:
        return int(math.floor(t / self.config.slot_hours))

    def slot_bounds(self, slot: int) -> tuple[float, float]:
        h = self.config.slot_hours
        return slot * h, (slot + 1) * h

    def block_of(self, t: float) -> int:
        n_blocks = self.n_blocks
        return min(max(int(math.floor(t / self.config.block_hours)), 0), n_blocks - 1)

    @property
    def n_blocks(self) -> int:
        return int(math.ceil(self.config.horizon / self.config.block_hours))

    @property
    def handling_nodes(self) -> list[int]:
        """Terminals where scheduled services unload (subject to handling uncertainty)."""
        return sorted({s.destination for s in self.scheduled_services})

    # ------------------------------------------------------- nominal timing

    def nominal_leg(self, service: Service, ready: float) -> NominalLeg | None:
        """Nominal (disturbance-free) timing of one leg; None if the cut-off is missed."""
        if service.scheduled:
            if ready > service.cutoff + 1e-9:
                return None
            return NominalLeg(
                service_id=service.id,
                load_start=service.cutoff,
                departure=service.departure,
                arrival=service.arrival,
                ready=service.arrival + service.unload_time,
                wait=service.cutoff - ready,
            )
        departure = ready + service.load_time
        arrival = departure + service.travel_time
        return NominalLeg(
            service_id=service.id,
            load_start=ready,
            departure=departure,
            arrival=arrival,
            ready=arrival + service.unload_time,
            wait=0.0,
        )

    def nominal_chain(
        self, order: Order, chain: Iterable[int], start_ready: float
    ) -> tuple[list[NominalLeg], float, float] | None:
        """Return (legs, arrival at destination, generalized cost) or None if infeasible."""
        legs: list[NominalLeg] = []
        ready = start_ready
        cost = 0.0
        arrival = start_ready
        for sid in chain:
            service = self.services[sid]
            leg = self.nominal_leg(service, ready)
            if leg is None:
                return None
            legs.append(leg)
            cost += order.volume * (service.cost_per_teu + self.cost_config.storage_cost * leg.wait)
            ready = leg.ready
            arrival = leg.arrival
        cost += order.delay_cost * order.volume * max(0.0, arrival - order.due)
        return legs, arrival, cost

    def enumerate_chains(
        self,
        order: Order,
        start_node: int,
        start_ready: float,
        max_legs: int,
        allowed: Callable[[Service], bool] | None = None,
    ) -> list[tuple[int, ...]]:
        """Time-feasible service chains from ``start_node`` to the order destination."""
        allowed = allowed or (lambda service: True)
        latest_arrival = order.due + self.config.max_lateness_prune
        chains: list[tuple[int, ...]] = []

        def dfs(node: int, ready: float, chain: tuple[int, ...], visited: frozenset[int]) -> None:
            if len(chain) >= max_legs:
                return
            for service in self.services_from(node):
                if service.destination in visited or not allowed(service):
                    continue
                if service.capacity < order.volume:
                    continue
                leg = self.nominal_leg(service, ready)
                if leg is None or leg.arrival > latest_arrival:
                    continue
                new_chain = chain + (service.id,)
                if service.destination == order.destination:
                    chains.append(new_chain)
                elif service.destination in self.transfer_nodes:
                    dfs(service.destination, leg.ready, new_chain, visited | {service.destination})

        if start_node == order.destination:
            return [()]
        dfs(start_node, start_ready, (), frozenset({start_node}))
        return chains


def build_network(problem: ProblemInstance, config) -> ServiceNetwork:
    """Convenience constructor from a top-level :class:`Config`."""
    return ServiceNetwork(problem, config.cost, config.network, config.stakeholders)
