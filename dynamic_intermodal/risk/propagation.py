"""Scenario-wise risk propagation.

For one joint scenario the engine runs an event-driven simulation of all orders of a plan:
cargo release -> waiting on the terminal yard -> boarding at the service cut-off (capacity
allocated to confirmed bookings first, in booking order, then to walk-ins) -> delayed
arrival -> terminal handling (slowed down when the yard is over capacity) -> next leg. A late or bumped order misses its planned service
(missed connection) and is rebooked on the actually available follow-up service of the same
leg (or trucked); the resulting waiting time, yard occupancy interval and downstream delay
are recorded, together with the cause chain (upstream delay, handling time, capacity).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Callable, Iterable

from dynamic_intermodal.config import CostConfig, NetworkConfig
from dynamic_intermodal.network import Service, ServiceNetwork
from dynamic_intermodal.plan import TransportPlan
from dynamic_intermodal.uncertainty.scenarios import ScenarioView

EPS = 1e-9


@dataclass
class CostBreakdown:
    transport: float = 0.0  # running + loading/unloading + carbon
    storage: float = 0.0
    delay: float = 0.0

    @property
    def total(self) -> float:
        return self.transport + self.storage + self.delay

    @property
    def operating(self) -> float:
        return self.transport + self.storage


@dataclass(frozen=True)
class LegOutcome:
    planned_service: int
    service: int
    origin: int
    destination: int
    ready_at_origin: float
    load_start: float
    departure: float
    arrival: float
    ready: float
    wait: float


@dataclass(frozen=True)
class PropagationRecord:
    order_id: int
    node: int
    planned_service: int
    replacement_service: int | None
    reason: str  # late_arrival | capacity | cancelled
    time: float
    ready: float
    cutoff: float
    lateness: float
    wait_added: float
    upstream_service: int | None
    upstream_delay: float
    handling_extra: float


@dataclass
class OrderOutcome:
    order_id: int
    volume: float
    release: float
    legs: list[LegOutcome] = field(default_factory=list)
    occupancy: list[tuple[int, float, float]] = field(default_factory=list)  # (node, start, end)
    final_arrival: float = math.nan
    delay: float = 0.0
    cost: CostBreakdown = field(default_factory=CostBreakdown)
    missed: int = 0
    stranded: bool = False

    @property
    def late(self) -> bool:
        return self.delay > 1e-6


@dataclass
class ScenarioOutcome:
    orders: dict[int, OrderOutcome]
    service_load: dict[int, float]
    service_bumped: dict[int, float]
    service_orders: dict[int, set[int]]
    records: list[PropagationRecord]

    @property
    def total_cost(self) -> float:
        return sum(o.cost.total for o in self.orders.values())

    @property
    def delay_cost(self) -> float:
        return sum(o.cost.delay for o in self.orders.values())

    @property
    def delay_hours(self) -> float:
        return sum(o.delay for o in self.orders.values())

    @property
    def n_missed(self) -> int:
        return sum(o.missed for o in self.orders.values())


class _State:
    __slots__ = (
        "oid", "order", "chain", "booked", "idx", "node", "ready", "out",
        "planned", "prev_service", "prev_delay", "prev_handling", "last_arrival", "transit",
    )

    def __init__(self, oid, order, chain, booked, node, ready, out):
        self.oid = oid
        self.order = order
        self.chain = chain
        self.booked = booked
        self.idx = 0
        self.node = node
        self.ready = ready
        self.out = out
        self.planned = list(chain)
        self.prev_service = None
        self.prev_delay = 0.0
        self.prev_handling = 0.0
        self.last_arrival = ready
        self.transit = None  # (service, load_start, departure, arrival, wait) while moving


class PropagationEngine:
    def __init__(
        self,
        network: ServiceNetwork,
        cost_config: CostConfig | None = None,
        network_config: NetworkConfig | None = None,
        allow_truck: Callable[[int], bool] | None = None,
    ):
        self.network = network
        self.cost = cost_config or network.cost_config
        self.config = network_config or network.config
        self.allow_truck = allow_truck or (lambda order_id: True)

    def simulate(
        self,
        plan: TransportPlan,
        scenario: ScenarioView,
        order_ids: Iterable[int] | None = None,
        yard_background: dict[tuple[int, int], float] | None = None,
    ) -> ScenarioOutcome:
        """Simulate the plan in one scenario. ``yard_background`` adds (node, slot) yard loads of
        orders that are not simulated (used when candidates of one order are evaluated alone)."""
        run = _Run(self, scenario, yard_background)
        for oid in (order_ids if order_ids is not None else plan):
            run.add_order(oid, plan[oid])
        return run.execute()


class _Run:
    def __init__(self, engine: PropagationEngine, scenario: ScenarioView, yard_background=None):
        self.e = engine
        self.yard_bg = yard_background or {}
        self.net = engine.network
        self.scen = scenario
        self.heap: list = []
        self.seq = 0
        self.states: dict[int, _State] = {}
        self.queues: dict[int, list] = {}
        self.cutoff_pushed: set[int] = set()
        self.processed: set[int] = set()
        self.service_load: dict[int, float] = {}
        self.service_bumped: dict[int, float] = {}
        self.service_orders: dict[int, set[int]] = {}
        self.records: list[PropagationRecord] = []
        self.yard: dict[int, float] = {}

    # ---------------------------------------------------------------- setup

    def _push(self, time: float, prio: int, kind: str, ref: int) -> None:
        self.seq += 1
        heapq.heappush(self.heap, (time, prio, self.seq, kind, ref))

    def add_order(self, oid: int, order_plan) -> None:
        order = self.net.orders[oid]
        volume = order.volume * self.scen.demand_factor(oid)
        release = order.release + self.scen.release_delay(oid)
        out = OrderOutcome(order_id=oid, volume=volume, release=release)
        chain = [leg.service_id for leg in order_plan.legs]
        booked = [leg.booked_at for leg in order_plan.legs]
        state = _State(oid, order, chain, booked, order.origin, release, out)
        self.states[oid] = state
        self._push(release, 0, "ready", oid)

    # ----------------------------------------------------------------- loop

    def execute(self) -> ScenarioOutcome:
        while self.heap:
            time, _, _, kind, ref = heapq.heappop(self.heap)
            if kind == "ready":
                self._on_ready(self.states[ref], time)
            elif kind == "arrive":
                self._on_arrive(self.states[ref], time)
            else:
                self._on_cutoff(ref, time)
        return ScenarioOutcome(
            orders={oid: st.out for oid, st in self.states.items()},
            service_load=self.service_load,
            service_bumped=self.service_bumped,
            service_orders=self.service_orders,
            records=self.records,
        )

    def _on_ready(self, st: _State, t: float) -> None:
        if st.node == st.order.destination:
            self._finish(st, st.last_arrival, stranded=False)
            return
        self.yard[st.node] = self.yard.get(st.node, 0.0) + st.out.volume
        if st.idx >= len(st.chain):
            self._finish(st, t + self.e.config.stranded_delay, stranded=True)
            return
        service = self.net.services[st.chain[st.idx]]
        if service.origin != st.node:
            raise ValueError(
                f"Order {st.oid}: leg {st.idx} service {service.id} starts at {service.origin}, "
                f"cargo is at {st.node}"
            )
        self._request(st, service, t, walk_in=False, key=st.booked[st.idx])

    def _request(self, st: _State, service: Service, t: float, walk_in: bool, key: float) -> None:
        if not service.scheduled:
            self._depart_truck(st, service, t)
            return
        if service.id in self.processed or t > service.cutoff + EPS:
            self._miss(st, service, t, "late_arrival")
            return
        self.queues.setdefault(service.id, []).append((1 if walk_in else 0, key, self.seq, st.oid))
        self.seq += 1
        if service.id not in self.cutoff_pushed:
            self.cutoff_pushed.add(service.id)
            self._push(service.cutoff, 1, "cutoff", service.id)

    def _on_cutoff(self, sid: int, t: float) -> None:
        self.processed.add(sid)
        service = self.net.services[sid]
        capacity = service.capacity * self.scen.capacity_factor(sid)
        load = 0.0
        bumped = 0.0
        queue = sorted(self.queues.get(sid, []))
        self.service_orders[sid] = {oid for *_, oid in queue}
        for _, _, _, oid in queue:
            st = self.states[oid]
            if capacity <= EPS:
                bumped += st.out.volume
                self._miss(st, service, t, "cancelled")
            elif load + st.out.volume <= capacity + EPS:
                load += st.out.volume
                self._board(st, service)
            else:
                bumped += st.out.volume
                self._miss(st, service, t, "capacity")
        self.service_load[sid] = load
        if bumped > 0:
            self.service_bumped[sid] = bumped

    # -------------------------------------------------------------- actions

    def _board(self, st: _State, service: Service) -> None:
        delay = self.scen.delay(service.id)
        arrival = service.arrival + delay
        wait = max(0.0, service.cutoff - st.ready)
        self._leave_yard(st)
        st.transit = (service, service.cutoff, service.departure, arrival, wait)
        st.prev_delay = delay
        self._push(arrival, -1, "arrive", st.oid)

    def _depart_truck(self, st: _State, service: Service, t: float) -> None:
        load_start = max(t, st.ready)
        departure = load_start + service.load_time
        travel = service.travel_time * self.scen.truck_factor(service.origin, service.destination, departure)
        arrival = departure + travel
        wait = load_start - st.ready
        self._leave_yard(st)
        st.transit = (service, load_start, departure, arrival, wait)
        st.prev_delay = travel - service.travel_time
        self._push(arrival, -1, "arrive", st.oid)

    def _leave_yard(self, st: _State) -> None:
        self.yard[st.node] = self.yard.get(st.node, 0.0) - st.out.volume

    def _on_arrive(self, st: _State, t: float) -> None:
        service, load_start, departure, arrival, wait = st.transit
        st.transit = None
        node = service.destination
        handling = self.scen.handling(node, arrival) if service.scheduled else 0.0
        if node != st.order.destination:
            capacity = self.net.yard_capacity[node]
            load = self.yard.get(node, 0.0) + self.yard_bg.get((node, self.net.slot_of(arrival)), 0.0)
            overflow = max(0.0, load - capacity) / capacity
            handling += self.e.config.yard_congestion_hours * overflow
        ready = arrival + service.unload_time + handling
        st.prev_handling = handling
        self._record_leg(st, service, load_start, departure, arrival, ready, wait)

    def _record_leg(self, st, service, load_start, departure, arrival, ready, wait) -> None:
        out = st.out
        out.legs.append(
            LegOutcome(
                planned_service=st.planned[st.idx],
                service=service.id,
                origin=service.origin,
                destination=service.destination,
                ready_at_origin=st.ready,
                load_start=load_start,
                departure=departure,
                arrival=arrival,
                ready=ready,
                wait=wait,
            )
        )
        if load_start > st.ready + EPS:
            out.occupancy.append((st.node, st.ready, load_start))
        out.cost.transport += out.volume * service.cost_per_teu
        out.cost.storage += out.volume * self.e.cost.storage_cost * wait
        st.prev_service = service.id
        st.idx += 1
        st.node = service.destination
        st.ready = ready
        st.last_arrival = arrival
        self._push(ready, 0, "ready", st.oid)

    def _miss(self, st: _State, service: Service, t: float, reason: str) -> None:
        st.out.missed += 1
        replacement = self._find_replacement(st, service, t)
        if replacement is None:
            new_load_start = math.nan
        elif replacement.scheduled:
            new_load_start = replacement.cutoff
        else:
            new_load_start = max(t, st.ready)
        self.records.append(
            PropagationRecord(
                order_id=st.oid,
                node=st.node,
                planned_service=service.id,
                replacement_service=None if replacement is None else replacement.id,
                reason=reason,
                time=t,
                ready=st.ready,
                cutoff=service.cutoff,
                lateness=max(0.0, st.ready - service.cutoff) if reason == "late_arrival" else 0.0,
                wait_added=0.0 if replacement is None else max(0.0, new_load_start - service.cutoff),
                upstream_service=st.prev_service,
                upstream_delay=st.prev_delay,
                handling_extra=st.prev_handling,
            )
        )
        if replacement is None:
            self._finish(st, t + self.e.config.stranded_delay, stranded=True)
            return
        st.chain[st.idx] = replacement.id
        self._request(st, replacement, t, walk_in=True, key=t)

    def _find_replacement(self, st: _State, missed: Service, t: float) -> Service | None:
        now = max(t, st.ready)
        options = [
            s
            for s in self.net.scheduled_on_leg(missed.origin, missed.destination)
            if s.id != missed.id
            and s.id not in self.processed
            and s.cutoff >= now - EPS
            and s.capacity >= st.out.volume
            and self.scen.capacity_factor(s.id) > 0
        ]
        within = [s for s in options if s.cutoff - now <= self.e.config.rebook_max_wait]
        if within:
            return min(within, key=lambda s: (s.arrival, s.cutoff, s.id))
        truck = self.net.truck_on_leg(missed.origin, missed.destination)
        if truck is not None and self.e.allow_truck(st.oid):
            return truck
        if options:
            return min(options, key=lambda s: (s.arrival, s.cutoff, s.id))
        return None

    def _finish(self, st: _State, arrival: float, stranded: bool) -> None:
        out = st.out
        out.final_arrival = arrival
        out.stranded = stranded
        out.delay = max(0.0, arrival - st.order.due)
        out.cost.delay = st.order.delay_cost * out.volume * out.delay
