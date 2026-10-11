"""Execution state at a decision epoch: realised legs are fixed, bookings past their change
cut-off are locked, and automatic rebookings that already happened are written into the plan."""

from __future__ import annotations

from dataclasses import dataclass

from dynamic_intermodal.coordination.stakeholders import StakeholderRegistry
from dynamic_intermodal.network import ServiceNetwork
from dynamic_intermodal.plan import Leg, TransportPlan
from dynamic_intermodal.risk.propagation import ScenarioOutcome


@dataclass
class OrderExecState:
    order_id: int
    status: str  # waiting | at_node | in_transit | delivered
    executed: int  # legs whose loading has started
    locked: int  # leading legs that can no longer change (executed + past change cut-off)
    node: int  # terminal where the cargo is, or will next be ready
    known_ready: float | None  # realised ready time at ``node`` if already observed
    released: bool

    @property
    def changeable(self) -> bool:
        return self.status != "delivered"


def sync_with_execution(
    plan: TransportPlan,
    truth: ScenarioOutcome,
    network: ServiceNetwork,
    registry: StakeholderRegistry,
    t: float,
) -> tuple[TransportPlan, dict[int, OrderExecState]]:
    synced = plan.copy()
    states: dict[int, OrderExecState] = {}
    records_by_order: dict[int, list] = {}
    for rec in truth.records:
        if rec.time <= t:
            records_by_order.setdefault(rec.order_id, []).append(rec)

    for oid, order_plan in plan.items():
        order = network.orders[oid]
        out = truth.orders[oid]
        legs = list(order_plan.legs)
        executed_legs = [leg for leg in out.legs if leg.load_start <= t]
        n_exec = len(executed_legs)

        new_legs: list[Leg] = []
        for i, realised in enumerate(executed_legs):
            booked_at = legs[i].booked_at if i < len(legs) and legs[i].service_id == realised.service else _rebook_time(
                records_by_order.get(oid, []), realised.service
            )
            new_legs.append(Leg(realised.service, booked_at))
        if n_exec < len(legs):
            current = legs[n_exec]
            for rec in records_by_order.get(oid, []):
                if rec.planned_service == current.service_id and rec.replacement_service is not None:
                    current = Leg(rec.replacement_service, rec.time)
            new_legs.append(current)
            new_legs.extend(legs[n_exec + 1:])
        synced.set(oid, new_legs)

        released = out.release <= t
        if n_exec == 0:
            node, known_ready = order.origin, (out.release if released else None)
            status = "waiting"
        else:
            last = executed_legs[-1]
            node = last.destination
            known_ready = last.ready if last.ready <= t else None
            delivered = node == order.destination and last.arrival <= t
            status = "delivered" if delivered else ("at_node" if known_ready is not None else "in_transit")

        locked = n_exec
        for leg in new_legs[n_exec:]:
            service = network.services[leg.service_id]
            ready = known_ready if locked == n_exec else None
            if registry.booking_locked(service, t, ready):
                locked += 1
            else:
                break
        states[oid] = OrderExecState(
            order_id=oid,
            status=status,
            executed=n_exec,
            locked=min(locked, len(new_legs)),
            node=node,
            known_ready=known_ready,
            released=released,
        )
    return synced, states


def _rebook_time(records: list, service_id: int) -> float:
    for rec in records:
        if rec.replacement_service == service_id:
            return rec.time
    return 0.0
