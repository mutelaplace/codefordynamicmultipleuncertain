"""Candidate plan generation within the services, resources and permissions of the stakeholders."""

from __future__ import annotations

from dataclasses import dataclass

from dynamic_intermodal.config import OptimizationConfig
from dynamic_intermodal.coordination.stakeholders import StakeholderRegistry
from dynamic_intermodal.network import Service, ServiceNetwork
from dynamic_intermodal.plan import Leg, OrderPlan, changed_positions


@dataclass(frozen=True)
class CandidatePlan:
    order_id: int
    legs: tuple[Leg, ...]
    is_current: bool
    changed_legs: int
    new_services: tuple[int, ...]
    cancelled_services: tuple[int, ...]
    carriers: frozenset[str]  # carriers that receive a new booking or a cancellation
    terminals: frozenset[str]  # terminals newly used for transfer
    change_cost: float
    nominal_score: float

    @property
    def service_ids(self) -> tuple[int, ...]:
        return tuple(leg.service_id for leg in self.legs)

    @property
    def stakeholders(self) -> frozenset[str]:
        return self.carriers | self.terminals


class CandidateGenerator:
    def __init__(
        self,
        network: ServiceNetwork,
        registry: StakeholderRegistry,
        config: OptimizationConfig,
        max_legs: int,
    ):
        self.network = network
        self.registry = registry
        self.config = config
        self.max_legs = max_legs

    def generate(
        self,
        order_id: int,
        current: OrderPlan | None,
        locked: int,
        start_ready: float,
        t: float,
        forbidden: set[tuple[int, int]] | frozenset = frozenset(),
        changes_so_far: int = 0,
        max_candidates: int | None = None,
    ) -> list[CandidatePlan]:
        net = self.network
        order = net.orders[order_id]
        current_legs = current.legs if current is not None else ()
        prefix = current_legs[:locked]
        start_node = net.services[prefix[-1].service_id].destination if prefix else order.origin
        current_suffix = tuple(leg.service_id for leg in current_legs[locked:])
        booked_at = {leg.service_id: leg.booked_at for leg in current_legs}
        shipper = self.registry.shipper_of(order_id)
        limit = max_candidates or self.config.max_candidates

        result: list[CandidatePlan] = []
        if current is not None:
            result.append(self._make(order_id, current_legs, current_legs, True, start_ready, locked))
            if changes_so_far >= shipper.max_route_changes:
                return result

        def allowed(service: Service) -> bool:
            if (order_id, service.id) in forbidden:
                return False
            if service.id in current_suffix:
                return True
            if not service.scheduled and not shipper.allow_truck:
                return False
            return self.registry.can_book(service, t)

        chains = net.enumerate_chains(order, start_node, start_ready, self.max_legs - locked, allowed)
        if not chains and current is None and not shipper.allow_truck:
            # No intermodal chain exists: the shipper's no-truck preference is waived for this order.
            def allowed(service: Service) -> bool:  # noqa: F811
                return (order_id, service.id) not in forbidden and self.registry.can_book(service, t)

            chains = net.enumerate_chains(order, start_node, start_ready, self.max_legs - locked, allowed)
        scored = []
        for chain in chains:
            if chain == current_suffix:
                continue
            nominal = net.nominal_chain(order, chain, start_ready)
            if nominal is None:
                continue
            scored.append((nominal[2], chain))
        scored.sort()

        # Diversify: best chain per first service first, then the remaining best chains.
        picked: list[tuple[int, ...]] = []
        seen_first: set[int] = set()
        n_new = limit - len(result)
        for _, chain in scored:
            if chain and chain[0] not in seen_first:
                seen_first.add(chain[0])
                picked.append(chain)
            if len(picked) >= n_new:
                break
        for _, chain in scored:
            if len(picked) >= n_new:
                break
            if chain not in picked:
                picked.append(chain)

        for chain in picked:
            legs = tuple(prefix) + tuple(Leg(sid, booked_at.get(sid, t)) for sid in chain)
            result.append(self._make(order_id, legs, current_legs, False, start_ready, locked))
        return result

    def _make(self, order_id, legs, current_legs, is_current, start_ready, locked) -> CandidatePlan:
        net = self.network
        order = net.orders[order_id]
        new_ids = tuple(leg.service_id for leg in legs)
        old_ids = tuple(leg.service_id for leg in current_legs)
        new_services = tuple(s for s in new_ids if s not in old_ids)
        cancelled = tuple(s for s in old_ids if s not in new_ids)
        carriers = frozenset(self.registry.carrier_of(s).id for s in new_services + cancelled)
        old_nodes = {net.services[s].destination for s in old_ids[:-1]}
        new_nodes = {net.services[s].destination for s in new_ids[:-1]}
        terminals = frozenset(self.registry.terminal_of(n).id for n in new_nodes - old_nodes) if old_ids else frozenset()
        changed = changed_positions(old_ids, new_ids) if old_ids else 0
        change_cost = (
            self.config.change_cost_per_leg * changed
            + self.config.change_cost_per_stakeholder * (len(carriers) + len(terminals))
            if old_ids
            else 0.0
        )
        nominal = net.nominal_chain(order, new_ids[locked:], start_ready)
        score = nominal[2] if nominal is not None else float("inf")
        return CandidatePlan(
            order_id=order_id,
            legs=tuple(legs),
            is_current=is_current,
            changed_legs=changed,
            new_services=new_services,
            cancelled_services=cancelled,
            carriers=carriers if old_ids else frozenset(),
            terminals=terminals,
            change_cost=change_cost,
            nominal_score=score,
        )
