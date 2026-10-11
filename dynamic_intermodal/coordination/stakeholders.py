"""Participating stakeholders (rail / barge / truck carriers, terminals, logistics companies)
with the services and resources they provide and the change permissions they grant."""

from __future__ import annotations

from dataclasses import dataclass, field

from dynamic_intermodal.config import CarrierPolicy, StakeholderConfig
from dynamic_intermodal.network import Service, ServiceNetwork


@dataclass
class Stakeholder:
    id: str
    kind: str  # carrier | terminal | shipper
    name: str
    services: list[int] = field(default_factory=list)
    nodes: list[int] = field(default_factory=list)
    orders: list[int] = field(default_factory=list)
    policy: CarrierPolicy | None = None
    accept_prob: float = 1.0
    allow_truck: bool = True
    max_route_changes: int = 3


class StakeholderRegistry:
    def __init__(self, network: ServiceNetwork, config: StakeholderConfig | None = None):
        self.network = network
        self.config = config or StakeholderConfig()
        self.stakeholders: dict[str, Stakeholder] = {}

        for service in network.services.values():
            carrier = self.stakeholders.get(service.carrier_id)
            if carrier is None:
                policy = self.config.carrier_policies[service.mode]
                carrier = Stakeholder(
                    id=service.carrier_id,
                    kind="carrier",
                    name=f"{service.mode} operator {service.carrier_id.split('-')[-1]}",
                    policy=policy,
                    accept_prob=policy.accept_prob,
                )
                self.stakeholders[carrier.id] = carrier
            carrier.services.append(service.id)

        for node, terminal in network.terminals.items():
            sid = f"terminal-{terminal.name}"
            self.stakeholders[sid] = Stakeholder(
                id=sid,
                kind="terminal",
                name=terminal.name,
                nodes=[node],
                accept_prob=self.config.terminal_accept_prob,
            )

        for order in network.orders.values():
            shipper = self.stakeholders.get(order.shipper_id)
            if shipper is None:
                shipper = Stakeholder(
                    id=order.shipper_id,
                    kind="shipper",
                    name=order.shipper_id,
                    allow_truck=self.config.shipper_allow_truck.get(order.shipper_id, True),
                    max_route_changes=self.config.shipper_max_route_changes.get(order.shipper_id, 3),
                )
                self.stakeholders[shipper.id] = shipper
            shipper.orders.append(order.id)

    # ------------------------------------------------------------- lookups

    def carrier_of(self, service_id: int) -> Stakeholder:
        return self.stakeholders[self.network.services[service_id].carrier_id]

    def terminal_of(self, node: int) -> Stakeholder:
        return self.stakeholders[f"terminal-{self.network.node_name(node)}"]

    def shipper_of(self, order_id: int) -> Stakeholder:
        return self.stakeholders[self.network.orders[order_id].shipper_id]

    def of_kind(self, kind: str) -> list[Stakeholder]:
        return [s for s in self.stakeholders.values() if s.kind == kind]

    # ---------------------------------------------------------- permissions

    def allow_truck(self, order_id: int) -> bool:
        return self.shipper_of(order_id).allow_truck

    def can_book(self, service: Service, t: float) -> bool:
        """May a *new* booking on this service still be requested at time t?"""
        policy = self.carrier_of(service.id).policy
        if not policy.accepts_new_bookings:
            return False
        if service.scheduled:
            return service.cutoff - t >= policy.change_cutoff_hours
        return True

    def booking_locked(self, service: Service, t: float, cargo_ready: float | None) -> bool:
        """Is an existing booking past the carrier's change cut-off (no longer changeable)?"""
        policy = self.carrier_of(service.id).policy
        if service.scheduled:
            return service.cutoff - t < policy.change_cutoff_hours
        return cargo_ready is not None and cargo_ready - t < policy.change_cutoff_hours
