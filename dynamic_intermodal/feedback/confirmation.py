"""Execution feedback: booking-change requests are sent to the stakeholders, who confirm or
reject them (deadline check, capacity check, own acceptance); confirmations commit the new
bookings and release the capacity of cancelled ones, rejections and disclosed capacities are
fed into the next optimisation round."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dynamic_intermodal.config import UncertaintyConfig
from dynamic_intermodal.coordination.candidates import CandidatePlan
from dynamic_intermodal.coordination.stakeholders import StakeholderRegistry
from dynamic_intermodal.network import ServiceNetwork
from dynamic_intermodal.plan import Leg, TransportPlan
from dynamic_intermodal.uncertainty.scenarios import ScenarioSet
from dynamic_intermodal.update.observations import Observation, Observer


@dataclass
class ChangeRequest:
    order_id: int
    old_legs: tuple[Leg, ...]
    new_legs: tuple[Leg, ...]
    new_services: tuple[int, ...]
    cancelled_services: tuple[int, ...]
    new_transfer_nodes: tuple[int, ...]
    change_cost: float
    issued_at: float
    rationale: str = ""

    @classmethod
    def from_candidate(cls, cand: CandidatePlan, old_legs, network: ServiceNetwork, t: float, rationale=""):
        old_nodes = {network.services[leg.service_id].destination for leg in old_legs[:-1]}
        new_nodes = [network.services[leg.service_id].destination for leg in cand.legs[:-1]]
        return cls(
            order_id=cand.order_id,
            old_legs=tuple(old_legs),
            new_legs=tuple(Leg(leg.service_id, t if leg.service_id in cand.new_services else leg.booked_at) for leg in cand.legs),
            new_services=cand.new_services,
            cancelled_services=cand.cancelled_services,
            new_transfer_nodes=tuple(n for n in new_nodes if n not in old_nodes) if old_legs else (),
            change_cost=cand.change_cost,
            issued_at=t,
            rationale=rationale,
        )


@dataclass
class StakeholderResponse:
    stakeholder_id: str
    accepted: bool
    reason: str
    responded_at: float
    service_id: int | None = None


@dataclass
class ConfirmationResult:
    request: ChangeRequest
    accepted: bool
    responses: list[StakeholderResponse] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)

    @property
    def rejected_services(self) -> list[int]:
        return [r.service_id for r in self.responses if not r.accepted and r.service_id is not None]


class ConfirmationSimulator:
    def __init__(
        self,
        network: ServiceNetwork,
        registry: StakeholderRegistry,
        truth: ScenarioSet,
        observer: Observer,
        config: UncertaintyConfig,
        rng: np.random.Generator,
    ):
        self.network = network
        self.registry = registry
        self.truth_view = truth.view(0)
        self.observer = observer
        self.config = config
        self.rng = rng

    def known_capacity(self, service_id: int, t: float) -> float:
        """Capacity as known to the carrier itself at time t (reductions known after the notice)."""
        service = self.network.services[service_id]
        if t >= service.cutoff - self.config.capacity_notice_hours:
            return service.capacity * self.truth_view.capacity_factor(service_id)
        return service.capacity

    def process(
        self,
        requests: list[ChangeRequest],
        plan: TransportPlan,
        t: float,
        declared_volume: dict[int, float],
        initial: bool = False,
    ) -> list[ConfirmationResult]:
        booked: dict[int, float] = {}
        for oid, order_plan in plan.items():
            for leg in order_plan.legs:
                booked[leg.service_id] = booked.get(leg.service_id, 0.0) + declared_volume[oid]
        results = []
        for req in requests:
            volume = declared_volume[req.order_id]
            responses: list[StakeholderResponse] = []
            observations: list[Observation] = []
            for sid in req.new_services:
                service = self.network.services[sid]
                carrier = self.registry.carrier_of(sid)
                policy = carrier.policy
                t_resp = t + policy.response_delay
                if service.scheduled and not initial and t_resp > service.cutoff - policy.change_cutoff_hours:
                    responses.append(StakeholderResponse(carrier.id, False, "deadline passed", t_resp, sid))
                    continue
                capacity = self.known_capacity(sid, t_resp)
                if service.scheduled and booked.get(sid, 0.0) + volume > capacity + 1e-9:
                    responses.append(StakeholderResponse(carrier.id, False, "insufficient capacity", t_resp, sid))
                    obs = self.observer.reveal(("capacity", sid), t_resp, "capacity")
                    if obs is not None and capacity < service.capacity:
                        observations.append(obs)
                    continue
                if not initial and self.rng.random() > carrier.accept_prob:
                    responses.append(StakeholderResponse(carrier.id, False, "declined by carrier", t_resp, sid))
                    continue
                responses.append(StakeholderResponse(carrier.id, True, "confirmed", t_resp, sid))
            for node in req.new_transfer_nodes:
                terminal = self.registry.terminal_of(node)
                ok = initial or self.rng.random() <= terminal.accept_prob
                responses.append(
                    StakeholderResponse(terminal.id, ok, "yard appointment confirmed" if ok else "yard appointment declined", t + 0.5)
                )
            for sid in req.cancelled_services:
                carrier = self.registry.carrier_of(sid)
                responses.append(StakeholderResponse(carrier.id, True, "cancellation, capacity released", t, sid))

            accepted = all(r.accepted for r in responses)
            if accepted:
                plan.set(req.order_id, req.new_legs)
                for sid in req.new_services:
                    booked[sid] = booked.get(sid, 0.0) + volume
                for sid in req.cancelled_services:
                    booked[sid] = booked.get(sid, 0.0) - volume
            results.append(ConfirmationResult(req, accepted, responses, observations))
        return results
