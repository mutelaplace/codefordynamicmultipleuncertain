"""Operational observations generated from the hidden truth.

Only information whose time stamp is <= the current epoch is released:
  * actual arrival of every barge / rail service (arrival delay)          -> "arrival"
  * terminal handling progress (first unloading completed in a block)     -> "handling"
  * road traffic state of each truck corridor at the start of a block     -> "traffic"
  * capacity notice of each service (cancellation / reduction)            -> "capacity"
  * confirmed order volume and actual cargo release                       -> "order"
  * noisy real-time indicators of the external factors                    -> "indicator"
  * "not yet happened" facts (a service still not arrived, cargo not yet released, handling
    still in progress) are released as censored observations (value is a lower bound).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamic_intermodal.config import UncertaintyConfig
from dynamic_intermodal.network import ServiceNetwork
from dynamic_intermodal.uncertainty.scenarios import ScenarioSet


@dataclass(frozen=True)
class Observation:
    time: float
    kind: str  # value | exceeds | indicator
    source: str
    key: tuple | None
    value: float
    factor: str | None = None
    noise: float = 0.0


class Observer:
    def __init__(
        self,
        truth: ScenarioSet,
        network: ServiceNetwork,
        config: UncertaintyConfig,
        rng: np.random.Generator,
    ):
        self.truth = truth
        self.view = truth.view(0)
        self.network = network
        self.config = config
        self.rng = rng
        self.emitted: set[tuple] = set()
        self.reveal_time: dict[tuple, tuple[float, str]] = {}
        self._handling_arrivals: dict[tuple, tuple[float, float]] = {}  # (node, block) -> (unload end, arrival)
        self._build_reveal_times()

    def _build_reveal_times(self) -> None:
        net, view, cfg = self.network, self.view, self.config
        block = net.config.block_hours
        for s in net.scheduled_services:
            arrival = s.arrival + view.delay(s.id)
            self.reveal_time[("delay", s.id)] = (arrival, "arrival")
            self.reveal_time[("capacity", s.id)] = (s.cutoff - cfg.capacity_notice_hours, "capacity")
            if view.capacity_factor(s.id) > 0:
                key = (s.destination, net.block_of(arrival))
                unload_end = arrival + s.unload_time
                if unload_end < self._handling_arrivals.get(key, (np.inf, 0.0))[0]:
                    self._handling_arrivals[key] = (unload_end, arrival)
        for node in net.handling_nodes:
            for b in range(net.n_blocks):
                key = ("handling", node, b)
                block_end = (b + 1) * block
                first = self._handling_arrivals.get((node, b))
                ready = first[0] + view.handling(node, first[1]) if first is not None else block_end
                self.reveal_time[key] = (min(ready, block_end), "handling")
        for o, d in net.truck_legs:
            for b in range(net.n_blocks):
                self.reveal_time[("truck", o, d, b)] = (b * block, "traffic")
        for oid, order in net.orders.items():
            self.reveal_time[("demand", oid)] = (order.release - cfg.demand_notice_hours, "order")
            self.reveal_time[("release", oid)] = (order.release + view.release_delay(oid), "order")

    def _truth_value(self, key: tuple) -> float:
        j = self.truth.model.index(key)
        return float(self.truth.X[0, j])

    def collect(self, t: float, indicators: bool = True) -> list[Observation]:
        observations: list[Observation] = []
        for key, (when, source) in self.reveal_time.items():
            if key in self.emitted or when > t:
                continue
            self.emitted.add(key)
            observations.append(Observation(max(when, 0.0), "value", source, key, self._truth_value(key)))

        net = self.network
        for s in net.scheduled_services:
            key = ("delay", s.id)
            if key not in self.emitted and s.arrival < t:
                observations.append(Observation(t, "exceeds", "arrival", key, t - s.arrival))
        for oid, order in net.orders.items():
            key = ("release", oid)
            if key not in self.emitted and order.release < t:
                observations.append(Observation(t, "exceeds", "order", key, t - order.release))
        for (node, b), (unload_end, _) in self._handling_arrivals.items():
            key = ("handling", node, b)
            if key not in self.emitted and unload_end < t:
                observations.append(Observation(t, "exceeds", "handling", key, t - unload_end))

        if indicators:
            F = self.truth.F[0]
            for name in self.config.indicators:
                f = self.truth.model.factor_index(name)
                y = float(F[f] + self.config.indicator_noise * self.rng.standard_normal())
                observations.append(
                    Observation(t, "indicator", "indicator", None, y, factor=name, noise=self.config.indicator_noise)
                )
        return observations

    def reveal(self, key: tuple, t: float, source: str) -> Observation | None:
        """Early release of a value, e.g. a carrier disclosing its capacity when it rejects a booking."""
        if key in self.emitted:
            return None
        self.emitted.add(key)
        return Observation(t, "value", source, key, self._truth_value(key))
