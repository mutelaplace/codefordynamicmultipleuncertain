"""Aggregation of scenario outcomes into order- and plan-level risk measures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from dynamic_intermodal.risk.propagation import ScenarioOutcome


def weighted_cvar(values: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    """CVaR_alpha (mean of the worst 1-alpha tail) of a discrete weighted distribution."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    k = int(np.searchsorted(cum, alpha - 1e-12))
    var = v[min(k, len(v) - 1)]
    return float(var + np.sum(w * np.maximum(v - var, 0.0)) / (1.0 - alpha))


@dataclass
class OrderRisk:
    order_id: int
    p_late: float
    expected_delay: float
    cvar_delay: float
    p_miss: float
    expected_missed: float
    expected_cost: float
    expected_delay_cost: float
    expected_arrival: float

    def exceeds(self, risk_config) -> list[str]:
        reasons = []
        if self.p_late > risk_config.late_prob_threshold:
            reasons.append(f"P(late)={self.p_late:.2f}")
        if self.expected_delay > risk_config.expected_delay_threshold:
            reasons.append(f"E[delay]={self.expected_delay:.1f}h")
        if self.p_miss > risk_config.miss_prob_threshold:
            reasons.append(f"P(miss)={self.p_miss:.2f}")
        return reasons


@dataclass
class PlanRisk:
    orders: dict[int, OrderRisk]
    scenario_cost: np.ndarray
    scenario_delay_cost: np.ndarray
    scenario_bumped: np.ndarray
    weights: np.ndarray
    alpha: float
    extra: dict = field(default_factory=dict)

    @property
    def expected_cost(self) -> float:
        return float(self.weights @ self.scenario_cost)

    @property
    def expected_delay_cost(self) -> float:
        return float(self.weights @ self.scenario_delay_cost)

    @property
    def cvar_delay_cost(self) -> float:
        return weighted_cvar(self.scenario_delay_cost, self.weights, self.alpha)

    @property
    def expected_bumped(self) -> float:
        return float(self.weights @ self.scenario_bumped)

    def risky_orders(self, risk_config, candidates: Iterable[int] | None = None) -> dict[int, list[str]]:
        ids = self.orders if candidates is None else candidates
        result = {}
        for oid in ids:
            reasons = self.orders[oid].exceeds(risk_config)
            if reasons:
                result[oid] = reasons
        return result


def assess_plan(
    outcomes: list[ScenarioOutcome],
    weights: np.ndarray,
    alpha: float,
    order_ids: Iterable[int] | None = None,
) -> PlanRisk:
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    ids = sorted(order_ids if order_ids is not None else outcomes[0].orders)
    orders: dict[int, OrderRisk] = {}
    for oid in ids:
        delay = np.array([o.orders[oid].delay for o in outcomes])
        missed = np.array([o.orders[oid].missed for o in outcomes], dtype=float)
        cost = np.array([o.orders[oid].cost.total for o in outcomes])
        delay_cost = np.array([o.orders[oid].cost.delay for o in outcomes])
        arrival = np.array([o.orders[oid].final_arrival for o in outcomes])
        orders[oid] = OrderRisk(
            order_id=oid,
            p_late=float(weights @ (delay > 1e-6)),
            expected_delay=float(weights @ delay),
            cvar_delay=weighted_cvar(delay, weights, alpha),
            p_miss=float(weights @ (missed > 0)),
            expected_missed=float(weights @ missed),
            expected_cost=float(weights @ cost),
            expected_delay_cost=float(weights @ delay_cost),
            expected_arrival=float(weights @ arrival),
        )
    scenario_cost = np.array([sum(o.orders[i].cost.total for i in ids) for o in outcomes])
    scenario_delay = np.array([sum(o.orders[i].cost.delay for i in ids) for o in outcomes])
    scenario_bumped = np.array([sum(o.service_bumped.values()) for o in outcomes])
    return PlanRisk(orders, scenario_cost, scenario_delay, scenario_bumped, weights, alpha)
