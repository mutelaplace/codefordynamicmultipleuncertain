"""Joint (capacity-aware) evaluation of whole plans and post-adjustment risk re-check."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dynamic_intermodal.config import OptimizationConfig, RiskConfig
from dynamic_intermodal.conflict.detector import ConflictDetector, ConflictRecord, yard_profile
from dynamic_intermodal.network import ServiceNetwork
from dynamic_intermodal.plan import TransportPlan
from dynamic_intermodal.risk.metrics import PlanRisk, assess_plan, weighted_cvar
from dynamic_intermodal.risk.propagation import PropagationEngine, ScenarioOutcome
from dynamic_intermodal.uncertainty.scenarios import ScenarioSet


@dataclass
class PlanEvaluation:
    outcomes: list[ScenarioOutcome]
    risk: PlanRisk
    conflicts: list[ConflictRecord]
    objective: float
    components: dict[str, float] = field(default_factory=dict)
    scenario_loss: np.ndarray | None = None  # per-scenario cost + yard penalty (for paired tests)
    weights: np.ndarray | None = None


class PlanEvaluator:
    def __init__(
        self,
        network: ServiceNetwork,
        engine: PropagationEngine,
        detector: ConflictDetector,
        opt_config: OptimizationConfig,
        risk_config: RiskConfig,
    ):
        self.network = network
        self.engine = engine
        self.detector = detector
        self.opt = opt_config
        self.risk_config = risk_config

    def evaluate(
        self, plan: TransportPlan, scenarios: ScenarioSet, t: float, change_cost: float = 0.0
    ) -> PlanEvaluation:
        w = scenarios.weights
        outcomes = [self.engine.simulate(plan, view) for view in scenarios.views()]
        risk = assess_plan(outcomes, w, self.risk_config.cvar_alpha)
        conflicts = self.detector.detect(outcomes, w, t)
        yard_overflow = np.array([self._yard_overflow(o, t) for o in outcomes])
        beta = self.risk_config.late_prob_threshold
        late_excess = sum(max(0.0, r.p_late - beta) for r in risk.orders.values())
        components = {
            "expected_cost": risk.expected_cost,
            "cvar_delay_cost": weighted_cvar(risk.scenario_delay_cost, w, self.risk_config.cvar_alpha),
            "yard_overflow": float(w @ yard_overflow),
            "late_excess": late_excess,
            "change_cost": change_cost,
        }
        objective = (
            components["expected_cost"]
            + self.opt.cvar_weight * components["cvar_delay_cost"]
            + self.opt.yard_overflow_penalty * components["yard_overflow"]
            + self.opt.late_prob_penalty * late_excess
            + change_cost
        )
        loss = risk.scenario_cost + self.opt.yard_overflow_penalty * yard_overflow
        return PlanEvaluation(outcomes, risk, conflicts, objective, components, loss, w)

    def _yard_overflow(self, outcome: ScenarioOutcome, t: float) -> float:
        return sum(
            max(0.0, load - self.network.yard_capacity[node])
            for (node, _), (load, _) in yard_profile(outcome, self.network, t).items()
        )


def paired_improvement(current: PlanEvaluation, proposal: PlanEvaluation) -> tuple[float, float]:
    """Objective gain of the proposal and the standard error of the per-scenario loss difference
    (both plans evaluated on the same scenarios)."""
    w = current.weights
    diff = current.scenario_loss - proposal.scenario_loss
    mean = float(w @ diff)
    n_eff = 1.0 / float(np.sum(w**2))
    se = float(np.sqrt(w @ (diff - mean) ** 2 / n_eff))
    return current.objective - proposal.objective, se


class RiskRechecker:
    """Finds what a candidate adjustment newly breaks: new / aggravated conflicts and orders
    outside the adjustment scope whose risk now exceeds the requirement."""

    def __init__(self, risk_config: RiskConfig):
        self.config = risk_config

    def new_conflicts(
        self, before: list[ConflictRecord], after: list[ConflictRecord], scope: set[int]
    ) -> list[ConflictRecord]:
        previous = {c.key: c for c in before}
        result = []
        for c in after:
            if c.probability < self.config.conflict_link_prob:
                continue
            old = previous.get(c.key)
            if old is None or c.probability > old.probability + 0.05 or (c.order_ids - old.order_ids - scope):
                result.append(c)
        return result

    def worsened_orders(
        self, before: PlanRisk, after: PlanRisk, scope: set[int], changeable: set[int]
    ) -> set[int]:
        result = set()
        for oid in changeable - scope:
            now = after.orders[oid]
            if not now.exceeds(self.config):
                continue
            prev = before.orders[oid]
            if not prev.exceeds(self.config) or now.expected_delay > prev.expected_delay + 1.0:
                result.add(oid)
        return result
