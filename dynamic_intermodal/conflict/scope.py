"""Collaborative adjustment scope: directly disturbed orders plus orders linked via conflicts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from dynamic_intermodal.config import RiskConfig
from dynamic_intermodal.conflict.detector import ConflictRecord
from dynamic_intermodal.risk.metrics import PlanRisk


@dataclass
class AdjustmentScope:
    direct: set[int] = field(default_factory=set)
    indirect: set[int] = field(default_factory=set)
    reasons: dict[int, list[str]] = field(default_factory=dict)
    triggering_conflicts: list[ConflictRecord] = field(default_factory=list)

    @property
    def orders(self) -> set[int]:
        return self.direct | self.indirect

    @property
    def triggered(self) -> bool:
        return bool(self.direct) or bool(self.triggering_conflicts)

    def add(self, order_id: int, reason: str, direct: bool = False) -> None:
        if direct:
            self.direct.add(order_id)
            self.indirect.discard(order_id)
        elif order_id not in self.direct:
            self.indirect.add(order_id)
        self.reasons.setdefault(order_id, []).append(reason)


class ScopeIdentifier:
    def __init__(self, risk_config: RiskConfig, mode: str = "conflict"):
        if mode not in {"conflict", "direct", "all"}:
            raise ValueError(f"Unknown scope mode: {mode}")
        self.config = risk_config
        self.mode = mode

    def identify(
        self,
        risk: PlanRisk,
        conflicts: list[ConflictRecord],
        changeable: Iterable[int],
    ) -> AdjustmentScope:
        changeable = set(changeable)
        scope = AdjustmentScope()

        for oid, reasons in risk.risky_orders(self.config, changeable).items():
            scope.add(oid, "risk: " + ", ".join(reasons), direct=True)
        scope.triggering_conflicts = [
            c for c in conflicts
            if c.probability >= self.config.conflict_prob_threshold and c.order_ids & changeable
        ]

        if self.mode == "all":
            if scope.triggered:
                for oid in changeable - scope.direct:
                    scope.add(oid, "global re-planning")
            return scope
        if self.mode == "direct":
            return scope

        # Orders on a high-probability conflict are disturbed through the shared resource.
        for c in scope.triggering_conflicts:
            for oid in c.order_ids & changeable:
                scope.add(oid, f"conflict {self._label(c)} p={c.probability:.2f}")

        # Expand along the order-resource conflict graph.
        frontier = set(scope.orders)
        links = [c for c in conflicts if c.probability >= self.config.conflict_link_prob]
        for _ in range(self.config.scope_hops):
            new: set[int] = set()
            for c in links:
                if c.order_ids & frontier:
                    for oid in (c.order_ids & changeable) - scope.orders:
                        scope.add(oid, f"shares {self._label(c)} p={c.probability:.2f}")
                        new.add(oid)
            if not new:
                break
            frontier = new
        return scope

    @staticmethod
    def _label(c: ConflictRecord) -> str:
        if c.resource == "yard":
            return f"yard@node{c.node}[{c.slot_start:.0f},{c.slot_end:.0f})"
        return f"service{c.service_id}@node{c.node}"
