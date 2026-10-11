import numpy as np

from dynamic_intermodal.config import RiskConfig
from dynamic_intermodal.conflict.detector import ConflictDetector
from dynamic_intermodal.conflict.scope import ScopeIdentifier
from dynamic_intermodal.plan import Leg, TransportPlan
from dynamic_intermodal.risk.metrics import assess_plan
from dynamic_intermodal.risk.propagation import PropagationEngine


def by_name(network, name):
    return next(s for s in network.services.values() if s.name == name)


def test_node_slot_order_conflict_and_scope(network, fake_view):
    # Three orders wait on the Duisburg yard for the same train after an early barge.
    barge, train = by_name(network, "Barge14"), by_name(network, "Train3")
    plan = TransportPlan()
    for oid in (0, 1, 4):
        plan.set(oid, [Leg(barge.id), Leg(train.id)])
    engine = PropagationEngine(network)
    views = [fake_view(), fake_view(demand={0: 1.4, 1: 1.4, 4: 1.4})]
    outcomes = [engine.simulate(plan, v) for v in views]
    weights = np.array([0.5, 0.5])
    conflicts = ConflictDetector(network, min_prob=0.05).detect(outcomes, weights, 0.0)
    yard = [c for c in conflicts if c.resource == "yard" and c.node == barge.destination]
    assert yard, "expected a yard conflict at the transfer terminal"
    assert yard[0].order_ids == frozenset({0, 1, 4})

    risk = assess_plan(outcomes, weights, 0.9)
    identifier = ScopeIdentifier(RiskConfig(conflict_prob_threshold=0.2), "conflict")
    scope = identifier.identify(risk, conflicts, changeable={0, 1, 4})
    assert scope.triggered and scope.orders == {0, 1, 4}
    direct_only = ScopeIdentifier(RiskConfig(conflict_prob_threshold=0.2), "direct").identify(risk, conflicts, {0, 1, 4})
    assert direct_only.orders <= scope.orders
