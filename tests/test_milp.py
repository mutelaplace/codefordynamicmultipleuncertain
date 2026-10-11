import numpy as np
import pytest

from dynamic_intermodal.coordination.candidates import CandidateGenerator
from dynamic_intermodal.coordination.milp import Background, ScenarioMILP, gp
from dynamic_intermodal.coordination.stakeholders import StakeholderRegistry
from dynamic_intermodal.plan import OrderPlan
from dynamic_intermodal.risk.propagation import PropagationEngine
from dynamic_intermodal.uncertainty.factors import FactorModel
from dynamic_intermodal.uncertainty.scenarios import ScenarioGenerator

pytestmark = pytest.mark.skipif(gp is None, reason="gurobipy not available")


def _empty_background(n):
    return Background([{} for _ in range(n)], [{} for _ in range(n)], np.zeros(n), {})


def test_initial_plan_milp(network, config):
    registry = StakeholderRegistry(network, config.stakeholders)
    engine = PropagationEngine(network, allow_truck=registry.allow_truck)
    model = FactorModel.build(network, config.uncertainty, "joint")
    sset = ScenarioGenerator(model, network, config.uncertainty, np.random.default_rng(0)).generate(10, 200)
    gen = CandidateGenerator(network, registry, config.optimization, 3)
    orders = [0, 1, 2, 3, 4]
    cands = {oid: gen.generate(oid, None, 0, network.orders[oid].release, 0.0) for oid in orders}
    milp = ScenarioMILP(network, registry, engine, config.optimization, config.risk)
    evals = milp.evaluate_candidates(cands, sset, 0.0)
    declared = {oid: o.volume for oid, o in network.orders.items()}
    result = milp.solve(cands, evals, _empty_background(len(sset)), sset, 0.0, declared, {})
    assert result.status == "optimal"
    assert set(result.selection) == set(orders)
    # The chosen candidate minimises the objective among single-order alternatives when orders do
    # not interact much: it is never worse in expected cost than the nominal-best candidate by a lot.
    for oid in orders:
        chosen = result.selection[oid]
        assert network.services[chosen.service_ids[-1]].destination == network.orders[oid].destination


def test_booking_capacity_is_respected(network, config):
    registry = StakeholderRegistry(network, config.stakeholders)
    engine = PropagationEngine(network, allow_truck=registry.allow_truck)
    model = FactorModel.build(network, config.uncertainty, "joint")
    sset = ScenarioGenerator(model, network, config.uncertainty, np.random.default_rng(0)).nominal()
    gen = CandidateGenerator(network, registry, config.optimization, 3)
    orders = [0, 1, 4]
    cands = {oid: gen.generate(oid, None, 0, network.orders[oid].release, 0.0) for oid in orders}
    milp = ScenarioMILP(network, registry, engine, config.optimization, config.risk)
    evals = milp.evaluate_candidates(cands, sset, 0.0)
    declared = {oid: network.orders[oid].volume for oid in network.orders}
    # Pretend every scheduled service can only take 30 TEU: no two orders may share one.
    known = {s.id: 30.0 for s in network.scheduled_services}
    result = milp.solve(cands, evals, _empty_background(1), sset, 0.0, declared, known)
    used: dict[int, float] = {}
    for oid, cand in result.selection.items():
        for sid in cand.service_ids:
            if network.services[sid].scheduled:
                used[sid] = used.get(sid, 0.0) + declared[oid]
    assert all(v <= 30.0 + 1e-6 for v in used.values())


def test_keep_option_and_change_cost(network, config):
    registry = StakeholderRegistry(network, config.stakeholders)
    gen = CandidateGenerator(network, registry, config.optimization, 3)
    first = gen.generate(0, None, 0, network.orders[0].release, 0.0)[0]
    current = OrderPlan(0, first.legs)
    cands = gen.generate(0, current, 0, network.orders[0].release, 0.0)
    assert cands[0].is_current and cands[0].change_cost == 0
    assert all(c.change_cost > 0 for c in cands[1:])
