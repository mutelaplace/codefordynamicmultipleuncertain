"""Rolling-horizon controller tying the six technical points together.

    t = 0 : joint scenarios (1) -> first observations (2) -> collaborative plan generation (5)
            -> stakeholder confirmation (6)
    each epoch t:
        execute the committed plan in the hidden truth up to t, fix realised states  (2)
        observations <= t -> scenario update (re-weight / repair / regenerate)        (2)
        scenario-wise risk propagation of the current plan                            (3)
        node-slot-order conflicts, direct + linked orders = adjustment scope          (4)
        if the risk requirement is violated: permission-constrained candidates and
        joint MILP selection of the scope                                             (5)
        joint re-check; new conflicts expand the scope / raise their penalty          (6)
        change requests -> confirmations / rejections / released capacity -> next round (6)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from dynamic_intermodal.config import Config, MethodConfig
from dynamic_intermodal.conflict.detector import ConflictDetector, ConflictRecord, yard_profile
from dynamic_intermodal.conflict.scope import AdjustmentScope, ScopeIdentifier
from dynamic_intermodal.coordination.candidates import CandidateGenerator, CandidatePlan
from dynamic_intermodal.coordination.milp import Background, ScenarioMILP
from dynamic_intermodal.coordination.stakeholders import StakeholderRegistry
from dynamic_intermodal.data import ProblemInstance, load_problem
from dynamic_intermodal.feedback.confirmation import ChangeRequest, ConfirmationSimulator
from dynamic_intermodal.feedback.recheck import PlanEvaluation, PlanEvaluator, RiskRechecker, paired_improvement
from dynamic_intermodal.network import ServiceNetwork, build_network
from dynamic_intermodal.plan import TransportPlan
from dynamic_intermodal.risk.propagation import PropagationEngine, ScenarioOutcome
from dynamic_intermodal.simulation.state import OrderExecState, sync_with_execution
from dynamic_intermodal.uncertainty.factors import FactorModel
from dynamic_intermodal.uncertainty.scenarios import ScenarioGenerator, ScenarioSet, sample_truth
from dynamic_intermodal.update.observations import Observer
from dynamic_intermodal.update.updater import ScenarioUpdater


@dataclass
class EpochLog:
    time: float
    observations: dict[str, int] = field(default_factory=dict)
    ess: float = 0.0
    n_scenarios: int = 0
    rejuvenated: bool = False
    factor_mean: dict[str, float] = field(default_factory=dict)
    expected_cost: float = 0.0
    cvar_delay_cost: float = 0.0
    risky_orders: dict[int, list[str]] = field(default_factory=dict)
    conflicts: list[dict] = field(default_factory=list)
    scope_direct: list[int] = field(default_factory=list)
    scope_indirect: list[int] = field(default_factory=list)
    scope_reasons: dict[int, list[str]] = field(default_factory=dict)
    triggered: bool = False
    milp: list[dict] = field(default_factory=list)
    recheck_rounds: int = 0
    proposal_accepted: bool = False
    objective_before: float | None = None
    objective_after: float | None = None
    validation: dict = field(default_factory=dict)
    requests: list[dict] = field(default_factory=list)


@dataclass
class RunResult:
    method: str
    seed: int
    metrics: dict[str, float]
    epochs: list[EpochLog]
    initial_plan: TransportPlan
    final_plan: TransportPlan
    truth_outcome: ScenarioOutcome
    network: ServiceNetwork


def conflict_summary(c: ConflictRecord, network: ServiceNetwork) -> dict:
    return {
        "resource": c.resource,
        "node": network.node_name(c.node),
        "slot": [c.slot_start, c.slot_end],
        "service": None if c.service_id is None else network.services[c.service_id].name,
        "orders": sorted(c.order_ids),
        "probability": round(c.probability, 3),
        "expected_overflow": round(c.expected_overflow, 2),
        "max_load": round(c.max_load, 1),
        "capacity": c.capacity,
    }


class RollingHorizonController:
    def __init__(
        self,
        config: Config,
        method: MethodConfig,
        seed: int,
        problem: ProblemInstance | None = None,
        verbose: bool = False,
    ):
        self.config = config
        self.method = method
        self.seed = seed
        self.verbose = verbose
        problem = problem or load_problem(config)
        self.network = build_network(problem, config)
        net = self.network

        self.registry = StakeholderRegistry(net, config.stakeholders)
        self.engine = PropagationEngine(net, config.cost, config.network, self.registry.allow_truck)
        self.detector = ConflictDetector(net, config.risk.conflict_min_prob)
        self.evaluator = PlanEvaluator(net, self.engine, self.detector, config.optimization, config.risk)
        self.rechecker = RiskRechecker(config.risk)
        self.scope_identifier = ScopeIdentifier(config.risk, method.scope_mode)
        self.candidates = CandidateGenerator(net, self.registry, config.optimization, config.data.max_transfers + 1)
        self.milp = ScenarioMILP(net, self.registry, self.engine, config.optimization, config.risk)

        # Hidden truth (always drawn from the joint model) and the planner's own model.
        self.truth_model = FactorModel.build(net, config.uncertainty, "joint")
        self.truth = sample_truth(self.truth_model, net, np.random.default_rng(seed))
        self.truth_view = self.truth.view(0)
        if method.uncertainty_mode == "joint":
            self.planner_model = self.truth_model
        else:
            self.planner_model = FactorModel.build(net, config.uncertainty, method.uncertainty_mode)
        planner_rng = np.random.default_rng(seed + 3)
        self.generator = ScenarioGenerator(self.planner_model, net, config.uncertainty, planner_rng)
        self.updater = ScenarioUpdater(self.planner_model, config.uncertainty, planner_rng, method.update_enabled)
        self.observer = Observer(self.truth, net, config.uncertainty, np.random.default_rng(seed + 1))
        self.confirmation = ConfirmationSimulator(
            net, self.registry, self.truth, self.observer, config.uncertainty, np.random.default_rng(seed + 2)
        )

        self.forbidden: set[tuple[int, int]] = set()
        self.realized_misses: set[tuple] = set()
        self.change_counts: dict[int, int] = {oid: 0 for oid in net.orders}
        self.stats = {
            "requests": 0, "accepted": 0, "rejected": 0, "changed_legs": 0, "change_cost": 0.0,
            "milp_solves": 0, "milp_time": 0.0, "triggered_epochs": 0, "recheck_expansions": 0,
        }

    # ================================================================ run

    def run(self) -> RunResult:
        started = time.perf_counter()
        sim = self.config.simulation
        scenarios = self.generator.generate()
        scenarios, _ = self.updater.update(scenarios, self.observer.collect(0.0), 0.0)
        plan = self._initial_plan(scenarios)
        initial_plan = plan.copy()

        epochs: list[EpochLog] = []
        t = sim.epoch_hours
        while t <= sim.end_time + 1e-9:
            truth_outcome = self.engine.simulate(plan, self.truth_view)
            self._record_misses(truth_outcome, t)
            plan, states = sync_with_execution(plan, truth_outcome, self.network, self.registry, t)
            if all(s.status == "delivered" for s in states.values()):
                break
            scenarios, report = self.updater.update(scenarios, self.observer.collect(t), t)
            log = EpochLog(
                time=t,
                observations=report.n_observations,
                ess=report.ess_after,
                n_scenarios=len(scenarios),
                rejuvenated=report.rejuvenated,
                factor_mean=report.factor_mean,
            )
            if self.method.replan_enabled:
                plan, scenarios = self._replan(plan, states, scenarios, t, log)
            epochs.append(log)
            if self.verbose:
                self._print_epoch(log)
            t += sim.epoch_hours

        final = self.engine.simulate(plan, self.truth_view)
        self._record_misses(final, math.inf)
        metrics = self._metrics(final, time.perf_counter() - started)
        return RunResult(self.method.name, self.seed, metrics, epochs, initial_plan, plan, final, self.network)

    # ====================================================== plan generation

    def _initial_plan(self, scenarios: ScenarioSet) -> TransportPlan:
        """Collaborative plan generation at t=0 within the services offered by the stakeholders."""
        net = self.network
        releases = {oid: self._min_release(scenarios, oid) for oid in net.orders}
        plan = TransportPlan()
        pending = set(net.orders)
        for _ in range(self.config.optimization.max_confirm_rounds + 1):
            cands = {
                oid: self.candidates.generate(oid, None, 0, releases[oid], 0.0, self.forbidden)
                for oid in sorted(pending)
            }
            empty = [oid for oid, c in cands.items() if not c]
            if empty:
                raise RuntimeError(f"No feasible service chain for orders {empty}")
            result = self._solve(plan, cands, pending, scenarios, 0.0, None, enforce_limits=False)
            requests = [
                ChangeRequest.from_candidate(result.selection[oid], (), net, 0.0, "initial plan")
                for oid in sorted(pending)
            ]
            responses = self.confirmation.process(requests, plan, 0.0, self._declared_volume(scenarios), initial=True)
            pending = set()
            for res in responses:
                if not res.accepted:
                    pending.add(res.request.order_id)
                    self.forbidden |= {(res.request.order_id, sid) for sid in res.rejected_services}
            if not pending:
                break
        for oid in pending:  # fall back to the best candidate without confirmation feedback
            plan.set(oid, self.candidates.generate(oid, None, 0, releases[oid], 0.0)[0].legs)
        return plan

    def _solve(
        self,
        plan: TransportPlan,
        cands: dict[int, list[CandidatePlan]],
        scope: set[int],
        scenarios: ScenarioSet,
        t: float,
        evaluation: PlanEvaluation | None,
        boost: dict | None = None,
        enforce_limits: bool = True,
    ):
        declared = self._declared_volume(scenarios)
        if evaluation is None:
            S = len(scenarios)
            background = Background([{} for _ in range(S)], [{} for _ in range(S)], np.zeros(S), {})
            for oid in plan:
                for leg in plan[oid].legs:
                    background.booked[leg.service_id] = background.booked.get(leg.service_id, 0.0) + declared[oid]
        else:
            background = self.milp.background(plan, scope, evaluation.outcomes, t, declared)
        evals = self.milp.evaluate_candidates(cands, scenarios, t, background)
        result = self.milp.solve(
            cands, evals, background, scenarios, t, declared, self._known_capacity(scenarios),
            penalty_boost=boost, enforce_change_limits=enforce_limits,
        )
        self.stats["milp_solves"] += 1
        self.stats["milp_time"] += result.runtime
        if result.objective is None:
            raise RuntimeError(f"MILP failed at t={t}: {result.status}")
        return result

    # ========================================================= re-planning

    def _replan(
        self,
        plan: TransportPlan,
        states: dict[int, OrderExecState],
        scenarios: ScenarioSet,
        t: float,
        log: EpochLog,
    ) -> tuple[TransportPlan, ScenarioSet]:
        cfg = self.config
        changeable = {
            oid for oid, st in states.items()
            if st.changeable
            and st.locked < len(plan[oid].legs)
            and self.change_counts[oid] < self.registry.shipper_of(oid).max_route_changes
        }
        evaluation = self.evaluator.evaluate(plan, scenarios, t)
        log.expected_cost = evaluation.risk.expected_cost
        log.cvar_delay_cost = evaluation.components["cvar_delay_cost"]
        log.objective_before = evaluation.objective
        log.risky_orders = evaluation.risk.risky_orders(cfg.risk, changeable)
        log.conflicts = [conflict_summary(c, self.network) for c in evaluation.conflicts]

        scope = self.scope_identifier.identify(evaluation.risk, evaluation.conflicts, changeable)
        log.triggered = scope.triggered and bool(scope.orders)
        self._log_scope(log, scope)
        if not log.triggered:
            return plan, scenarios
        self.stats["triggered_epochs"] += 1

        for round_index in range(cfg.optimization.max_confirm_rounds):
            proposal, selection, new_eval = self._optimise_scope(plan, states, scenarios, t, evaluation, scope, changeable, log)
            if round_index == 0:
                self._log_scope(log, scope)  # including re-check expansions
            if proposal is None:
                break

            requests = [
                ChangeRequest.from_candidate(
                    cand, plan[oid].legs, self.network, t, "; ".join(scope.reasons.get(oid, []))
                )
                for oid, cand in sorted(selection.items())
                if not cand.is_current
            ]
            if not requests:
                break
            requests.sort(key=lambda r: -len(scope.reasons.get(r.order_id, [])))
            results = self.confirmation.process(requests, plan, t, self._declared_volume(scenarios))
            rejected: set[int] = set()
            new_obs = []
            for res in results:
                req = res.request
                self.stats["requests"] += 1
                log.requests.append({
                    "order": req.order_id,
                    "old": [self.network.services[l.service_id].name for l in req.old_legs],
                    "new": [self.network.services[l.service_id].name for l in req.new_legs],
                    "accepted": res.accepted,
                    "responses": [(r.stakeholder_id, r.accepted, r.reason) for r in res.responses],
                    "rationale": req.rationale,
                    "round": round_index,
                })
                if res.accepted:
                    self.stats["accepted"] += 1
                    self.stats["changed_legs"] += sum(
                        1 for a, b in zip(req.old_legs, req.new_legs) if a.service_id != b.service_id
                    ) + abs(len(req.old_legs) - len(req.new_legs))
                    self.stats["change_cost"] += req.change_cost
                    self.change_counts[req.order_id] += 1
                else:
                    self.stats["rejected"] += 1
                    rejected.add(req.order_id)
                    self.forbidden |= {(req.order_id, sid) for sid in res.rejected_services}
                    for r in res.responses:
                        if not r.accepted and r.stakeholder_id.startswith("terminal-"):
                            node = self.network.name_to_id[r.stakeholder_id.split("-", 1)[1]]
                            self.forbidden |= {
                                (req.order_id, sid) for sid in req.new_services
                                if self.network.services[sid].destination == node
                            }
                new_obs.extend(res.observations)
            log.proposal_accepted = log.proposal_accepted or any(r.accepted for r in results)
            if new_obs:
                scenarios, _ = self.updater.update(scenarios, new_obs, t)
            if not rejected:
                break
            # Next round: only the orders whose change was rejected, with the rejected options removed.
            evaluation = self.evaluator.evaluate(plan, scenarios, t)
            scope = AdjustmentScope()
            for oid in rejected:
                if oid in changeable and self.change_counts[oid] < self.registry.shipper_of(oid).max_route_changes:
                    scope.add(oid, "rejected in confirmation", direct=True)
            if not scope.orders:
                break
        return plan, scenarios

    def _optimise_scope(self, plan, states, scenarios, t, evaluation, scope, changeable, log):
        cfg = self.config.optimization
        boost: dict[tuple, float] = {}
        proposal = selection = new_eval = None
        for recheck_round in range(cfg.max_recheck_rounds):
            cands = {
                oid: self.candidates.generate(
                    oid, plan[oid], states[oid].locked, self._start_ready(evaluation, plan, states, oid, t),
                    t, self.forbidden, self.change_counts[oid],
                )
                for oid in sorted(scope.orders)
            }
            result = self._solve(plan, cands, set(cands), scenarios, t, evaluation, boost)
            log.milp.append({
                "round": recheck_round,
                "status": result.status,
                "objective": result.objective,
                "keep_objective": result.keep_objective,
                "runtime": round(result.runtime, 3),
                "n_vars": result.n_vars,
                "n_constrs": result.n_constrs,
                "scope": len(cands),
                "candidates": sum(len(c) for c in cands.values()),
                "breakdown": {k: round(v, 1) for k, v in result.breakdown.items()},
            })
            selection = result.selection
            proposal = plan.copy()
            for oid, cand in selection.items():
                proposal.set(oid, cand.legs)
            change_cost = sum(c.change_cost for c in selection.values())
            if not self.method.recheck_enabled:
                if result.keep_objective is not None and result.objective >= result.keep_objective - cfg.improvement_tolerance:
                    return None, None, None
                return proposal, selection, None
            new_eval = self.evaluator.evaluate(proposal, scenarios, t, change_cost)
            log.recheck_rounds = recheck_round + 1
            new_conflicts = self.rechecker.new_conflicts(evaluation.conflicts, new_eval.conflicts, scope.orders)
            worsened = self.rechecker.worsened_orders(evaluation.risk, new_eval.risk, scope.orders, changeable)
            added = False
            for c in new_conflicts:
                boost[c.key] = boost.get(c.key, 1.0) * cfg.conflict_penalty_boost
                for oid in (c.order_ids & changeable) - scope.orders:
                    scope.add(oid, "re-check: new conflict")
                    added = True
            for oid in worsened - scope.orders:
                scope.add(oid, "re-check: risk increased")
                added = True
            if not new_conflicts and not worsened:
                break
            self.stats["recheck_expansions"] += int(added)
        log.objective_after = new_eval.objective
        if new_eval.objective >= evaluation.objective - cfg.improvement_tolerance:
            return None, None, None
        # Out-of-sample re-check on independent posterior scenarios (guards the optimiser's curse).
        validation = self.updater.validation_set(scenarios, cfg.validation_scenarios)
        current_v = self.evaluator.evaluate(plan, validation, t)
        proposal_v = self.evaluator.evaluate(proposal, validation, t, change_cost)
        gain, se = paired_improvement(current_v, proposal_v)
        log.validation = {"gain": round(gain, 1), "se": round(se, 1), "in_sample_gain": round(evaluation.objective - new_eval.objective, 1)}
        if gain <= cfg.improvement_tolerance + cfg.validation_z * se:
            return None, None, None
        return proposal, selection, new_eval

    # ============================================================ helpers

    @staticmethod
    def _log_scope(log: EpochLog, scope: AdjustmentScope) -> None:
        log.scope_direct = sorted(scope.direct)
        log.scope_indirect = sorted(scope.indirect)
        log.scope_reasons = {oid: list(r) for oid, r in scope.reasons.items()}

    def _record_misses(self, outcome: ScenarioOutcome, t: float) -> None:
        """Missed connections that actually happened (they disappear from later replays once the
        automatic rebooking has been written into the plan)."""
        for rec in outcome.records:
            if rec.time <= t:
                self.realized_misses.add((rec.order_id, rec.planned_service, rec.reason))

    def _start_ready(self, evaluation, plan, states, oid, t) -> float:
        locked = states[oid].locked
        readies = []
        for outcome in evaluation.outcomes:
            out = outcome.orders[oid]
            if locked == 0:
                readies.append(out.release)
            elif len(out.legs) >= locked:
                readies.append(out.legs[locked - 1].ready)
        return min(readies) if readies else t

    def _min_release(self, scenarios: ScenarioSet, oid: int) -> float:
        order = self.network.orders[oid]
        j = scenarios.model.index(("release", oid))
        return order.release + float(np.min(scenarios.X[:, j]))

    def _declared_volume(self, scenarios: ScenarioSet) -> dict[int, float]:
        result = {}
        for oid, order in self.network.orders.items():
            j = scenarios.model.index(("demand", oid))
            result[oid] = order.volume * scenarios.realized.get(j, 1.0)
        return result

    def _known_capacity(self, scenarios: ScenarioSet) -> dict[int, float]:
        result = {}
        for s in self.network.scheduled_services:
            j = scenarios.model.index(("capacity", s.id))
            result[s.id] = s.capacity * scenarios.realized.get(j, 1.0)
        return result

    def _metrics(self, final: ScenarioOutcome, runtime: float) -> dict[str, float]:
        net = self.network
        orders = final.orders.values()
        n = len(final.orders)
        yard_overflow = sum(
            max(0.0, load - net.yard_capacity[node]) * net.config.slot_hours
            for (node, _), (load, _) in yard_profile(final, net).items()
        )
        legs = [leg for o in orders for leg in o.legs]
        truck_legs = sum(1 for leg in legs if not net.services[leg.service].scheduled)
        total_cost = sum(o.cost.total for o in orders)
        return {
            "total_cost": total_cost,
            "operating_cost": sum(o.cost.operating for o in orders),
            "delay_cost": sum(o.cost.delay for o in orders),
            "total_delay_h": sum(o.delay for o in orders),
            "on_time_rate": sum(1 for o in orders if not o.late) / n,
            "late_orders": sum(1 for o in orders if o.late),
            "missed_connections": len(self.realized_misses),
            "stranded": sum(1 for o in orders if o.stranded),
            "yard_overflow_teu_h": yard_overflow,
            "bumped_teu": sum(final.service_bumped.values()),
            "truck_leg_share": truck_legs / max(1, len(legs)),
            "change_requests": self.stats["requests"],
            "changes_accepted": self.stats["accepted"],
            "changes_rejected": self.stats["rejected"],
            "changed_legs": self.stats["changed_legs"],
            "change_cost": self.stats["change_cost"],
            "total_with_change": total_cost + self.stats["change_cost"],
            "triggered_epochs": self.stats["triggered_epochs"],
            "recheck_expansions": self.stats["recheck_expansions"],
            "milp_solves": self.stats["milp_solves"],
            "milp_time_s": self.stats["milp_time"],
            "runtime_s": runtime,
        }

    def _print_epoch(self, log: EpochLog) -> None:
        msg = (
            f"t={log.time:6.1f} ess={log.ess:5.1f}/{log.n_scenarios:<2d} E[cost]={log.expected_cost:10.0f} "
            f"CVaR={log.cvar_delay_cost:9.0f} risky={len(log.risky_orders):2d} conflicts={len(log.conflicts):2d}"
        )
        if log.triggered:
            msg += (
                f" | scope {len(log.scope_direct)}+{len(log.scope_indirect)} milp={len(log.milp)} "
                f"recheck={log.recheck_rounds} requests={len(log.requests)} "
                f"accepted={sum(1 for r in log.requests if r['accepted'])}"
            )
        print(msg)
