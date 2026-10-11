"""Scenario-based MILP for joint (collaborative) plan selection, solved exactly with Gurobi.

For every order r in the adjustment scope and candidate plan p in P_r, risk propagation
pre-computes in each scenario s the realised operating cost c_rps, delay cost d_rps, late flag,
service usage a_rps^k (after automatic rebooking) and yard occupancy o_rps^(n,tau).

    min  sum_s w_s [ sum_rp (c_rps + d_rps) x_rp + piK * sum_k ovK_sk + piY * sum_(n,tau) ovY_s,n,tau ]
         + lambda * ( eta + 1/(1-alpha) sum_s w_s u_s )                      (CVaR of delay cost)
         + piL * sum_r slL_r + sum_rp g_rp x_rp                              (risk requirement, change cost)
    s.t. sum_p x_rp = 1                                                       for all r
         bgK_sk + sum_rp a_rps^k x_rp <= cap_k * kappa_sk + ovK_sk            for all s, k   (service capacity)
         bgY_s,n,tau + sum_rp o_rps^(n,tau) x_rp <= Y_n + ovY_s,n,tau         for all s, n, tau (yard)
         u_s >= bgD_s + sum_rp d_rps x_rp - eta,  u_s >= 0                     for all s
         sum_p P(late)_rp x_rp - beta <= slL_r                                for all r
         sum_rp [c in carriers(r,p)] x_rp <= M_c                              for all carriers c (change permission)
         sum_rp q_r [k in p] x_rp + B_k <= max(capKnown_k, B_k + current_k)  for all k (booking capacity)
         x binary
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from dynamic_intermodal.config import OptimizationConfig, RiskConfig
from dynamic_intermodal.conflict.detector import order_slot_occupancy
from dynamic_intermodal.coordination.candidates import CandidatePlan
from dynamic_intermodal.coordination.stakeholders import StakeholderRegistry
from dynamic_intermodal.network import ServiceNetwork
from dynamic_intermodal.plan import OrderPlan, TransportPlan
from dynamic_intermodal.risk.propagation import PropagationEngine, ScenarioOutcome
from dynamic_intermodal.uncertainty.scenarios import ScenarioSet

try:  # pragma: no cover - depends on the local installation
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None

_ENV = None


def gurobi_env():
    global _ENV
    if gp is None:
        raise RuntimeError("gurobipy is not available")
    if _ENV is None:
        _ENV = gp.Env(empty=True)
        _ENV.setParam("OutputFlag", 0)
        _ENV.start()
    return _ENV


@dataclass
class CandidateEvaluation:
    cost: np.ndarray  # (S,) operating cost
    delay_cost: np.ndarray  # (S,)
    delay: np.ndarray  # (S,) hours
    late: np.ndarray  # (S,) 0/1
    missed: np.ndarray  # (S,)
    usage: list[dict[int, float]]  # per scenario: future scheduled service -> TEU
    occupancy: list[dict[tuple[int, int], float]]  # per scenario: (node, slot) -> TEU


@dataclass
class Background:
    usage: list[dict[int, float]]
    occupancy: list[dict[tuple[int, int], float]]
    delay_cost: np.ndarray
    booked: dict[int, float]


@dataclass
class MILPResult:
    status: str
    objective: float | None
    selection: dict[int, CandidatePlan]
    runtime: float
    mip_gap: float | None
    n_vars: int
    n_constrs: int
    breakdown: dict[str, float] = field(default_factory=dict)
    keep_objective: float | None = None


class ScenarioMILP:
    def __init__(
        self,
        network: ServiceNetwork,
        registry: StakeholderRegistry,
        engine: PropagationEngine,
        opt_config: OptimizationConfig,
        risk_config: RiskConfig,
    ):
        self.network = network
        self.registry = registry
        self.engine = engine
        self.config = opt_config
        self.risk = risk_config

    # --------------------------------------------------------- pre-compute

    def evaluate_candidates(
        self,
        candidates: dict[int, list[CandidatePlan]],
        scenarios: ScenarioSet,
        t: float,
        background: "Background | None" = None,
    ) -> dict[tuple[int, int], CandidateEvaluation]:
        """Scenario-wise propagation of each candidate alone; the yard load of the orders outside
        the scope (background) is added so that congestion they cause is felt by the candidate."""
        views = scenarios.views()
        yard_bg = background.occupancy if background is not None else [None] * len(views)
        evals: dict[tuple[int, int], CandidateEvaluation] = {}
        for oid, cands in candidates.items():
            for i, cand in enumerate(cands):
                single = TransportPlan({oid: OrderPlan(oid, cand.legs)})
                outs = [
                    self.engine.simulate(single, view, yard_background=bg).orders[oid]
                    for view, bg in zip(views, yard_bg)
                ]
                evals[(oid, i)] = CandidateEvaluation(
                    cost=np.array([o.cost.operating for o in outs]),
                    delay_cost=np.array([o.cost.delay for o in outs]),
                    delay=np.array([o.delay for o in outs]),
                    late=np.array([1.0 if o.late else 0.0 for o in outs]),
                    missed=np.array([float(o.missed) for o in outs]),
                    usage=[self._future_usage(o, t) for o in outs],
                    occupancy=[order_slot_occupancy(o, self.network, t) for o in outs],
                )
        return evals

    def _future_usage(self, outcome, t: float) -> dict[int, float]:
        usage: dict[int, float] = {}
        for leg in outcome.legs:
            service = self.network.services[leg.service]
            if service.scheduled and service.cutoff > t:
                usage[leg.service] = usage.get(leg.service, 0.0) + outcome.volume
        return usage

    def background(
        self,
        plan: TransportPlan,
        scope: set[int],
        outcomes: list[ScenarioOutcome],
        t: float,
        declared_volume: dict[int, float],
    ) -> Background:
        others = [oid for oid in plan if oid not in scope]
        usage, occupancy, delay_cost = [], [], []
        for outcome in outcomes:
            u: dict[int, float] = {}
            occ: dict[tuple[int, int], float] = {}
            d = 0.0
            for oid in others:
                o = outcome.orders[oid]
                for k, v in self._future_usage(o, t).items():
                    u[k] = u.get(k, 0.0) + v
                for key, v in order_slot_occupancy(o, self.network, t).items():
                    occ[key] = occ.get(key, 0.0) + v
                d += o.cost.delay
            usage.append(u)
            occupancy.append(occ)
            delay_cost.append(d)
        booked: dict[int, float] = {}
        for oid in others:
            for leg in plan[oid].legs:
                service = self.network.services[leg.service_id]
                if service.scheduled and service.cutoff > t:
                    booked[leg.service_id] = booked.get(leg.service_id, 0.0) + declared_volume[oid]
        return Background(usage, occupancy, np.asarray(delay_cost), booked)

    # ---------------------------------------------------------------- solve

    def solve(
        self,
        candidates: dict[int, list[CandidatePlan]],
        evals: dict[tuple[int, int], CandidateEvaluation],
        background: Background,
        scenarios: ScenarioSet,
        t: float,
        declared_volume: dict[int, float],
        known_capacity: dict[int, float],
        penalty_boost: dict[tuple, float] | None = None,
        enforce_change_limits: bool = True,
    ) -> MILPResult:
        started = time.perf_counter()
        boost = penalty_boost or {}
        cfg = self.config
        w = scenarios.weights
        S = len(w)
        alpha = self.risk.cvar_alpha
        beta = self.risk.late_prob_threshold

        m = gp.Model(env=gurobi_env())
        m.Params.OutputFlag = 1 if cfg.output_flag else 0
        m.Params.TimeLimit = cfg.time_limit
        m.Params.MIPGap = cfg.mip_gap
        if cfg.threads:
            m.Params.Threads = cfg.threads

        x = {key: m.addVar(vtype=GRB.BINARY, name=f"x_{key[0]}_{key[1]}") for key in evals}
        exp_cost = gp.LinExpr()
        change = gp.LinExpr()
        for (oid, i), ev in evals.items():
            exp_cost.add(x[(oid, i)], float(w @ (ev.cost + ev.delay_cost)))
            change.add(x[(oid, i)], candidates[oid][i].change_cost)
        for oid, cands in candidates.items():
            m.addConstr(gp.quicksum(x[(oid, i)] for i in range(len(cands))) == 1, name=f"assign_{oid}")

        # Scenario-wise service capacity and yard capacity (soft, penalised overflow).
        overflow = gp.LinExpr()
        for s in range(S):
            svc_terms: dict[int, gp.LinExpr] = {}
            yard_terms: dict[tuple[int, int], gp.LinExpr] = {}
            for key, ev in evals.items():
                for k, v in ev.usage[s].items():
                    svc_terms.setdefault(k, gp.LinExpr()).add(x[key], v)
                for nk, v in ev.occupancy[s].items():
                    yard_terms.setdefault(nk, gp.LinExpr()).add(x[key], v)
            view = scenarios.view(s)
            for k, expr in svc_terms.items():
                cap = self.network.services[k].capacity * view.capacity_factor(k)
                ov = m.addVar(lb=0.0)
                m.addConstr(background.usage[s].get(k, 0.0) + expr <= cap + ov)
                factor = boost.get(("service", self.network.services[k].origin, self.network.slot_of(self.network.services[k].cutoff), k), 1.0)
                overflow.add(ov, w[s] * cfg.service_overflow_penalty * factor)
            for (node, slot), expr in yard_terms.items():
                ov = m.addVar(lb=0.0)
                m.addConstr(background.occupancy[s].get((node, slot), 0.0) + expr <= self.network.yard_capacity[node] + ov)
                factor = boost.get(("yard", node, slot, None), 1.0)
                overflow.add(ov, w[s] * cfg.yard_overflow_penalty * factor)

        # CVaR of the delay cost (Rockafellar-Uryasev linearisation).
        eta = m.addVar(lb=-GRB.INFINITY, name="eta")
        u = m.addVars(S, lb=0.0, name="u")
        for s in range(S):
            expr = gp.LinExpr()
            for key, ev in evals.items():
                expr.add(x[key], float(ev.delay_cost[s]))
            m.addConstr(u[s] >= float(background.delay_cost[s]) + expr - eta)
        cvar = eta + (1.0 / (1.0 - alpha)) * gp.quicksum(w[s] * u[s] for s in range(S))

        # Operational risk requirement P(late) <= beta (soft).
        late_pen = gp.LinExpr()
        for oid, cands in candidates.items():
            probs = [float(w @ evals[(oid, i)].late) for i in range(len(cands))]
            if max(probs) <= beta:
                continue
            sl = m.addVar(lb=0.0)
            m.addConstr(gp.quicksum(probs[i] * x[(oid, i)] for i in range(len(cands))) - beta <= sl)
            late_pen.add(sl, cfg.late_prob_penalty)

        # Change permissions of carriers (number of booking changes they accept per epoch).
        if enforce_change_limits:
            per_carrier: dict[str, list] = {}
            for (oid, i) in evals:
                for c in candidates[oid][i].carriers:
                    per_carrier.setdefault(c, []).append(x[(oid, i)])
            for c, terms in per_carrier.items():
                policy = self.registry.stakeholders[c].policy
                m.addConstr(gp.quicksum(terms) <= policy.max_changes_per_epoch, name=f"perm_{c}")

        # Booking capacity of scheduled services (declared volumes, known capacity).
        book_terms: dict[int, gp.LinExpr] = {}
        current_book: dict[int, float] = {}
        for (oid, i) in evals:
            cand = candidates[oid][i]
            for sid in set(cand.service_ids):
                service = self.network.services[sid]
                if service.scheduled and service.cutoff > t:
                    book_terms.setdefault(sid, gp.LinExpr()).add(x[(oid, i)], declared_volume[oid])
                    if cand.is_current:
                        current_book[sid] = current_book.get(sid, 0.0) + declared_volume[oid]
        for sid, expr in book_terms.items():
            base = background.booked.get(sid, 0.0)
            rhs = max(known_capacity.get(sid, self.network.services[sid].capacity), base + current_book.get(sid, 0.0))
            m.addConstr(base + expr <= rhs, name=f"book_{sid}")

        lam = cfg.cvar_weight
        m.setObjective(exp_cost + overflow + lam * cvar + late_pen + change, GRB.MINIMIZE)
        m.optimize()

        status = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit", GRB.INFEASIBLE: "infeasible"}.get(
            m.Status, f"status_{m.Status}"
        )
        selection: dict[int, CandidatePlan] = {}
        breakdown: dict[str, float] = {}
        objective = None
        gap = None
        if m.SolCount > 0:
            objective = float(m.ObjVal)
            gap = float(m.MIPGap) if m.IsMIP else 0.0
            for oid, cands in candidates.items():
                chosen = max(range(len(cands)), key=lambda i: x[(oid, i)].X)
                selection[oid] = cands[chosen]
            breakdown = {
                "expected_cost": exp_cost.getValue(),
                "overflow_penalty": overflow.getValue(),
                "cvar_delay_cost": cvar.getValue(),
                "late_penalty": late_pen.getValue(),
                "change_cost": change.getValue(),
            }
            # Objective of keeping all current plans, for the improvement estimate.
            keep = {oid: next((i for i, c in enumerate(cands) if c.is_current), None) for oid, cands in candidates.items()}
            if all(v is not None for v in keep.values()):
                for (oid, i), var in x.items():
                    var.LB = var.UB = 1.0 if keep[oid] == i else 0.0
                m.optimize()
                keep_obj = float(m.ObjVal) if m.SolCount > 0 else None
            else:
                keep_obj = None
        else:
            keep_obj = None
        return MILPResult(
            status=status,
            objective=objective,
            selection=selection,
            runtime=time.perf_counter() - started,
            mip_gap=gap,
            n_vars=m.NumVars,
            n_constrs=m.NumConstrs,
            breakdown=breakdown,
            keep_objective=keep_obj,
        )
