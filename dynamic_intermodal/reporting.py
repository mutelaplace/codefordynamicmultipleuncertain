"""Decision-support output of a run: plan generation, operation monitoring, risk warnings,
node-slot-order conflicts, adjustment suggestions with stakeholder responses, final outcome."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from dynamic_intermodal.simulation.rolling import RunResult


def _chain(result: RunResult, legs) -> list[str]:
    return [result.network.services[leg.service_id].name for leg in legs]


def run_report(result: RunResult) -> dict:
    net = result.network
    plan_generation = []
    for oid, order_plan in result.initial_plan.items():
        order = net.orders[oid]
        services = [net.services[leg.service_id] for leg in order_plan.legs]
        plan_generation.append({
            "order": oid,
            "shipper": order.shipper_id,
            "od": [net.node_name(order.origin), net.node_name(order.destination)],
            "release": order.release,
            "due": order.due,
            "volume": order.volume,
            "services": [s.name for s in services],
            "modes": [s.mode for s in services],
            "carriers": [s.carrier_id for s in services],
            "transfer_nodes": [net.node_name(s.destination) for s in services[:-1]],
        })

    monitoring, warnings, suggestions = [], [], []
    for log in result.epochs:
        monitoring.append({
            "time": log.time,
            "observations": log.observations,
            "scenarios": log.n_scenarios,
            "ess": round(log.ess, 1),
            "factor_posterior_mean": {k: round(v, 2) for k, v in log.factor_mean.items()},
            "expected_cost": round(log.expected_cost, 1),
            "cvar_delay_cost": round(log.cvar_delay_cost, 1),
        })
        if log.risky_orders or log.conflicts:
            warnings.append({
                "time": log.time,
                "risky_orders": {str(k): v for k, v in log.risky_orders.items()},
                "conflicts": log.conflicts,
                "scope_direct": log.scope_direct,
                "scope_indirect": log.scope_indirect,
            })
        if log.requests or log.milp:
            suggestions.append({
                "time": log.time,
                "scope_reasons": {str(k): v for k, v in log.scope_reasons.items()},
                "milp": log.milp,
                "objective_before": log.objective_before,
                "objective_after": log.objective_after,
                "validation": log.validation,
                "requests": log.requests,
            })

    outcomes = []
    for oid, out in sorted(result.truth_outcome.orders.items()):
        outcomes.append({
            "order": oid,
            "services": [net.services[leg.service].name for leg in out.legs],
            "final_arrival": round(out.final_arrival, 2),
            "due": net.orders[oid].due,
            "delay_h": round(out.delay, 2),
            "cost": {k: round(v, 1) for k, v in asdict(out.cost).items()},
            "stranded": out.stranded,
        })

    return {
        "method": result.method,
        "seed": result.seed,
        "metrics": {k: round(v, 3) for k, v in result.metrics.items()},
        "plan_generation": plan_generation,
        "monitoring": monitoring,
        "warnings": warnings,
        "adjustment_suggestions": suggestions,
        "final_plan": {str(oid): _chain(result, p.legs) for oid, p in result.final_plan.items()},
        "realised_outcome": outcomes,
    }


def write_report(result: RunResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = run_report(result)
    path = out_dir / f"report_{result.method}_seed{result.seed}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))

    m = result.metrics
    lines = [
        f"# 运行报告：{result.method}（seed {result.seed}）",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 实际总成本（含变更代价） | {m['total_with_change']:.0f} |",
        f"| 运营成本 / 延误成本 / 变更代价 | {m['operating_cost']:.0f} / {m['delay_cost']:.0f} / {m['change_cost']:.0f} |",
        f"| 总延误 (h) / 准时率 | {m['total_delay_h']:.1f} / {m['on_time_rate']:.2%} |",
        f"| 错失班期次数 | {m['missed_connections']:.0f} |",
        f"| 堆场超载 (TEU·h) | {m['yard_overflow_teu_h']:.1f} |",
        f"| 变更请求 / 确认 / 拒绝 | {m['change_requests']:.0f} / {m['changes_accepted']:.0f} / {m['changes_rejected']:.0f} |",
        f"| 触发调整的时点数 / MILP 求解次数 / MILP 总时间 (s) | {m['triggered_epochs']:.0f} / {m['milp_solves']:.0f} / {m['milp_time_s']:.1f} |",
        "",
        "## 调整建议与主体反馈",
        "",
    ]
    for item in report["adjustment_suggestions"]:
        for req in item["requests"]:
            status = "确认" if req["accepted"] else "拒绝"
            refusals = [f"{sid}: {reason}" for sid, ok, reason in req["responses"] if not ok]
            lines.append(
                f"- t={item['time']:.0f}h 订单 {req['order']}：{' → '.join(req['old'])} ⇒ "
                f"{' → '.join(req['new'])}（{status}）；依据：{req['rationale']}"
                + (f"；拒绝原因：{'; '.join(refusals)}" if refusals else "")
            )
    (out_dir / f"report_{result.method}_seed{result.seed}.md").write_text("\n".join(lines) + "\n")
    return path
