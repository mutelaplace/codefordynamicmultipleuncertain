"""Method presets: the full method, baselines and ablations (one switch each)."""

from __future__ import annotations

from dynamic_intermodal.config import MethodConfig

METHODS: dict[str, MethodConfig] = {
    # B0: robust initial plan, no dynamic adjustment (operations only rebook automatically).
    "static": MethodConfig("static", "joint", True, "conflict", True, replan_enabled=False),
    # B1: deterministic event-driven re-planning on one nominal scenario, directly affected orders only.
    "deterministic": MethodConfig("deterministic", "nominal", False, "direct", False, True),
    # Full method (technical points 1-6).
    "full": MethodConfig("full", "joint", True, "conflict", True, True),
    # Ablations.
    "no_joint": MethodConfig("no_joint", "independent", True, "conflict", True, True),
    "no_update": MethodConfig("no_update", "joint", False, "conflict", True, True),
    "direct_scope": MethodConfig("direct_scope", "joint", True, "direct", True, True),
    "no_recheck": MethodConfig("no_recheck", "joint", True, "conflict", False, True),
}


def get_method(name: str) -> MethodConfig:
    try:
        return METHODS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown method {name!r}; choose from {sorted(METHODS)}") from exc
