"""Configuration dataclasses.

DataConfig / CostConfig are copied from ``intermodal_refactor`` (intermodal/config/settings.py)
so that cost semantics stay comparable with the original ALNS / Gurobi planning code.
The remaining configs belong to the dynamic multi-source-uncertainty method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data_input" / "2trans_instances"


# ---------------------------------------------------------------------------
# Copied from intermodal_refactor
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    data_dir: Path | str = DEFAULT_DATA_DIR
    main_file: str = "Intermodal_EGS_data.xlsx"
    distance_file: str = "Distance_EGS.xlsx"
    fixed_file: str = "Fixed_vehicles.xlsx"
    agent_file: str = "route_agent.xlsx"
    num_requests: int = 20
    num_transshipment_nodes: int | None = None
    max_transfers: int = 2
    augment_to: int | None = None  # resample extra orders up to this number (denser instance)
    augment_seed: int = 0

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)


@dataclass
class CostConfig:
    load_cost: dict[int, float] = field(default_factory=lambda: {1: 25.0, 2: 25.0, 3: 3.0})
    storage_cost: float = 1.0
    waiting_cost: float = 1.0
    carbon_emission: float = 125.0
    loading_time: float = 1.0
    delay_cost: dict[str, float] = field(
        default_factory=lambda: {"severe": 100.0, "moderate": 70.0, "minor": 50.0}
    )
    delay_threshold_severe: float = 30.0
    delay_threshold_moderate: float = 54.0
    unserved_penalty: float = 10000.0


# ---------------------------------------------------------------------------
# Service network / node resources
# ---------------------------------------------------------------------------


@dataclass
class NetworkConfig:
    # Yard (storage) capacity of each terminal in TEU, keyed by terminal name.
    yard_capacity: dict[str, float] = field(
        default_factory=lambda: {
            "Delta": 2000.0,
            "Venlo": 70.0,
            "Duisburg": 55.0,
            "Neuss": 40.0,
            "Dortmund": 400.0,
            "Nuremberg": 400.0,
        }
    )
    default_yard_capacity: float = 200.0
    truck_handling_time: float = 1.0  # loading / unloading time of a truck leg (h)
    rebook_max_wait: float = 24.0  # wait at most this long for the next scheduled service before trucking
    stranded_delay: float = 72.0  # extra delay (h) charged when no onward service exists
    yard_congestion_hours: float = 8.0  # extra handling time per 100 % yard overflow at arrival
    slot_hours: float = 4.0  # time-slot length for node-slot-order conflict checks
    block_hours: float = 24.0  # time block of time-varying uncertainties (handling, road traffic)
    horizon: float = 240.0
    max_lateness_prune: float = 72.0  # candidate enumeration prunes chains later than due + this


# ---------------------------------------------------------------------------
# Multi-source joint uncertainty
# ---------------------------------------------------------------------------


@dataclass
class UncertaintyConfig:
    # Global (common external) factors and their correlation.
    global_factors: tuple[str, ...] = (
        "weather",
        "rail_congestion",
        "port_congestion",
        "road_congestion",
        "demand_shock",
    )
    factor_correlation: dict[tuple[str, str], float] = field(
        default_factory=lambda: {
            ("weather", "port_congestion"): 0.3,
            ("weather", "road_congestion"): 0.3,
            ("demand_shock", "port_congestion"): 0.4,
            ("rail_congestion", "port_congestion"): 0.2,
        }
    )
    # Factor loadings per uncertainty source: {global factor: loading}; group loading is separate.
    loadings: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "rail_delay": {"rail_congestion": 0.55, "weather": 0.2},
            "barge_delay": {"weather": 0.6, "port_congestion": 0.25},
            "handling": {"port_congestion": 0.55, "weather": 0.15, "demand_shock": 0.15},
            "truck_time": {"road_congestion": 0.6, "weather": 0.2},
            "rail_capacity": {"rail_congestion": 0.45},
            "barge_capacity": {"weather": 0.55},
            "demand": {"demand_shock": 0.65},
            "release_delay": {"port_congestion": 0.5, "demand_shock": 0.2},
        }
    )
    group_loadings: dict[str, float] = field(
        default_factory=lambda: {
            "rail_delay": 0.4,
            "barge_delay": 0.35,
            "handling": 0.45,
            "truck_time": 0.35,
            "rail_capacity": 0.3,
            "barge_capacity": 0.3,
            "demand": 0.0,
            "release_delay": 0.0,
        }
    )
    # Marginal distributions.
    rail_delay: tuple[float, float, float, float] = (0.45, 2.5, 0.8, 24.0)  # p0, median, sigma, cap
    barge_delay: tuple[float, float, float, float] = (0.35, 4.0, 0.9, 36.0)
    handling_extra: tuple[float, float, float, float] = (0.35, 1.5, 0.8, 16.0)
    release_delay: tuple[float, float, float, float] = (0.5, 3.0, 0.8, 24.0)
    truck_factor: tuple[float, float, float, float] = (1.05, 0.25, 0.9, 2.5)  # median, sigma, lo, hi
    demand_factor: tuple[float, float, float, float] = (1.0, 0.12, 0.7, 1.5)
    capacity_levels: tuple[tuple[float, float], ...] = ((0.87, 1.0), (0.10, 0.6), (0.03, 0.0))
    # Scenario generation / reduction.
    n_prior_samples: int = 600
    n_scenarios: int = 30
    kmeans_iterations: int = 15
    # Observation model.
    capacity_notice_hours: float = 24.0
    demand_notice_hours: float = 24.0
    indicators: tuple[str, ...] = ("weather", "rail_congestion", "port_congestion", "road_congestion")
    indicator_noise: float = 1.0
    # Scenario update.
    ess_ratio_threshold: float = 0.5
    min_active_scenarios: int = 10
    weight_floor: float = 1e-4
    rejuvenation_samples: int = 600


# ---------------------------------------------------------------------------
# Risk evaluation / conflicts / scope
# ---------------------------------------------------------------------------


@dataclass
class RiskConfig:
    cvar_alpha: float = 0.9
    late_prob_threshold: float = 0.25
    expected_delay_threshold: float = 3.0
    miss_prob_threshold: float = 0.3
    conflict_prob_threshold: float = 0.2  # conflicts above this probability trigger re-planning
    conflict_link_prob: float = 0.1  # conflicts above this probability link orders into the scope
    conflict_min_prob: float = 0.05  # conflicts below this probability are not recorded
    scope_hops: int = 1


# ---------------------------------------------------------------------------
# Stakeholders and permissions
# ---------------------------------------------------------------------------


@dataclass
class CarrierPolicy:
    change_cutoff_hours: float
    accept_prob: float
    response_delay: float
    max_changes_per_epoch: int
    accepts_new_bookings: bool = True


@dataclass
class StakeholderConfig:
    carrier_policies: dict[str, CarrierPolicy] = field(
        default_factory=lambda: {
            "barge": CarrierPolicy(change_cutoff_hours=12.0, accept_prob=0.85, response_delay=1.0, max_changes_per_epoch=4),
            "rail": CarrierPolicy(change_cutoff_hours=6.0, accept_prob=0.85, response_delay=1.0, max_changes_per_epoch=4),
            "truck": CarrierPolicy(change_cutoff_hours=1.0, accept_prob=0.95, response_delay=0.5, max_changes_per_epoch=8),
        }
    )
    terminal_accept_prob: float = 0.95
    # Shippers / logistics companies owning the orders (round-robin assignment).
    shippers: tuple[str, ...] = ("LSP-A", "LSP-B", "LSP-C")
    shipper_allow_truck: dict[str, bool] = field(
        default_factory=lambda: {"LSP-A": True, "LSP-B": True, "LSP-C": False}
    )
    shipper_max_route_changes: dict[str, int] = field(
        default_factory=lambda: {"LSP-A": 3, "LSP-B": 2, "LSP-C": 2}
    )


# ---------------------------------------------------------------------------
# Optimisation (scenario-based MILP)
# ---------------------------------------------------------------------------


@dataclass
class OptimizationConfig:
    max_candidates: int = 12
    cvar_weight: float = 0.5  # lambda: weight of CVaR of delay cost
    change_cost_per_leg: float = 300.0
    change_cost_per_stakeholder: float = 150.0
    service_overflow_penalty: float = 1500.0  # per TEU bumped from a service in a scenario
    yard_overflow_penalty: float = 200.0  # per TEU over yard capacity per slot in a scenario
    late_prob_penalty: float = 1000.0  # per unit of P(late) above the requirement
    conflict_penalty_boost: float = 2.0  # multiplier applied to re-check conflicts
    time_limit: float = 60.0
    mip_gap: float = 1e-4
    threads: int = 0
    output_flag: bool = False
    max_recheck_rounds: int = 3
    max_confirm_rounds: int = 2
    improvement_tolerance: float = 1.0
    validation_scenarios: int = 40  # independent posterior scenarios for the out-of-sample re-check
    validation_z: float = 1.0  # required improvement in standard errors of the paired difference


# ---------------------------------------------------------------------------
# Rolling horizon and method switches
# ---------------------------------------------------------------------------


@dataclass
class SimulationConfig:
    epoch_hours: float = 6.0
    end_time: float = 200.0


@dataclass
class MethodConfig:
    name: str = "full"
    uncertainty_mode: str = "joint"  # joint | independent | nominal
    update_enabled: bool = True
    scope_mode: str = "conflict"  # conflict | direct | all
    recheck_enabled: bool = True
    replan_enabled: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    stakeholders: StakeholderConfig = field(default_factory=StakeholderConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
