"""Joint representation of multi-source uncertainty: a Gaussian-copula common-factor model.

Every uncertain quantity j (rail/barge arrival delay, terminal handling time, truck travel
time, service capacity, order volume, order release) has a latent value

    z_j = lambda_j^T F + psi_j * eps_j,     F ~ N(0, Sigma_F),  eps_j ~ N(0, 1),  Var(z_j) = 1

and the physical value x_j = Q_j(z_j) through its marginal. F stacks *global* external
factors (weather / water level, rail congestion, port congestion, road congestion, demand
shock) and *group* factors (rail/barge corridor, terminal, road region) that tie variables of
the same corridor or terminal together over time. Shared factors carry the cross-source
dependence, so a scenario drawn from this model is a coherent joint state of all sources.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamic_intermodal.config import UncertaintyConfig
from dynamic_intermodal.network import ServiceNetwork
from dynamic_intermodal.uncertainty.marginals import (
    BoundedLogNormal,
    DiscreteLevels,
    Marginal,
    ZeroInflatedLogNormal,
)

# Patent source categories.
CATEGORY_BY_SOURCE = {
    "rail_delay": "arrival",
    "barge_delay": "arrival",
    "handling": "handling",
    "truck_time": "road",
    "rail_capacity": "capacity",
    "barge_capacity": "capacity",
    "demand": "order",
    "release_delay": "order",
}


@dataclass(frozen=True)
class UncertainVariable:
    index: int
    key: tuple
    source: str
    marginal: Marginal
    group: str | None

    @property
    def category(self) -> str:
        return CATEGORY_BY_SOURCE[self.source]


class FactorModel:
    def __init__(
        self,
        factor_names: list[str],
        factor_cov: np.ndarray,
        variables: list[UncertainVariable],
        loadings: np.ndarray,
        mode: str,
    ):
        self.factor_names = factor_names
        self.factor_cov = factor_cov
        self.variables = variables
        self.L = loadings
        self.mode = mode
        explained = np.einsum("jf,fg,jg->j", loadings, factor_cov, loadings)
        self.psi = np.sqrt(np.clip(1.0 - explained, 1e-6, None))
        self._index = {v.key: v.index for v in variables}
        self._factor_index = {name: i for i, name in enumerate(factor_names)}
        self._chol = np.linalg.cholesky(factor_cov)
        self._cov_inv = np.linalg.inv(factor_cov)
        groups: dict[int, tuple[Marginal, list[int]]] = {}
        for v in variables:
            groups.setdefault(id(v.marginal), (v.marginal, []))[1].append(v.index)
        self._marginal_groups = [(m, np.asarray(cols)) for m, cols in groups.values()]

    # ------------------------------------------------------------ factory

    @classmethod
    def build(
        cls, network: ServiceNetwork, config: UncertaintyConfig, mode: str = "joint"
    ) -> "FactorModel":
        if mode not in {"joint", "independent", "nominal"}:
            raise ValueError(f"Unknown uncertainty mode: {mode}")
        marginals: dict[str, Marginal] = {
            "rail_delay": ZeroInflatedLogNormal(*config.rail_delay),
            "barge_delay": ZeroInflatedLogNormal(*config.barge_delay),
            "handling": ZeroInflatedLogNormal(*config.handling_extra),
            "truck_time": BoundedLogNormal(*config.truck_factor),
            "rail_capacity": DiscreteLevels(tuple(config.capacity_levels)),
            "barge_capacity": DiscreteLevels(tuple(config.capacity_levels)),
            "demand": BoundedLogNormal(*config.demand_factor),
            "release_delay": ZeroInflatedLogNormal(*config.release_delay),
        }

        specs: list[tuple[tuple, str, str | None]] = []
        for s in network.scheduled_services:
            corridor = f"corridor:{s.origin}-{s.destination}:{s.mode}"
            specs.append((("delay", s.id), f"{s.mode}_delay", corridor))
            specs.append((("capacity", s.id), f"{s.mode}_capacity", corridor))
        for node in network.handling_nodes:
            for b in range(network.n_blocks):
                specs.append((("handling", node, b), "handling", f"node:{node}"))
        for o, d in network.truck_legs:
            for b in range(network.n_blocks):
                specs.append((("truck", o, d, b), "truck_time", f"road:{o}"))
        for oid in sorted(network.orders):
            specs.append((("demand", oid), "demand", None))
            specs.append((("release", oid), "release_delay", None))

        global_names = list(config.global_factors)
        group_names = sorted({g for _, _, g in specs if g is not None})
        factor_names = global_names + group_names
        n_g, m = len(global_names), len(global_names) + len(group_names)

        cov = np.eye(m)
        for (a, b), rho in config.factor_correlation.items():
            if a in global_names and b in global_names:
                i, j = global_names.index(a), global_names.index(b)
                cov[i, j] = cov[j, i] = rho
        eigval, eigvec = np.linalg.eigh(cov[:n_g, :n_g])
        cov[:n_g, :n_g] = eigvec @ np.diag(np.clip(eigval, 1e-3, None)) @ eigvec.T

        variables: list[UncertainVariable] = []
        loadings = np.zeros((len(specs), m))
        for j, (key, source, group) in enumerate(specs):
            variables.append(UncertainVariable(j, key, source, marginals[source], group))
            if mode == "independent":
                continue
            for name, value in config.loadings.get(source, {}).items():
                loadings[j, global_names.index(name)] = value
            if group is not None:
                loadings[j, factor_names.index(group)] = config.group_loadings.get(source, 0.0)
            explained = loadings[j] @ cov @ loadings[j]
            if explained > 0.9:
                loadings[j] *= np.sqrt(0.9 / explained)

        return cls(factor_names, cov, variables, loadings, mode)

    # ------------------------------------------------------------ queries

    @property
    def n_factors(self) -> int:
        return len(self.factor_names)

    @property
    def n_vars(self) -> int:
        return len(self.variables)

    def index(self, key: tuple) -> int:
        return self._index[key]

    def get_index(self, key: tuple) -> int | None:
        return self._index.get(key)

    def factor_index(self, name: str) -> int:
        return self._factor_index[name]

    # ----------------------------------------------------------- sampling

    def sample_factors(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal((n, self.n_factors)) @ self._chol.T

    def latent_mean(self, F: np.ndarray) -> np.ndarray:
        return F @ self.L.T

    def sample_latent(self, F: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return self.latent_mean(F) + rng.standard_normal((F.shape[0], self.n_vars)) * self.psi

    def to_values(self, Z: np.ndarray) -> np.ndarray:
        X = np.empty_like(Z)
        for marginal, cols in self._marginal_groups:
            X[:, cols] = marginal.from_z(Z[:, cols])
        return X

    def value_of_z(self, j: int, z: float) -> float:
        return float(self.variables[j].marginal.from_z(np.asarray([z]))[0])

    def log_prior_factors(self, F: np.ndarray) -> np.ndarray:
        return -0.5 * np.einsum("nf,fg,ng->n", F, self._cov_inv, F)

    def correlation_of_latents(self) -> np.ndarray:
        """Model-implied correlation matrix of the latent variables z."""
        cov = self.L @ self.factor_cov @ self.L.T
        cov[np.diag_indices_from(cov)] = 1.0
        return cov
