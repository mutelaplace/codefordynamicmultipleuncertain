"""Joint scenario sets: generation, reduction and per-scenario access."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dynamic_intermodal.config import UncertaintyConfig
from dynamic_intermodal.network import ServiceNetwork
from dynamic_intermodal.uncertainty.factors import FactorModel


class ScenarioView:
    """Read access to one joint scenario, used by risk propagation."""

    __slots__ = ("_x", "_model", "_network")

    def __init__(self, x: np.ndarray, model: FactorModel, network: ServiceNetwork):
        self._x = x
        self._model = model
        self._network = network

    def _value(self, key: tuple, default: float) -> float:
        j = self._model.get_index(key)
        return default if j is None else float(self._x[j])

    def delay(self, service_id: int) -> float:
        return self._value(("delay", service_id), 0.0)

    def capacity_factor(self, service_id: int) -> float:
        return self._value(("capacity", service_id), 1.0)

    def handling(self, node: int, t: float) -> float:
        return self._value(("handling", node, self._network.block_of(t)), 0.0)

    def truck_factor(self, origin: int, destination: int, t: float) -> float:
        return self._value(("truck", origin, destination, self._network.block_of(t)), 1.0)

    def demand_factor(self, order_id: int) -> float:
        return self._value(("demand", order_id), 1.0)

    def release_delay(self, order_id: int) -> float:
        return self._value(("release", order_id), 0.0)


@dataclass
class ScenarioSet:
    model: FactorModel
    network: ServiceNetwork
    F: np.ndarray  # (N, m) factor values
    Z: np.ndarray  # (N, J) latent values
    X: np.ndarray  # (N, J) physical values
    weights: np.ndarray  # (N,) normalised probabilities
    base_weights: np.ndarray | None = None  # weights right after (re)generation
    base_loglik: np.ndarray | None = None  # observation log-likelihood at (re)generation
    realized: dict[int, float] = field(default_factory=dict)  # fixed (observed) values

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=float)
        self.weights = self.weights / self.weights.sum()
        if self.base_weights is None:
            self.base_weights = self.weights.copy()
        if self.base_loglik is None:
            self.base_loglik = np.zeros(len(self.weights))

    def __len__(self) -> int:
        return len(self.weights)

    def view(self, s: int) -> ScenarioView:
        return ScenarioView(self.X[s], self.model, self.network)

    def views(self) -> list[ScenarioView]:
        return [self.view(s) for s in range(len(self))]

    def ess(self) -> float:
        return float(1.0 / np.sum(self.weights**2))

    def free_columns(self) -> np.ndarray:
        mask = np.ones(self.model.n_vars, dtype=bool)
        if self.realized:
            mask[list(self.realized)] = False
        return mask

    def subset(self, indices: np.ndarray, weights: np.ndarray | None = None) -> "ScenarioSet":
        indices = np.asarray(indices)
        w = self.weights[indices] if weights is None else weights
        return ScenarioSet(
            model=self.model,
            network=self.network,
            F=self.F[indices].copy(),
            Z=self.Z[indices].copy(),
            X=self.X[indices].copy(),
            weights=w,
            base_weights=self.base_weights[indices].copy(),
            base_loglik=self.base_loglik[indices].copy(),
            realized=dict(self.realized),
        )

    def expected(self, key: tuple) -> float:
        j = self.model.index(key)
        return float(self.weights @ self.X[:, j])


def reduce_scenarios(
    points: np.ndarray,
    weights: np.ndarray,
    n_target: int,
    rng: np.random.Generator,
    iterations: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted k-means reduction; returns medoid indices and aggregated cluster weights."""
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    n = len(points)
    if n <= n_target:
        return np.arange(n), weights

    centers = np.empty((n_target, points.shape[1]))
    first = rng.choice(n, p=weights)
    centers[0] = points[first]
    d2 = np.sum((points - centers[0]) ** 2, axis=1)
    for k in range(1, n_target):
        prob = weights * d2
        total = prob.sum()
        idx = rng.choice(n, p=prob / total) if total > 0 else rng.choice(n, p=weights)
        centers[k] = points[idx]
        d2 = np.minimum(d2, np.sum((points - centers[k]) ** 2, axis=1))

    for _ in range(iterations):
        dist = (
            np.sum(points**2, axis=1)[:, None]
            - 2.0 * points @ centers.T
            + np.sum(centers**2, axis=1)[None, :]
        )
        assign = np.argmin(dist, axis=1)
        for k in range(n_target):
            members = assign == k
            wk = weights[members].sum()
            if wk > 0:
                centers[k] = weights[members] @ points[members] / wk

    dist = (
        np.sum(points**2, axis=1)[:, None]
        - 2.0 * points @ centers.T
        + np.sum(centers**2, axis=1)[None, :]
    )
    assign = np.argmin(dist, axis=1)
    medoids, cluster_weights = [], []
    for k in range(n_target):
        members = np.flatnonzero(assign == k)
        if len(members) == 0:
            continue
        wk = weights[members].sum()
        if wk <= 0:
            continue
        medoids.append(members[np.argmin(dist[members, k])])
        cluster_weights.append(wk)
    cluster_weights = np.asarray(cluster_weights)
    return np.asarray(medoids), cluster_weights / cluster_weights.sum()


class ScenarioGenerator:
    def __init__(
        self,
        model: FactorModel,
        network: ServiceNetwork,
        config: UncertaintyConfig,
        rng: np.random.Generator,
    ):
        self.model = model
        self.network = network
        self.config = config
        self.rng = rng

    def generate(self, n_scenarios: int | None = None, n_samples: int | None = None) -> ScenarioSet:
        if self.model.mode == "nominal":
            return self.nominal()
        n_scenarios = n_scenarios or self.config.n_scenarios
        n_samples = max(n_samples or self.config.n_prior_samples, n_scenarios)
        F = self.model.sample_factors(n_samples, self.rng)
        Z = self.model.sample_latent(F, self.rng)
        idx, w = reduce_scenarios(
            Z, np.ones(n_samples), n_scenarios, self.rng, self.config.kmeans_iterations
        )
        return ScenarioSet(
            self.model, self.network, F[idx], Z[idx], self.model.to_values(Z[idx]), w
        )

    def nominal(self) -> ScenarioSet:
        """Single 'typical' scenario (all latent values at their median) for deterministic planning."""
        F = np.zeros((1, self.model.n_factors))
        Z = np.zeros((1, self.model.n_vars))
        return ScenarioSet(self.model, self.network, F, Z, self.model.to_values(Z), np.ones(1))


def sample_truth(
    model: FactorModel, network: ServiceNetwork, rng: np.random.Generator
) -> ScenarioSet:
    """Hidden realisation of the world (one draw of the joint model)."""
    F = model.sample_factors(1, rng)
    Z = model.sample_latent(F, rng)
    return ScenarioSet(model, network, F, Z, model.to_values(Z), np.ones(1))
