"""Marginal distributions expressed as monotone maps of a standard-normal latent value.

Each marginal maps latent z ~ N(0, 1) to the physical value (``from_z``) and maps an
observed physical value back to the latent interval consistent with it (``z_interval``),
which the scenario updater turns into a likelihood. Higher z always means a worse outcome
(longer delay, slower trucks, less capacity, more demand).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri

INF = math.inf


def _ndtri(u: float) -> float:
    return float(ndtri(min(max(u, 1e-12), 1 - 1e-12)))


@dataclass(frozen=True)
class ZeroInflatedLogNormal:
    """P(X=0)=p0, otherwise log-normal(median, sigma), truncated at cap (delays, extra time)."""

    p0: float
    median: float
    sigma: float
    cap: float

    def from_z(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        u = ndtr(z)
        q = np.clip((u - self.p0) / (1.0 - self.p0), 1e-12, 1 - 1e-12)
        positive = np.exp(math.log(self.median) + self.sigma * ndtri(q))
        return np.where(u <= self.p0, 0.0, np.minimum(positive, self.cap))

    def z_of(self, x: float) -> float:
        """Latent value at which the distribution reaches x (x > 0)."""
        if x <= 0:
            return _ndtri(self.p0)
        if x >= self.cap:
            x = self.cap
        q = ndtr((math.log(x) - math.log(self.median)) / self.sigma)
        return _ndtri(self.p0 + (1.0 - self.p0) * float(q))

    def z_interval(self, x: float) -> tuple[float, float]:
        if x <= 1e-9:
            return (-INF, _ndtri(self.p0))
        if x >= self.cap - 1e-9:
            return (self.z_of(self.cap), INF)
        z = self.z_of(x)
        return (z, z)

    def exceed_threshold(self, x: float) -> float:
        """Latent threshold c such that X > x  <=>  z > c."""
        return self.z_of(x) if x > 0 else _ndtri(self.p0)


@dataclass(frozen=True)
class BoundedLogNormal:
    """Multiplicative factor, log-normal(median, sigma) clipped to [lo, hi]."""

    median: float
    sigma: float
    lo: float
    hi: float

    def from_z(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        return np.clip(np.exp(math.log(self.median) + self.sigma * z), self.lo, self.hi)

    def z_of(self, x: float) -> float:
        return (math.log(x) - math.log(self.median)) / self.sigma

    def z_interval(self, x: float) -> tuple[float, float]:
        if x <= self.lo + 1e-9:
            return (-INF, self.z_of(self.lo))
        if x >= self.hi - 1e-9:
            return (self.z_of(self.hi), INF)
        z = self.z_of(x)
        return (z, z)

    def exceed_threshold(self, x: float) -> float:
        return self.z_of(min(max(x, self.lo), self.hi))


@dataclass(frozen=True)
class DiscreteLevels:
    """Ordered levels from best to worst, e.g. capacity ((0.87, 1.0), (0.10, 0.6), (0.03, 0.0))."""

    levels: tuple[tuple[float, float], ...]

    @property
    def thresholds(self) -> list[float]:
        cum = np.cumsum([p for p, _ in self.levels])[:-1]
        return [_ndtri(float(c)) for c in cum]

    def from_z(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        idx = np.searchsorted(np.asarray(self.thresholds), z, side="right")
        values = np.asarray([v for _, v in self.levels])
        return values[idx]

    def z_interval(self, x: float) -> tuple[float, float]:
        values = [v for _, v in self.levels]
        idx = int(np.argmin([abs(v - x) for v in values]))
        bounds = [-INF, *self.thresholds, INF]
        return (bounds[idx], bounds[idx + 1])

    def exceed_threshold(self, x: float) -> float:
        raise NotImplementedError("censored observations are not used for discrete levels")


Marginal = ZeroInflatedLogNormal | BoundedLogNormal | DiscreteLevels
