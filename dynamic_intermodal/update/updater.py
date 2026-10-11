"""Observation-driven update of the joint scenario set.

1. Fix realised states: every observed value overwrites that variable in all scenarios, so the
   past is common to all scenarios and decisions only use information available now.
2. Repair the feasible scenario set: censored facts ("service not arrived yet") resample the
   latent value of inconsistent scenarios from its truncated conditional distribution.
3. Re-weight: w_s ∝ w_s^0 * p(all observations | F_s) / p(observations at (re)generation | F_s),
   the likelihood being evaluated in the latent factor space. Because sources share factors,
   observing e.g. rail delays also shifts the weight of future handling / road / capacity values.
4. Filter negligible scenarios; if the effective sample size collapses, regenerate scenarios
   from the factor posterior (importance sampling + systematic resampling).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import log_ndtr, ndtr, ndtri
from scipy.stats import truncnorm

from dynamic_intermodal.config import UncertaintyConfig
from dynamic_intermodal.uncertainty.factors import FactorModel
from dynamic_intermodal.uncertainty.scenarios import ScenarioSet
from dynamic_intermodal.update.observations import Observation


@dataclass
class UpdateReport:
    time: float
    n_observations: dict[str, int] = field(default_factory=dict)
    ess_before: float = 0.0
    ess_after: float = 0.0
    n_filtered: int = 0
    n_repaired: int = 0
    rejuvenated: bool = False
    factor_mean: dict[str, float] = field(default_factory=dict)


def _mvn_logpdf(F: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    sign, logdet = np.linalg.slogdet(cov)
    diff = F - mean
    sol = np.linalg.solve(cov, diff.T).T
    return -0.5 * (np.sum(diff * sol, axis=1) + logdet + F.shape[1] * np.log(2 * np.pi))


class ScenarioUpdater:
    def __init__(
        self,
        model: FactorModel,
        config: UncertaintyConfig,
        rng: np.random.Generator,
        enabled: bool = True,
    ):
        self.model = model
        self.config = config
        self.rng = rng
        self.enabled = enabled and model.mode != "nominal"
        self.point_obs: dict[int, tuple[float, float]] = {}  # j -> latent interval
        self.exceed_obs: dict[int, float] = {}  # j -> latent lower bound
        self.indicator_obs: list[tuple[int, float, float]] = []  # (factor, value, noise sd)

    # ---------------------------------------------------------- likelihood

    def loglik(self, F: np.ndarray) -> np.ndarray:
        """log p(all observations so far | factors F) for each row of F."""
        L, psi = self.model.L, self.model.psi
        ll = np.zeros(F.shape[0])
        if self.point_obs:
            idx = np.fromiter(self.point_obs, dtype=int)
            lo = np.array([self.point_obs[j][0] for j in idx])
            hi = np.array([self.point_obs[j][1] for j in idx])
            mean = F @ L[idx].T
            sd = psi[idx]
            is_point = lo == hi
            if is_point.any():
                zc = (lo[is_point] - mean[:, is_point]) / sd[is_point]
                ll += np.sum(-0.5 * zc**2 - np.log(sd[is_point]), axis=1)
            if (~is_point).any():
                a = (lo[~is_point] - mean[:, ~is_point]) / sd[~is_point]
                b = (hi[~is_point] - mean[:, ~is_point]) / sd[~is_point]
                prob = np.clip(ndtr(b) - ndtr(a), 1e-300, None)
                ll += np.sum(np.log(prob), axis=1)
        if self.exceed_obs:
            idx = np.fromiter(self.exceed_obs, dtype=int)
            c = np.array([self.exceed_obs[j] for j in idx])
            mean = F @ L[idx].T
            ll += np.sum(log_ndtr((mean - c) / psi[idx]), axis=1)
        for f, y, sd in self.indicator_obs:
            ll += -0.5 * ((y - F[:, f]) / sd) ** 2
        return ll

    # -------------------------------------------------------------- update

    def update(
        self, sset: ScenarioSet, observations: list[Observation], t: float
    ) -> tuple[ScenarioSet, UpdateReport]:
        report = UpdateReport(time=t, ess_before=sset.ess())
        model = self.model
        for obs in observations:
            report.n_observations[obs.source] = report.n_observations.get(obs.source, 0) + 1
            if obs.kind == "indicator":
                if model.mode == "joint":
                    self.indicator_obs.append((model.factor_index(obs.factor), obs.value, obs.noise))
                continue
            j = model.get_index(obs.key)
            if j is None:
                continue
            marginal = model.variables[j].marginal
            if obs.kind == "value":
                lo, hi = marginal.z_interval(obs.value)
                self.point_obs[j] = (lo, hi)
                self.exceed_obs.pop(j, None)
                sset.realized[j] = obs.value
                sset.X[:, j] = obs.value
                sset.Z[:, j] = np.clip(sset.Z[:, j], lo, hi) if lo != hi else lo
            elif obs.kind == "exceeds" and j not in sset.realized:
                c = marginal.exceed_threshold(obs.value)
                self.exceed_obs[j] = max(c, self.exceed_obs.get(j, -np.inf))
                bad = np.flatnonzero(sset.X[:, j] <= obs.value + 1e-9)
                if len(bad):
                    report.n_repaired += len(bad)
                    z = self._truncated(sset.F[bad] @ model.L[j], model.psi[j], self.exceed_obs[j])
                    sset.Z[bad, j] = z
                    sset.X[bad, j] = np.maximum(
                        marginal.from_z(z), obs.value + 1e-6
                    )

        if self.enabled and len(sset) > 1:
            ll = self.loglik(sset.F)
            logw = np.log(np.clip(sset.base_weights, 1e-300, None)) + ll - sset.base_loglik
            logw -= logw.max()
            w = np.exp(logw)
            sset.weights = w / w.sum()
            keep = np.flatnonzero(sset.weights >= self.config.weight_floor)
            report.n_filtered = len(sset) - len(keep)
            if report.n_filtered:
                sset = sset.subset(keep)
                sset.weights = sset.weights / sset.weights.sum()
            if (
                sset.ess() < self.config.ess_ratio_threshold * self.config.n_scenarios
                or len(sset) < self.config.min_active_scenarios
            ):
                sset = self.rejuvenate(sset)
                report.rejuvenated = True

        report.ess_after = sset.ess()
        mean_f = sset.weights @ sset.F
        report.factor_mean = {
            name: float(mean_f[model.factor_index(name)]) for name in self.config.global_factors
        }
        return sset, report

    def _truncated(self, mean: np.ndarray, sd: float, lower: float) -> np.ndarray:
        if self.model.mode == "nominal":
            a = (lower - mean) / sd
            u = 0.5 * (1.0 + ndtr(a))  # conditional median above the bound
            return mean + sd * ndtri(np.clip(u, 1e-12, 1 - 1e-12))
        a = (lower - mean) / sd
        return truncnorm.rvs(a, np.inf, loc=mean, scale=sd, random_state=self.rng)

    # -------------------------------------------------------- rejuvenation

    def gaussian_posterior(self) -> tuple[np.ndarray, np.ndarray]:
        """Exact posterior of F given the point observations and indicators (linear-Gaussian part)."""
        model = self.model
        rows, values, variances = [], [], []
        for j, (lo, hi) in self.point_obs.items():
            if lo == hi:
                rows.append(model.L[j])
                values.append(lo)
                variances.append(model.psi[j] ** 2)
        for f, y, sd in self.indicator_obs:
            row = np.zeros(model.n_factors)
            row[f] = 1.0
            rows.append(row)
            values.append(y)
            variances.append(sd**2)
        mean, cov = np.zeros(model.n_factors), model.factor_cov.copy()
        if rows:
            H = np.asarray(rows)
            r_inv = 1.0 / np.asarray(variances)
            precision = np.linalg.inv(model.factor_cov) + H.T @ (H * r_inv[:, None])
            cov = np.linalg.inv(precision)
            mean = cov @ (H.T @ (r_inv * np.asarray(values)))
        # Interval and censored observations: assumed-density filtering (moment matching).
        bounds = [(j, lo, hi) for j, (lo, hi) in self.point_obs.items() if lo != hi]
        bounds += [(j, c, np.inf) for j, c in self.exceed_obs.items()]
        for j, lo, hi in bounds:
            h = model.L[j]
            ch = cov @ h
            a = float(h @ mean)
            s2 = float(h @ ch) + model.psi[j] ** 2
            s = np.sqrt(s2)
            al, be = (lo - a) / s, (hi - a) / s
            z = max(float(ndtr(be) - ndtr(al)), 1e-12)
            pa = 0.0 if np.isinf(al) else np.exp(-0.5 * al**2) / np.sqrt(2 * np.pi)
            pb = 0.0 if np.isinf(be) else np.exp(-0.5 * be**2) / np.sqrt(2 * np.pi)
            apa = 0.0 if np.isinf(al) else al * pa
            bpb = 0.0 if np.isinf(be) else be * pb
            m_t = a + s * (pa - pb) / z
            v_t = s2 * max(1.0 + (apa - bpb) / z - ((pa - pb) / z) ** 2, 1e-6)
            mean = mean + ch * (m_t - a) / s2
            cov = cov - np.outer(ch, ch) * (s2 - v_t) / s2**2
        return mean, 0.5 * (cov + cov.T)

    def rejuvenate(self, sset: ScenarioSet) -> ScenarioSet:
        return self.sample_posterior(sset, self.config.n_scenarios, use_observations=True)

    def validation_set(self, sset: ScenarioSet, n: int) -> ScenarioSet:
        """Independent scenario set for out-of-sample re-check of a proposed adjustment."""
        if self.model.mode == "nominal":
            return sset
        return self.sample_posterior(sset, n, use_observations=self.enabled)

    def sample_posterior(self, sset: ScenarioSet, n: int, use_observations: bool = True) -> ScenarioSet:
        """Draw fresh scenarios from the factor posterior given all observations so far.

        Proposal: Gaussian posterior of the linear part (point observations, indicators), slightly
        inflated, mixed with the prior; censored / interval observations enter via the weights.
        Without ``use_observations`` the factors come from the prior (realised values stay fixed).
        """
        model, cfg = self.model, self.config
        M = cfg.rejuvenation_samples
        m = model.n_factors
        if use_observations:
            mu, cov = self.gaussian_posterior()
            cov = 1.3 * cov
            n_post = int(0.9 * M)
            F_new = np.vstack([
                self.rng.multivariate_normal(mu, cov, size=n_post),
                model.sample_factors(M - n_post, self.rng),
            ])
            log_q = np.logaddexp(
                np.log(0.9) + _mvn_logpdf(F_new, mu, cov),
                np.log(0.1) + _mvn_logpdf(F_new, np.zeros(m), model.factor_cov),
            )
            log_target = _mvn_logpdf(F_new, np.zeros(m), model.factor_cov) + self.loglik(F_new)
            logw = log_target - log_q
            logw -= logw.max()
            iw = np.exp(logw)
            iw /= iw.sum()
        else:
            F_new = model.sample_factors(M, self.rng)
            iw = np.full(M, 1.0 / M)

        Z = model.sample_latent(F_new, self.rng)
        for j, c in self.exceed_obs.items():
            Z[:, j] = self._truncated(F_new @ model.L[j], model.psi[j], c)
        X = model.to_values(Z)
        for j, value in sset.realized.items():
            X[:, j] = value
            lo, hi = self.point_obs.get(j, (-np.inf, np.inf))
            Z[:, j] = np.clip(Z[:, j], lo, hi) if lo != hi else lo

        # Systematic resampling to n equally weighted draws; duplicates are merged.
        positions = (self.rng.random() + np.arange(n)) / n
        draws = np.minimum(np.searchsorted(np.cumsum(iw), positions), M - 1)
        idx, counts = np.unique(draws, return_counts=True)
        weights = counts / n
        return ScenarioSet(
            model=model,
            network=sset.network,
            F=F_new[idx],
            Z=Z[idx],
            X=X[idx],
            weights=weights,
            base_weights=weights.copy(),
            base_loglik=self.loglik(F_new[idx]),
            realized=dict(sset.realized),
        )
