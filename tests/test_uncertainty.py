import numpy as np

from dynamic_intermodal.uncertainty.factors import FactorModel
from dynamic_intermodal.uncertainty.marginals import BoundedLogNormal, DiscreteLevels, ZeroInflatedLogNormal
from dynamic_intermodal.uncertainty.scenarios import ScenarioGenerator, reduce_scenarios


def test_latent_variables_have_unit_variance(network, config):
    model = FactorModel.build(network, config.uncertainty, "joint")
    explained = np.einsum("jf,fg,jg->j", model.L, model.factor_cov, model.L)
    assert np.allclose(explained + model.psi**2, 1.0, atol=1e-6)
    F = model.sample_factors(20000, np.random.default_rng(0))
    Z = model.sample_latent(F, np.random.default_rng(1))
    assert np.allclose(Z.std(axis=0), 1.0, atol=0.05)


def test_joint_model_links_sources_and_independent_does_not(network, config):
    joint = FactorModel.build(network, config.uncertainty, "joint")
    indep = FactorModel.build(network, config.uncertainty, "independent")
    trains = [s for s in network.scheduled_services if s.mode == "rail"]
    same_corridor = [s for s in trains if s.leg == trains[0].leg][:2]
    a, b = (joint.index(("delay", s.id)) for s in same_corridor)
    h = joint.index(("handling", network.handling_nodes[0], 1))
    r = joint.index(("release", 0))
    corr = joint.correlation_of_latents()
    assert corr[a, b] > 0.3  # same corridor, same source
    assert corr[h, r] > 0.1  # terminal handling vs order release (port congestion)
    assert np.allclose(indep.L, 0.0)


def test_marginal_intervals_are_consistent():
    rng = np.random.default_rng(0)
    for marginal in (
        ZeroInflatedLogNormal(0.4, 3.0, 0.8, 24.0),
        BoundedLogNormal(1.05, 0.25, 0.9, 2.5),
        DiscreteLevels(((0.87, 1.0), (0.1, 0.6), (0.03, 0.0))),
    ):
        for z in rng.standard_normal(200) * 1.5:
            x = float(marginal.from_z(np.array([z]))[0])
            lo, hi = marginal.z_interval(x)
            assert lo - 1e-6 <= z <= hi + 1e-6


def test_scenario_reduction_weights(network, config):
    model = FactorModel.build(network, config.uncertainty, "joint")
    sset = ScenarioGenerator(model, network, config.uncertainty, np.random.default_rng(0)).generate(20, 300)
    assert len(sset) <= 20
    assert abs(sset.weights.sum() - 1.0) < 1e-9
    idx, w = reduce_scenarios(np.random.default_rng(1).standard_normal((50, 3)), np.ones(50), 60, np.random.default_rng(2))
    assert len(idx) == 50 and abs(w.sum() - 1) < 1e-9
