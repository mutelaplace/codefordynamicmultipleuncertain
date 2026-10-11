import numpy as np

from dynamic_intermodal.uncertainty.factors import FactorModel
from dynamic_intermodal.uncertainty.scenarios import ScenarioGenerator
from dynamic_intermodal.update.observations import Observation
from dynamic_intermodal.update.updater import ScenarioUpdater


def _setup(network, config, mode):
    model = FactorModel.build(network, config.uncertainty, mode)
    sset = ScenarioGenerator(model, network, config.uncertainty, np.random.default_rng(0)).generate()
    updater = ScenarioUpdater(model, config.uncertainty, np.random.default_rng(1))
    return model, sset, updater


def test_realised_values_are_fixed_and_censoring_repaired(network, config):
    model, sset, updater = _setup(network, config, "joint")
    s0, s1 = network.scheduled_services[:2]
    obs = [
        Observation(30.0, "value", "arrival", ("delay", s0.id), 5.0),
        Observation(30.0, "exceeds", "arrival", ("delay", s1.id), 7.0),
    ]
    sset, _ = updater.update(sset, obs, 30.0)
    assert np.allclose(sset.X[:, model.index(("delay", s0.id))], 5.0)
    assert np.all(sset.X[:, model.index(("delay", s1.id))] > 7.0)


def test_rail_observations_shift_unobserved_sources(network, config):
    """Observed late trains raise the rail factor and, through it, expected delays elsewhere."""
    model, sset, updater = _setup(network, config, "joint")
    trains = [s for s in network.scheduled_services if s.mode == "rail"]
    observed, future = trains[:12], [s for s in trains[12:]]
    prior = np.mean([sset.expected(("delay", s.id)) for s in future])
    obs = [Observation(40.0, "value", "arrival", ("delay", s.id), 12.0) for s in observed]
    sset, report = updater.update(sset, obs, 40.0)
    posterior = np.mean([sset.expected(("delay", s.id)) for s in future])
    assert report.factor_mean["rail_congestion"] > 0.5
    assert posterior > prior + 0.5


def test_independent_model_does_not_transfer_information(network, config):
    model, sset, updater = _setup(network, config, "independent")
    trains = [s for s in network.scheduled_services if s.mode == "rail"]
    future = trains[12:]
    prior = np.mean([sset.expected(("delay", s.id)) for s in future])
    obs = [Observation(40.0, "value", "arrival", ("delay", s.id), 12.0) for s in trains[:12]]
    sset, _ = updater.update(sset, obs, 40.0)
    posterior = np.mean([sset.expected(("delay", s.id)) for s in future])
    assert abs(posterior - prior) < 0.5
