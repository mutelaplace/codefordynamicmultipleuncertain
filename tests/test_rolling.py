import pytest

from dynamic_intermodal.config import Config, DataConfig
from dynamic_intermodal.coordination.milp import gp
from dynamic_intermodal.simulation.methods import METHODS
from dynamic_intermodal.simulation.rolling import RollingHorizonController

pytestmark = pytest.mark.skipif(gp is None, reason="gurobipy not available")


@pytest.mark.parametrize("method", ["full", "deterministic", "static"])
def test_rolling_run_completes(method):
    config = Config(data=DataConfig(num_requests=10))
    config.uncertainty.n_scenarios = 12
    config.uncertainty.n_prior_samples = 200
    config.uncertainty.rejuvenation_samples = 200
    config.optimization.validation_scenarios = 12
    controller = RollingHorizonController(config, METHODS[method], seed=3)
    result = controller.run()
    m = result.metrics
    assert m["total_cost"] > 0
    assert 0.0 <= m["on_time_rate"] <= 1.0
    assert set(result.final_plan.orders) == set(controller.network.orders)
    for oid, out in result.truth_outcome.orders.items():
        assert out.final_arrival == out.final_arrival  # not NaN: every order finished
    if method == "static":
        assert m["change_requests"] == 0
