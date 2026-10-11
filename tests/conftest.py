import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dynamic_intermodal  # noqa: E402,F401  (BLAS thread limits before numpy)
import pytest  # noqa: E402

from dynamic_intermodal.config import Config, DataConfig  # noqa: E402
from dynamic_intermodal.data import load_problem  # noqa: E402
from dynamic_intermodal.network import build_network  # noqa: E402


class FakeView:
    """Hand-made scenario: only the listed values deviate from the nominal ones."""

    def __init__(self, delay=None, capacity=None, handling=None, truck=None, demand=None, release=None):
        self._delay = delay or {}
        self._capacity = capacity or {}
        self._handling = handling or {}
        self._truck = truck or {}
        self._demand = demand or {}
        self._release = release or {}

    def delay(self, sid):
        return self._delay.get(sid, 0.0)

    def capacity_factor(self, sid):
        return self._capacity.get(sid, 1.0)

    def handling(self, node, t):
        return self._handling.get(node, 0.0)

    def truck_factor(self, o, d, t):
        return self._truck.get((o, d), 1.0)

    def demand_factor(self, oid):
        return self._demand.get(oid, 1.0)

    def release_delay(self, oid):
        return self._release.get(oid, 0.0)


@pytest.fixture(scope="session")
def config():
    return Config(data=DataConfig(num_requests=20))


@pytest.fixture(scope="session")
def network(config):
    return build_network(load_problem(config), config)


@pytest.fixture
def fake_view():
    return FakeView
