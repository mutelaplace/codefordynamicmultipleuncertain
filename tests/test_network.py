from dynamic_intermodal.network import ServiceNetwork


def service_by_name(network: ServiceNetwork, name: str):
    return next(s for s in network.services.values() if s.name == name)


def test_services_and_orders(network):
    assert len(network.services) == 52
    assert len(network.scheduled_services) == 42
    assert len(network.truck_legs) == 10
    assert len(network.orders) == 20
    barge = service_by_name(network, "Barge1")
    # Fixed_vehicles: Delta (11, 12) -> Venlo (25, 26): cut-off = loading start, arrival, unloading.
    assert (barge.cutoff, barge.departure, barge.arrival, barge.unload_time) == (11, 12, 25, 1)
    assert barge.carrier_id.startswith("barge-")


def test_chains_are_time_feasible(network):
    order = network.orders[0]
    chains = network.enumerate_chains(order, order.origin, order.release, 3)
    assert chains
    for chain in chains:
        legs = network.nominal_chain(order, chain, order.release)
        assert legs is not None
        ready = order.release
        for leg in legs[0]:
            service = network.services[leg.service_id]
            if service.scheduled:
                assert ready <= service.cutoff + 1e-9
            ready = leg.ready
        assert network.services[chain[-1]].destination == order.destination


def test_nominal_leg_rejects_missed_cutoff(network):
    barge = service_by_name(network, "Barge1")
    assert network.nominal_leg(barge, barge.cutoff + 0.5) is None
    leg = network.nominal_leg(barge, barge.cutoff - 3)
    assert leg.wait == 3 and leg.ready == barge.arrival + barge.unload_time
