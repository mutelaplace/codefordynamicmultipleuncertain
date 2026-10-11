from dynamic_intermodal.plan import Leg, TransportPlan
from dynamic_intermodal.risk.propagation import PropagationEngine


def by_name(network, name):
    return next(s for s in network.services.values() if s.name == name)


def test_delay_causes_missed_connection_and_rebooking(network, fake_view):
    # Order 0: Delta -> Nuremberg. Barge14 (Delta 21 -> Duisburg 38) + Train2 (Duisburg 41 -> Nuremberg).
    barge, train = by_name(network, "Barge14"), by_name(network, "Train2")
    plan = TransportPlan()
    plan.set(0, [Leg(barge.id), Leg(train.id)])
    engine = PropagationEngine(network)

    nominal = engine.simulate(plan, fake_view()).orders[0]
    assert nominal.missed == 0 and [l.service for l in nominal.legs] == [barge.id, train.id]

    late = engine.simulate(plan, fake_view(delay={barge.id: 8.0})).orders[0]
    assert late.missed == 1
    replacement = late.legs[1]
    assert replacement.service != train.id
    assert network.services[replacement.service].leg == train.leg  # same leg, next available service
    assert late.legs[1].wait > 0  # waits on the yard for the follow-up service
    assert any(node == train.origin for node, *_ in late.occupancy)


def test_capacity_goes_to_earlier_bookings(network, fake_view):
    barge = by_name(network, "Barge14")
    plan = TransportPlan()
    for oid, booked_at in ((0, 0.0), (1, 5.0)):
        plan.set(oid, [Leg(barge.id, booked_at), Leg(by_name(network, "Train2").id, booked_at)])
    # Capacity reduced so that only one of the two orders fits.
    factor = 30.0 / barge.capacity
    out = PropagationEngine(network).simulate(plan, fake_view(capacity={barge.id: factor}))
    assert out.orders[0].missed == 0
    assert out.orders[1].missed >= 1
    assert out.service_bumped[barge.id] > 0
    assert any(r.reason == "capacity" for r in out.records)


def test_yard_congestion_slows_handling(network, fake_view):
    barge, train = by_name(network, "Barge14"), by_name(network, "Train2")
    plan = TransportPlan()
    plan.set(0, [Leg(barge.id), Leg(train.id)])
    engine = PropagationEngine(network)
    free = engine.simulate(plan, fake_view()).orders[0]
    slot = network.slot_of(barge.arrival)
    jammed = engine.simulate(
        plan, fake_view(), yard_background={(barge.destination, slot): 2 * network.yard_capacity[barge.destination]}
    ).orders[0]
    assert jammed.legs[0].ready > free.legs[0].ready
