"""Transport plan: per order, the chain of booked services."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator


@dataclass(frozen=True)
class Leg:
    service_id: int
    booked_at: float = 0.0  # when the booking was confirmed; earlier bookings board first


@dataclass(frozen=True)
class OrderPlan:
    order_id: int
    legs: tuple[Leg, ...]

    @property
    def service_ids(self) -> tuple[int, ...]:
        return tuple(leg.service_id for leg in self.legs)


@dataclass
class TransportPlan:
    orders: dict[int, OrderPlan] = field(default_factory=dict)

    def __contains__(self, order_id: int) -> bool:
        return order_id in self.orders

    def __getitem__(self, order_id: int) -> OrderPlan:
        return self.orders[order_id]

    def __iter__(self) -> Iterator[int]:
        return iter(sorted(self.orders))

    def __len__(self) -> int:
        return len(self.orders)

    def items(self) -> list[tuple[int, OrderPlan]]:
        return [(oid, self.orders[oid]) for oid in sorted(self.orders)]

    def set(self, order_id: int, legs: Iterable[Leg]) -> None:
        self.orders[order_id] = OrderPlan(order_id, tuple(legs))

    def copy(self) -> "TransportPlan":
        return TransportPlan(dict(self.orders))

    def subset(self, order_ids: Iterable[int]) -> "TransportPlan":
        return TransportPlan({oid: self.orders[oid] for oid in order_ids if oid in self.orders})

    def bookings(self) -> dict[int, list[int]]:
        """service_id -> order ids booked on it."""
        result: dict[int, list[int]] = {}
        for oid, order_plan in self.items():
            for leg in order_plan.legs:
                result.setdefault(leg.service_id, []).append(oid)
        return result


def changed_positions(old: tuple[int, ...], new: tuple[int, ...]) -> int:
    """Number of legs that differ between two service chains."""
    length = max(len(old), len(new))
    return sum(
        1
        for i in range(length)
        if i >= len(old) or i >= len(new) or old[i] != new[i]
    )
