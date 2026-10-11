#!/usr/bin/env python3
"""Run one method on one hidden-truth realisation, print the epoch log and write the report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dynamic_intermodal  # noqa: F401  (sets BLAS thread limits before numpy)

from dynamic_intermodal.config import Config, DataConfig
from dynamic_intermodal.reporting import write_report
from dynamic_intermodal.simulation.methods import METHODS
from dynamic_intermodal.simulation.rolling import RollingHorizonController


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHODS), default="full")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--requests", type=int, default=30, help=">20 adds resampled orders to R_20")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "demo")
    args = parser.parse_args()

    config = Config(
        data=DataConfig(
            num_requests=min(args.requests, 20),
            augment_to=args.requests if args.requests > 20 else None,
        )
    )
    result = RollingHorizonController(config, METHODS[args.method], args.seed, verbose=True).run()
    print()
    for key, value in result.metrics.items():
        print(f"{key:22s} {value:12.2f}")
    path = write_report(result, args.out)
    print(f"\nreport: {path} (+ .md)")


if __name__ == "__main__":
    main()
