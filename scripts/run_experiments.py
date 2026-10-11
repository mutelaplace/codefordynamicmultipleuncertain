#!/usr/bin/env python3
"""Compare the full method with baselines and ablations over many hidden-truth replications."""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dynamic_intermodal  # noqa: F401  (sets BLAS thread limits before numpy)
import pandas as pd

from dynamic_intermodal.config import Config, DataConfig
from dynamic_intermodal.simulation.methods import METHODS
from dynamic_intermodal.simulation.rolling import RollingHorizonController

KEY_METRICS = [
    "total_with_change", "total_cost", "operating_cost", "delay_cost", "total_delay_h",
    "on_time_rate", "missed_connections", "yard_overflow_teu_h", "bumped_teu",
    "changes_accepted", "changes_rejected", "change_cost", "milp_time_s", "runtime_s",
]


def build_config(args) -> Config:
    config = Config(
        data=DataConfig(
            num_requests=min(args.requests, 20),
            augment_to=args.requests if args.requests > 20 else None,
        )
    )
    for name in config.network.yard_capacity:
        if name in {"Venlo", "Duisburg", "Neuss"}:
            config.network.yard_capacity[name] *= args.yard_scale
    config.uncertainty.n_scenarios = args.scenarios
    config.simulation.epoch_hours = args.epoch
    if args.correlation_scale != 1.0:
        for source, loads in config.uncertainty.loadings.items():
            config.uncertainty.loadings[source] = {k: v * args.correlation_scale for k, v in loads.items()}
        config.uncertainty.group_loadings = {
            k: v * args.correlation_scale for k, v in config.uncertainty.group_loadings.items()
        }
    return config


def run_one(task):
    method, seed, args = task
    config = build_config(args)
    result = RollingHorizonController(config, METHODS[method], seed).run()
    return {"method": method, "seed": seed, **result.metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-offset", type=int, default=1)
    parser.add_argument("--requests", type=int, default=30, help=">20 adds resampled orders to R_20")
    parser.add_argument("--yard-scale", type=float, default=1.0, help="scale of inland yard capacities")
    parser.add_argument("--scenarios", type=int, default=30)
    parser.add_argument("--epoch", type=float, default=6.0)
    parser.add_argument("--correlation-scale", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "experiments.csv")
    args = parser.parse_args()

    tasks = [(m, s, args) for s in range(args.seed_offset, args.seed_offset + args.seeds) for m in args.methods]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_one, tasks))
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    summary = df.groupby("method")[KEY_METRICS].mean().reindex(args.methods)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(summary.round(2).to_string())
    base = summary.loc["static", "total_with_change"] if "static" in summary.index else None
    if base:
        print("\nTotal (incl. change cost) relative to static:")
        print((summary["total_with_change"] / base - 1).mul(100).round(2).astype(str).add(" %").to_string())
    print(f"\n{len(tasks)} runs in {time.perf_counter() - started:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
