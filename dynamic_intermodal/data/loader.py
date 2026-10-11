"""Excel data loading. Copied from intermodal_refactor (intermodal/data/loader.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dynamic_intermodal.config import CostConfig, DataConfig
from dynamic_intermodal.data.models import ProblemInstance, Request, Terminal, Vehicle


def _resolve_existing(data_dir: Path, preferred: str, fallbacks: tuple[str, ...]) -> Path:
    candidates = (preferred, *fallbacks)
    for name in candidates:
        path = data_dir / name
        if path.exists():
            return path
    return data_dir / preferred


def load_terminals(
    file_path: Path,
    config: DataConfig,
) -> tuple[dict[int, Terminal], dict[str, int]]:
    df_nodes = pd.read_excel(file_path, sheet_name="N")

    sheet_name = "T" if config.num_transshipment_nodes is None else f"T_{config.num_transshipment_nodes}"
    df_transfer = pd.read_excel(file_path, sheet_name=sheet_name)
    transfer_names = set(df_transfer["T"].values)

    terminals: dict[int, Terminal] = {}
    name_to_id: dict[str, int] = {}
    for idx, row in df_nodes.iterrows():
        name = row["N"]
        terminals[idx] = Terminal(id=idx, name=name, is_transfer=name in transfer_names)
        name_to_id[name] = idx
    return terminals, name_to_id


def load_agent_mapping(
    file_path: Path,
    name_to_id: dict[str, int],
) -> dict[tuple[int, int], int]:
    if not file_path.exists():
        return {}

    df_agent = pd.read_excel(file_path, sheet_name="Sheet1", index_col=0)
    mapping: dict[tuple[int, int], int] = {}

    for row_name, row in df_agent.iterrows():
        origin_name = str(row_name).strip()
        if origin_name not in name_to_id:
            continue
        origin_id = name_to_id[origin_name]

        for col_name, value in row.items():
            dest_name = str(col_name).strip()
            if dest_name not in name_to_id or pd.isna(value):
                continue
            try:
                agent_id = int(value)
            except (TypeError, ValueError):
                continue
            if agent_id <= 0:
                continue
            mapping[(origin_id, name_to_id[dest_name])] = agent_id
    return mapping


def load_vehicles(
    file_path: Path,
    fixed_file: Path,
    name_to_id: dict[str, int],
    cost_config: CostConfig,
    agent_mapping: dict[tuple[int, int], int] | None = None,
) -> dict[int, Vehicle]:
    df_vehicles = pd.read_excel(file_path, sheet_name="K")
    df_origins = pd.read_excel(file_path, sheet_name="o")

    origin_map: dict[str, int] = {}
    destination_map: dict[str, int] = {}
    for _, row in df_origins.iterrows():
        v_name = row["K"]
        origin_map[v_name] = name_to_id[row["o"]]
        destination_map[v_name] = name_to_id[row["o2"]]

    timewindows = _load_timewindows(fixed_file, name_to_id)
    agent_mapping = agent_mapping or {}

    vehicles: dict[int, Vehicle] = {}
    for idx, row in df_vehicles.iterrows():
        v_name = row["K"]
        v_type = int(row["type"])
        origin_id = origin_map.get(v_name, -1)
        destination_id = destination_map.get(v_name, -1)
        timetable = timewindows.get(v_name) if v_type != 3 else None

        vehicles[idx] = Vehicle(
            id=idx,
            name=v_name,
            type=v_type,
            capacity=int(row["u"]),
            speed=float(row["speed"]),
            origin_depot=origin_id,
            destination_depot=destination_id,
            km_cost=float(row["c2"]),
            h_cost=float(row["c1"]),
            carbon_cost=float(row["c5"]),
            load_cost=cost_config.load_cost.get(v_type, 0.0),
            storage_cost=cost_config.storage_cost,
            waiting_cost=cost_config.waiting_cost,
            carbon_emission=cost_config.carbon_emission,
            timetable=timetable,
            agent_id=agent_mapping.get((origin_id, destination_id)),
            fixed_transportation=bool(row.get("fixed_transportation", v_type != 3)),
        )
    return vehicles


def _load_timewindows(
    file_path: Path,
    name_to_id: dict[str, int],
) -> dict[str, dict[int, tuple[float, float]]]:
    df_fixed = pd.read_excel(file_path, sheet_name="FixedK")
    fixed_names = df_fixed["FixedK"].tolist()

    timewindows: dict[str, dict[int, tuple[float, float]]] = {}
    for v_name in fixed_names:
        try:
            df_tw = pd.read_excel(file_path, sheet_name=v_name)
        except Exception:
            continue

        tw: dict[int, tuple[float, float]] = {}
        for _, row in df_tw.iterrows():
            terminal_name = row["p"]
            if terminal_name in name_to_id:
                tw[name_to_id[terminal_name]] = (float(row["ap"]), float(row["bp"]))
        timewindows[v_name] = tw
    return timewindows


def load_requests(
    file_path: Path,
    num_requests: int,
    name_to_id: dict[str, int],
    cost_config: CostConfig,
) -> dict[int, Request]:
    sheet_name = f"R_{num_requests}"
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    requests: dict[int, Request] = {}
    for idx, row in df.iterrows():
        pickup = _resolve_node(row["p"], name_to_id)
        delivery = _resolve_node(row["d"], name_to_id)
        pickup_tw = (float(row["ap"]), float(row["bp"]))
        delivery_tw = (float(row["ad"]), float(row["bd"]))

        tw_length = delivery_tw[1] - pickup_tw[0]
        if tw_length < cost_config.delay_threshold_severe:
            delay_cost = cost_config.delay_cost["severe"]
        elif tw_length < cost_config.delay_threshold_moderate:
            delay_cost = cost_config.delay_cost["moderate"]
        else:
            delay_cost = cost_config.delay_cost["minor"]

        requests[idx] = Request(
            id=idx,
            pickup_node=pickup,
            delivery_node=delivery,
            pickup_time_window=pickup_tw,
            delivery_time_window=delivery_tw,
            demand=float(row["qr"]),
            delay_cost=float(delay_cost),
        )
    return requests


def _resolve_node(value, name_to_id: dict[str, int]) -> int:
    if isinstance(value, str):
        return name_to_id[value]
    return int(value) - 1


def load_distances(file_path: Path, num_terminals: int | None = None) -> dict[int, np.ndarray]:
    distances = {
        1: pd.read_excel(file_path, sheet_name="Barge", index_col=0).values.astype(float),
        2: pd.read_excel(file_path, sheet_name="Train", index_col=0).values.astype(float),
        3: pd.read_excel(file_path, sheet_name="Truck", index_col=0).values.astype(float),
    }

    if num_terminals is not None:
        for v_type, matrix in distances.items():
            if matrix.shape != (num_terminals, num_terminals):
                raise ValueError(f"Distance matrix shape mismatch for mode {v_type}: {matrix.shape}")
    return distances


def build_problem_instance(config: DataConfig, cost_config: CostConfig) -> ProblemInstance:
    data_dir = Path(config.data_dir)
    main_file = _resolve_existing(
        data_dir,
        config.main_file,
        ("Intermodal_EGS_data.xlsx", "Intermodal_EGS_data_all.xlsx"),
    )
    distance_file = _resolve_existing(data_dir, config.distance_file, ("Distance_EGS.xlsx",))
    fixed_file = _resolve_existing(data_dir, config.fixed_file, ("Fixed_vehicles.xlsx",))
    agent_file = data_dir / config.agent_file

    terminals, name_to_id = load_terminals(main_file, config)
    agent_mapping = load_agent_mapping(agent_file, name_to_id)
    vehicles = load_vehicles(main_file, fixed_file, name_to_id, cost_config, agent_mapping)
    requests = load_requests(main_file, config.num_requests, name_to_id, cost_config)
    distances = load_distances(distance_file, len(terminals))
    transfer_nodes = {t.id for t in terminals.values() if t.is_transfer}

    _validate(terminals, vehicles, requests, distances)

    return ProblemInstance(
        terminals=terminals,
        vehicles=vehicles,
        requests=requests,
        distances=distances,
        transfer_nodes=transfer_nodes,
        loading_time=cost_config.loading_time,
        agent_mapping=agent_mapping,
    )


def _validate(
    terminals: dict[int, Terminal],
    vehicles: dict[int, Vehicle],
    requests: dict[int, Request],
    distances: dict[int, np.ndarray],
) -> None:
    if not terminals:
        raise ValueError("No terminals loaded")
    if not vehicles:
        raise ValueError("No vehicles loaded")
    if not requests:
        raise ValueError("No requests loaded")

    for req in requests.values():
        if req.pickup_node not in terminals:
            raise ValueError(f"Request {req.id} pickup node is unknown")
        if req.delivery_node not in terminals:
            raise ValueError(f"Request {req.id} delivery node is unknown")

    for mode, matrix in distances.items():
        expected = (len(terminals), len(terminals))
        if matrix.shape != expected:
            raise ValueError(f"Distance matrix shape mismatch for mode {mode}: {matrix.shape}")

