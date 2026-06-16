"""Station-count sweep for gappy POD reconstruction (rank 151, low-frequency).

Runs four station layouts across n_observed ∈ {32, 64, 96, 128, 192, 256, 384, 512}:
  random    — uniform random, 5 seeds each
  farthest  — farthest-point greedy from a fixed random seed
  pod_qr    — DEIM-style QR pivoting on the POD basis
  ncsn      — real NCSN-derived station positions (max 144 stations)

Existing rank-151 results for n_obs ∈ {8, 16, 32, 64, 128} are preserved in the
multi-seed CSV; this script writes a fresh focused sweep CSV so the figure script
can read a single consistent file.

Results are written to:
    results/tables/station_count_sweep_mw7_lowfreq.csv

Usage
-----
    python -m src.station_count_sweep
    python -m src.station_count_sweep --max-events 40
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .balanced_pod_analysis import FREQUENCY_BANDS
from .config import DATA_PROCESSED, RESULTS_TABLES
from .gappy_pod_reconstruction import (
    N_COMPONENTS,
    evaluate_event,
    n_stations_from_snapshots,
)
from .plot_spatial_error_map import (
    DEFAULT_STATION_MAPPING,
    load_mapped_station_indices,
    pod_qr_station_order,
    select_stations_farthest,
    select_stations_ncsn,
    select_stations_pod_qr,
    select_stations_random,
    station_coords_km,
)
from .train_lowfreq_opinf import (
    DEFAULT_BAND,
    DEFAULT_EVENT_CLASS,
    assemble_snapshot_matrix,
    compute_pod_basis,
    discover_processed_events,
    load_bandpassed_snapshots,
    split_records,
)

N_OBS_LIST = [32, 64, 96, 128, 192, 256, 384, 512]
N_RANDOM_SEEDS = 5
DEFAULT_RANK = 151
DEFAULT_REG = 1e-8
NCSN_MAX = 144


def run(
    data_root: Path,
    mapping_csv: Path,
    rank: int,
    train_fraction: float,
    seed: int,
    max_events: int | None,
    regularization: float,
    output_csv: Path,
    n_obs_list: list[int] = N_OBS_LIST,
    n_random_seeds: int = N_RANDOM_SEEDS,
) -> None:
    band = FREQUENCY_BANDS[DEFAULT_BAND]
    records = discover_processed_events(data_root, event_class=DEFAULT_EVENT_CLASS)
    if max_events is not None:
        records = records[:max_events]
    train_records, val_records = split_records(
        records, train_fraction=train_fraction, seed=seed
    )
    print(f"Train: {len(train_records)}  Val: {len(val_records)}")

    train_matrix = assemble_snapshot_matrix(train_records, band=band)
    basis, _ = compute_pod_basis(train_matrix, rank=rank)
    basis = basis[:, :rank]
    n_stations = train_matrix.shape[0] // N_COMPONENTS

    coords = station_coords_km(n_stations)
    ncsn_indices = load_mapped_station_indices(mapping_csv)
    ncsn_valid = ncsn_indices[(ncsn_indices >= 0) & (ncsn_indices < n_stations)]

    qr_order = pod_qr_station_order(basis, n_stations)
    dt = train_records[0].dt

    val_data = [
        (rec, load_bandpassed_snapshots(rec, band=band)) for rec in val_records
    ]

    rows: list[dict] = []

    for n_obs in n_obs_list:
        if n_obs >= n_stations:
            print(f"skip n_obs={n_obs} (only {n_stations} stations total)")
            continue

        # --- Random: multiple seeds ---
        for s in range(n_random_seeds):
            rng = np.random.default_rng(seed * 1000 + s * 17 + n_obs)
            obs = select_stations_random(n_stations, n_obs, rng)
            metrics = [
                evaluate_event(basis, snaps, obs, dt=rec.dt, regularization=regularization)
                for rec, snaps in val_data
            ]
            rows.append(_aggregate("random", s, n_obs, n_stations, val_records, metrics))
            print(
                f"n_obs={n_obs:4d}  random seed={s}  "
                f"wf={rows[-1]['mean_rel_error_heldout']:.4f}  "
                f"pgv={rows[-1]['mean_pgv_error_heldout']:.4f}",
                flush=True,
            )

        # --- Farthest-point ---
        rng_fp = np.random.default_rng(seed + 9999)
        obs_fp = select_stations_farthest(coords, n_obs, rng_fp)
        metrics_fp = [
            evaluate_event(basis, snaps, obs_fp, dt=rec.dt, regularization=regularization)
            for rec, snaps in val_data
        ]
        rows.append(_aggregate("farthest", 0, n_obs, n_stations, val_records, metrics_fp))
        print(
            f"n_obs={n_obs:4d}  farthest          "
            f"wf={rows[-1]['mean_rel_error_heldout']:.4f}  "
            f"pgv={rows[-1]['mean_pgv_error_heldout']:.4f}",
            flush=True,
        )

        # --- POD-QR ---
        obs_qr = select_stations_pod_qr(qr_order, n_obs, n_stations)
        metrics_qr = [
            evaluate_event(basis, snaps, obs_qr, dt=rec.dt, regularization=regularization)
            for rec, snaps in val_data
        ]
        rows.append(_aggregate("pod_qr", 0, n_obs, n_stations, val_records, metrics_qr))
        print(
            f"n_obs={n_obs:4d}  pod_qr            "
            f"wf={rows[-1]['mean_rel_error_heldout']:.4f}  "
            f"pgv={rows[-1]['mean_pgv_error_heldout']:.4f}",
            flush=True,
        )

        # --- NCSN (only up to 144 stations) ---
        ncsn_n = min(n_obs, NCSN_MAX)
        if ncsn_n == n_obs:
            obs_ncsn = select_stations_ncsn(ncsn_valid, ncsn_n, n_stations)
            metrics_ncsn = [
                evaluate_event(basis, snaps, obs_ncsn, dt=rec.dt, regularization=regularization)
                for rec, snaps in val_data
            ]
            rows.append(_aggregate("ncsn", 0, n_obs, n_stations, val_records, metrics_ncsn))
            print(
                f"n_obs={n_obs:4d}  ncsn              "
                f"wf={rows[-1]['mean_rel_error_heldout']:.4f}  "
                f"pgv={rows[-1]['mean_pgv_error_heldout']:.4f}",
                flush=True,
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved → {output_csv}")


def _aggregate(
    layout: str,
    seed_used: int,
    n_obs: int,
    n_stations: int,
    val_records,
    metrics: list[dict],
) -> dict:
    return {
        "class": DEFAULT_EVENT_CLASS,
        "frequency_band": DEFAULT_BAND,
        "layout": layout,
        "seed": seed_used,
        "rank": DEFAULT_RANK,
        "n_observed_stations": n_obs,
        "n_heldout_stations": n_stations - n_obs,
        "n_val_events": len(val_records),
        "mean_rel_error_heldout": float(np.mean([m["relative_error_heldout"] for m in metrics])),
        "mean_pgv_error_heldout": float(np.mean([m["pgv_error_heldout"] for m in metrics])),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",      type=Path, default=DATA_PROCESSED / "events")
    p.add_argument("--mapping-csv",    type=Path, default=DEFAULT_STATION_MAPPING)
    p.add_argument("--rank",           type=int,  default=DEFAULT_RANK)
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument("--seed",           type=int,  default=42)
    p.add_argument("--max-events",     type=int,  default=100)
    p.add_argument("--regularization", type=float, default=DEFAULT_REG)
    p.add_argument("--output-csv",     type=Path,
                   default=RESULTS_TABLES / "station_count_sweep_mw7_lowfreq.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(
        data_root=args.data_root,
        mapping_csv=args.mapping_csv,
        rank=args.rank,
        train_fraction=args.train_fraction,
        seed=args.seed,
        max_events=args.max_events,
        regularization=args.regularization,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
