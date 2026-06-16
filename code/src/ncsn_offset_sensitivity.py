"""NCSN mapping-offset sensitivity analysis.

Runs gappy POD reconstruction with three filtered subsets of the NCSN-derived
station layout:

  all        All 144 mapped stations (original, mean offset 9.9 km)
  le20km     Stations with mapping offset <= 20 km  (105 stations)
  le10km     Stations with mapping offset <= 10 km  (85 stations)

Each filtered set is tested at its full size and at n_observed = 64 (the
largest count feasible for all three sets).  A matched random layout is also
run at each n_observed for direct comparison.

For each configuration the script reports mean relative waveform error and
mean relative PGV error across all held-out validation events (rank = 151,
default regularisation).

Results are written to:
    results/tables/ncsn_offset_sensitivity.csv

Usage
-----
    python -m src.ncsn_offset_sensitivity
    python -m src.ncsn_offset_sensitivity --max-events 40
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .balanced_pod_analysis import FREQUENCY_BANDS
from .config import DATA_PROCESSED, RESULTS_TABLES
from .gappy_pod_reconstruction import (
    N_COMPONENTS,
    evaluate_event,
    n_stations_from_snapshots,
    station_dof_indices,
    solve_gappy_coefficients,
    reconstruct_from_coefficients,
    relative_frobenius_error,
    pgv_from_snapshots,
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

DEFAULT_RANK = 151
DEFAULT_REG = 1e-8
MAPPING_CSV = DATA_PROCESSED / "ncsn_station_mapping.csv"

# Common n_observed feasible for all three filtered sets (<=10 km has 85 stations)
N_COMMON = 64


def _load_filtered_indices(mapping_csv: Path, max_offset_km: float | None) -> np.ndarray:
    """Return unique nearest_station_index values, optionally filtered by offset."""
    df = pd.read_csv(mapping_csv)
    if max_offset_km is not None:
        df = df[df["distance_km"] <= max_offset_km]
    indices = (
        df["nearest_station_index"]
        .dropna()
        .astype(np.int64)
        .drop_duplicates()
        .to_numpy()
    )
    return indices


def _run_layout(
    label: str,
    obs_stations: np.ndarray,
    basis: np.ndarray,
    val_records: list,
    band,
    dt: float,
    regularization: float,
    n_stations: int,
) -> dict:
    """Run gappy POD for a fixed station set across all val events."""
    n_obs = len(obs_stations)
    hid_stations = np.setdiff1d(np.arange(n_stations), obs_stations)

    wf_errors, pgv_errors = [], []
    for record in val_records:
        snapshots = load_bandpassed_snapshots(record, band=band)
        coefficients, _ = solve_gappy_coefficients(
            basis, snapshots, obs_stations, regularization
        )
        recon = reconstruct_from_coefficients(basis, coefficients)
        hid_idx = station_dof_indices(hid_stations)
        wf_errors.append(relative_frobenius_error(snapshots[hid_idx], recon[hid_idx]))
        pgv_t = pgv_from_snapshots(snapshots, dt=record.dt)
        pgv_r = pgv_from_snapshots(recon, dt=record.dt)
        pgv_errors.append(relative_frobenius_error(pgv_t[hid_stations], pgv_r[hid_stations]))

    return {
        "layout": label,
        "n_observed": n_obs,
        "n_heldout": n_stations - n_obs,
        "n_val_events": len(val_records),
        "mean_wf_error_heldout": float(np.mean(wf_errors)),
        "mean_pgv_error_heldout": float(np.mean(pgv_errors)),
    }


def run(
    data_root: Path,
    mapping_csv: Path,
    rank: int,
    train_fraction: float,
    seed: int,
    max_events: int,
    regularization: float,
    output_csv: Path,
) -> None:
    band = FREQUENCY_BANDS[DEFAULT_BAND]
    records = discover_processed_events(data_root, event_class=DEFAULT_EVENT_CLASS)
    if max_events:
        records = records[:max_events]
    train_records, val_records = split_records(
        records, train_fraction=train_fraction, seed=seed
    )
    print(f"Train: {len(train_records)}  Val: {len(val_records)}")

    train_matrix = assemble_snapshot_matrix(train_records, band=band)
    basis, _ = compute_pod_basis(train_matrix, rank=rank)
    n_stations = train_matrix.shape[0] // N_COMPONENTS

    dt = train_records[0].dt

    # Load all three filtered index sets
    idx_all  = _load_filtered_indices(mapping_csv, None)
    idx_le20 = _load_filtered_indices(mapping_csv, 20.0)
    idx_le10 = _load_filtered_indices(mapping_csv, 10.0)

    # Clip to valid domain
    idx_all  = idx_all [(idx_all  >= 0) & (idx_all  < n_stations)]
    idx_le20 = idx_le20[(idx_le20 >= 0) & (idx_le20 < n_stations)]
    idx_le10 = idx_le10[(idx_le10 >= 0) & (idx_le10 < n_stations)]

    print(f"NCSN station counts — all: {len(idx_all)},  <=20 km: {len(idx_le20)},  <=10 km: {len(idx_le10)}")

    rng = np.random.default_rng(seed)

    configs: list[tuple[str, np.ndarray]] = [
        # Full-size sets
        (f"ncsn_all_{len(idx_all)}",   np.sort(idx_all)),
        (f"ncsn_le20_{len(idx_le20)}", np.sort(idx_le20)),
        (f"ncsn_le10_{len(idx_le10)}", np.sort(idx_le10)),
        # Random matched baselines at full sizes
        (f"random_{len(idx_all)}",
         np.sort(rng.choice(n_stations, size=len(idx_all),  replace=False))),
        (f"random_{len(idx_le20)}",
         np.sort(rng.choice(n_stations, size=len(idx_le20), replace=False))),
        (f"random_{len(idx_le10)}",
         np.sort(rng.choice(n_stations, size=len(idx_le10), replace=False))),
        # Common N_COMMON for fair cross-set comparison
        (f"ncsn_all_{N_COMMON}",   np.sort(idx_all [:N_COMMON])),
        (f"ncsn_le20_{N_COMMON}",  np.sort(idx_le20[:N_COMMON])),
        (f"ncsn_le10_{N_COMMON}",  np.sort(idx_le10[:N_COMMON])),
        (f"random_{N_COMMON}",
         np.sort(rng.choice(n_stations, size=N_COMMON, replace=False))),
    ]

    rows = []
    for label, obs in configs:
        print(f"  Running {label} ({len(obs)} stations) …", flush=True)
        row = _run_layout(
            label=label,
            obs_stations=obs,
            basis=basis,
            val_records=val_records,
            band=band,
            dt=dt,
            regularization=regularization,
            n_stations=n_stations,
        )
        rows.append(row)
        print(f"    wf={row['mean_wf_error_heldout']:.4f}  "
              f"pgv={row['mean_pgv_error_heldout']:.4f}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved → {output_csv}")

    # Pretty summary
    print(f"\n{'Layout':<25} {'n_obs':>6} {'wf_err':>8} {'pgv_err':>8}")
    print("-" * 52)
    for row in rows:
        print(f"{row['layout']:<25} {row['n_observed']:>6} "
              f"{row['mean_wf_error_heldout']:>8.4f} "
              f"{row['mean_pgv_error_heldout']:>8.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",     type=Path, default=DATA_PROCESSED / "events")
    p.add_argument("--mapping-csv",   type=Path, default=MAPPING_CSV)
    p.add_argument("--rank",          type=int,  default=DEFAULT_RANK)
    p.add_argument("--train-fraction",type=float,default=0.8)
    p.add_argument("--seed",          type=int,  default=42)
    p.add_argument("--max-events",    type=int,  default=100)
    p.add_argument("--regularization",type=float,default=DEFAULT_REG)
    p.add_argument("--output-csv",    type=Path,
                   default=RESULTS_TABLES / "ncsn_offset_sensitivity.csv")
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
