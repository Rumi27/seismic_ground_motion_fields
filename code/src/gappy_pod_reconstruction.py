"""Gappy POD reconstruction from sparse station observations (Mw7, low-frequency).

Reconstructs held-out station wavefields from a POD basis trained on the training
ensemble, using only a subset of observed stations at each experiment setting.

Example
-------
    python -m src.gappy_pod_reconstruction
    python -m src.gappy_pod_reconstruction --ranks 50 100 --n-observed 8 16 32
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .balanced_pod_analysis import FREQUENCY_BANDS
from .config import DATA_PROCESSED, RESULTS_TABLES
from .train_lowfreq_opinf import (
    DEFAULT_BAND,
    DEFAULT_EVENT_CLASS,
    assemble_snapshot_matrix,
    compute_pod_basis,
    discover_processed_events,
    load_bandpassed_snapshots,
    split_records,
)


N_COMPONENTS = 3
DEFAULT_RANKS = [50, 100, 151]
DEFAULT_N_OBSERVED = [8, 16, 32, 64, 128]


def station_dof_indices(station_ids: np.ndarray, n_components: int = N_COMPONENTS) -> np.ndarray:
    """Map station indices to flattened snapshot row indices."""
    idx = []
    for station in station_ids:
        base = int(station) * n_components
        idx.extend(range(base, base + n_components))
    return np.asarray(idx, dtype=np.int64)


def n_stations_from_snapshots(snapshots: np.ndarray) -> int:
    if snapshots.shape[0] % N_COMPONENTS != 0:
        raise ValueError(f"Snapshot rows {snapshots.shape[0]} not divisible by {N_COMPONENTS}")
    return snapshots.shape[0] // N_COMPONENTS


def solve_gappy_coefficients(
    basis: np.ndarray,
    snapshots: np.ndarray,
    observed_stations: np.ndarray,
    regularization: float,
) -> np.ndarray:
    """Return POD coefficients Q (rank, n_time) from observed station data."""
    n_stations = n_stations_from_snapshots(snapshots)
    obs_idx = station_dof_indices(observed_stations)
    hidden_stations = np.setdiff1d(
        np.arange(n_stations), observed_stations, assume_unique=True
    )
    if hidden_stations.size == 0:
        raise ValueError("No held-out stations remain")

    v_obs = basis[obs_idx].astype(np.float64)
    x_obs = snapshots[obs_idx].astype(np.float64)
    gram = v_obs.T @ v_obs
    reg = regularization * np.eye(gram.shape[0], dtype=np.float64)
    coefficients = np.linalg.solve(gram + reg, v_obs.T @ x_obs)
    return coefficients.astype(np.float32), hidden_stations


def reconstruct_from_coefficients(basis: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return (basis @ coefficients).astype(np.float32, copy=False)


def relative_frobenius_error(reference: np.ndarray, approximation: np.ndarray) -> float:
    num = np.linalg.norm(reference - approximation)
    den = np.linalg.norm(reference)
    return float(num / den) if den > 0 else float("nan")


def pgv_from_snapshots(snapshots: np.ndarray, dt: float) -> np.ndarray:
    """Per-station pseudo-PGV: max |du/dt| over time (from displacement)."""
    n_stations = n_stations_from_snapshots(snapshots)
    arr = snapshots.reshape(N_COMPONENTS, n_stations, snapshots.shape[1])
    vel = np.diff(arr, axis=2) / dt
    horiz = np.sqrt(vel[0] ** 2 + vel[1] ** 2)
    return np.max(horiz, axis=1)


def evaluate_event(
    basis: np.ndarray,
    snapshots: np.ndarray,
    observed_stations: np.ndarray,
    dt: float,
    regularization: float,
) -> dict:
    coefficients, hidden_stations = solve_gappy_coefficients(
        basis, snapshots, observed_stations, regularization=regularization
    )
    reconstruction = reconstruct_from_coefficients(basis, coefficients)

    obs_idx = station_dof_indices(observed_stations)
    hid_idx = station_dof_indices(hidden_stations)

    rel_obs = relative_frobenius_error(snapshots[obs_idx], reconstruction[obs_idx])
    rel_hid = relative_frobenius_error(snapshots[hid_idx], reconstruction[hid_idx])
    rel_full = relative_frobenius_error(snapshots, reconstruction)

    pgv_true = pgv_from_snapshots(snapshots, dt=dt)
    pgv_pred = pgv_from_snapshots(reconstruction, dt=dt)
    pgv_hid_err = relative_frobenius_error(
        pgv_true[hidden_stations], pgv_pred[hidden_stations]
    )

    return {
        "relative_error_observed": rel_obs,
        "relative_error_heldout": rel_hid,
        "relative_error_full": rel_full,
        "pgv_error_heldout": pgv_hid_err,
    }


def run_experiment(
    data_root: Path,
    event_class: str,
    band_name: str,
    ranks: list[int],
    n_observed_list: list[int],
    train_fraction: float,
    seed: int,
    max_events: int | None,
    regularization: float,
) -> Path:
    band = FREQUENCY_BANDS[band_name]
    records = discover_processed_events(data_root, event_class=event_class)
    if max_events is not None:
        records = records[:max_events]
    train_records, val_records = split_records(
        records, train_fraction=train_fraction, seed=seed
    )
    n_stations = n_stations_from_snapshots(
        load_bandpassed_snapshots(val_records[0], band=band)
    )

    rows = []
    for rank in ranks:
        train_matrix = assemble_snapshot_matrix(train_records, band=band)
        basis, _ = compute_pod_basis(train_matrix, rank=rank)

        for n_observed in n_observed_list:
            if n_observed >= n_stations:
                print(f"skip n_observed={n_observed} (only {n_stations} stations)")
                continue

            rng = np.random.default_rng(seed + n_observed)
            metrics = []
            for record in val_records:
                snapshots = load_bandpassed_snapshots(record, band=band)
                observed = np.sort(rng.choice(n_stations, size=n_observed, replace=False))
                metrics.append(
                    evaluate_event(
                        basis=basis,
                        snapshots=snapshots,
                        observed_stations=observed,
                        dt=record.dt,
                        regularization=regularization,
                    )
                )

            rows.append(
                {
                    "class": event_class,
                    "frequency_band": band_name,
                    "rank": rank,
                    "n_observed_stations": n_observed,
                    "n_heldout_stations": n_stations - n_observed,
                    "n_val_events": len(val_records),
                    "mean_rel_error_observed": float(
                        np.mean([m["relative_error_observed"] for m in metrics])
                    ),
                    "mean_rel_error_heldout": float(
                        np.mean([m["relative_error_heldout"] for m in metrics])
                    ),
                    "mean_rel_error_full": float(
                        np.mean([m["relative_error_full"] for m in metrics])
                    ),
                    "mean_pgv_error_heldout": float(
                        np.mean([m["pgv_error_heldout"] for m in metrics])
                    ),
                }
            )
            print(
                f"rank={rank} n_obs={n_observed}: "
                f"heldout={rows[-1]['mean_rel_error_heldout']:.3f} "
                f"observed={rows[-1]['mean_rel_error_observed']:.3f}"
            )

    out_path = RESULTS_TABLES / "gappy_pod_mw7_lowfreq.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_PROCESSED / "events")
    parser.add_argument("--event-class", default=DEFAULT_EVENT_CLASS)
    parser.add_argument("--band", choices=sorted(FREQUENCY_BANDS), default=DEFAULT_BAND)
    parser.add_argument("--ranks", type=int, nargs="+", default=DEFAULT_RANKS)
    parser.add_argument("--n-observed", type=int, nargs="+", default=DEFAULT_N_OBSERVED)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-events", type=int, default=100)
    parser.add_argument("--regularization", type=float, default=1e-8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        data_root=args.data_root,
        event_class=args.event_class,
        band_name=args.band,
        ranks=args.ranks,
        n_observed_list=args.n_observed,
        train_fraction=args.train_fraction,
        seed=args.seed,
        max_events=args.max_events,
        regularization=args.regularization,
    )


if __name__ == "__main__":
    main()
