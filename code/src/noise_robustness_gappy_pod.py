"""Noise-robustness study for gappy POD reconstruction (Mw7, low-frequency).

Injects Gaussian noise at SNR-representative delta levels into observed station
waveforms and evaluates waveform/PGV reconstruction error for:
  - Layout A: random, seed 42, 128 stations, rank 151
  - Layout B: NCSN-real geometry, 128 stations, rank 151

For each noise level delta (fraction of observed-station RMS) the script tests:
  - Fixed regularization lambda = 1e-8 (matches clean-data optimum)
  - Discrepancy-principle lambda: lambda_dp = delta^2 * rms_obs^2

Ten noise realizations per event are drawn so error statistics are stable.

Usage
-----
    python -m src.noise_robustness_gappy_pod
    python -m src.noise_robustness_gappy_pod --n-noise-seeds 5 --max-events 50
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .balanced_pod_analysis import FREQUENCY_BANDS
from .config import DATA_PROCESSED, DATA_RAW, RESULTS_TABLES
from .gappy_pod_reconstruction import (
    DEFAULT_STATION_MAPPING,
    N_COMPONENTS,
    find_reference_h5,
    load_bandpassed_snapshots,
    load_mapped_station_indices,
    load_station_coordinates,
    n_stations_from_snapshots,
    pgv_from_snapshots,
    reconstruct_from_coefficients,
    relative_frobenius_error,
    select_stations_ncsn,
    select_stations_random,
    station_dof_indices,
)
from .train_lowfreq_opinf import (
    DEFAULT_BAND,
    DEFAULT_EVENT_CLASS,
    assemble_snapshot_matrix,
    compute_pod_basis,
    discover_processed_events,
    split_records,
)


DELTA_LEVELS = [0.0, 0.001, 0.01, 0.05]
DEFAULT_RANK = 151
DEFAULT_N_OBSERVED = 128
DEFAULT_LAMBDA_FIXED = 1e-8
DEFAULT_N_NOISE_SEEDS = 10


def solve_gappy_noisy(
    basis: np.ndarray,
    x_obs_noisy: np.ndarray,
    observed_stations: np.ndarray,
    n_stations: int,
    regularization: float,
) -> np.ndarray:
    """Solve gappy coefficients from (possibly noisy) observed snapshots."""
    obs_idx = station_dof_indices(observed_stations, n_stations=n_stations)
    v_obs = basis[obs_idx].astype(np.float64)
    x = x_obs_noisy[obs_idx].astype(np.float64)
    gram = v_obs.T @ v_obs
    reg = regularization * np.eye(gram.shape[0], dtype=np.float64)
    return np.linalg.solve(gram + reg, v_obs.T @ x).astype(np.float32)


def lambda_dp_for_event(
    x_obs: np.ndarray,
    delta: float,
    n_obs_dofs: int,
) -> float:
    """Discrepancy-principle Tikhonov parameter.

    For a field normalised to unit RMS, the optimal lambda scales as delta^2.
    We scale by the per-DOF observed-field variance to handle non-unit RMS data.
    """
    if delta == 0.0:
        return DEFAULT_LAMBDA_FIXED
    rms_per_dof_sq = float(np.mean(x_obs.astype(np.float64) ** 2))
    return delta**2 * rms_per_dof_sq


def evaluate_noise_event(
    basis: np.ndarray,
    snapshots: np.ndarray,
    observed_stations: np.ndarray,
    dt: float,
    delta: float,
    lambda_fixed: float,
    n_noise_seeds: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Reconstruct one event under noise and return mean errors over realisations."""
    n_stations = n_stations_from_snapshots(snapshots)
    hidden_stations = np.setdiff1d(np.arange(n_stations), observed_stations)
    obs_idx = station_dof_indices(observed_stations, n_stations=n_stations)
    hid_idx = station_dof_indices(hidden_stations, n_stations=n_stations)

    x_obs_clean = snapshots[obs_idx].astype(np.float64)
    rms_obs = float(np.sqrt(np.mean(x_obs_clean**2)))
    noise_std = delta * rms_obs

    pgv_true = pgv_from_snapshots(snapshots, dt=dt)
    lam_dp = lambda_dp_for_event(x_obs_clean, delta=delta, n_obs_dofs=obs_idx.size)

    err_wf_fixed_list: list[float] = []
    err_pgv_fixed_list: list[float] = []
    err_wf_dp_list: list[float] = []
    err_pgv_dp_list: list[float] = []

    for _ in range(n_noise_seeds):
        if delta == 0.0:
            x_obs_noisy = snapshots
        else:
            noise = rng.normal(0.0, noise_std, snapshots.shape).astype(np.float32)
            x_obs_noisy = snapshots + noise

        coeffs_fixed = solve_gappy_noisy(
            basis, x_obs_noisy, observed_stations, n_stations, lambda_fixed
        )
        recon_fixed = reconstruct_from_coefficients(basis, coeffs_fixed)
        err_wf_fixed_list.append(
            relative_frobenius_error(snapshots[hid_idx], recon_fixed[hid_idx])
        )
        err_pgv_fixed_list.append(
            relative_frobenius_error(
                pgv_true[hidden_stations],
                pgv_from_snapshots(recon_fixed, dt=dt)[hidden_stations],
            )
        )

        coeffs_dp = solve_gappy_noisy(
            basis, x_obs_noisy, observed_stations, n_stations, lam_dp
        )
        recon_dp = reconstruct_from_coefficients(basis, coeffs_dp)
        err_wf_dp_list.append(
            relative_frobenius_error(snapshots[hid_idx], recon_dp[hid_idx])
        )
        err_pgv_dp_list.append(
            relative_frobenius_error(
                pgv_true[hidden_stations],
                pgv_from_snapshots(recon_dp, dt=dt)[hidden_stations],
            )
        )

        if delta == 0.0:
            break

    return {
        "wf_fixed": float(np.mean(err_wf_fixed_list)),
        "pgv_fixed": float(np.mean(err_pgv_fixed_list)),
        "wf_dp": float(np.mean(err_wf_dp_list)),
        "pgv_dp": float(np.mean(err_pgv_dp_list)),
        "lambda_dp": lam_dp,
        "noise_std": noise_std,
        "rms_obs": rms_obs,
    }


def run_noise_study(
    data_root: Path,
    event_class: str,
    band_name: str,
    rank: int,
    n_observed: int,
    train_fraction: float,
    seed: int,
    max_events: int | None,
    lambda_fixed: float,
    n_noise_seeds: int,
    output_csv: Path,
    raw_root: Path,
    station_mapping_csv: Path,
    delta_levels: list[float] = DELTA_LEVELS,
) -> Path:
    band = FREQUENCY_BANDS[band_name]
    records = discover_processed_events(data_root, event_class=event_class)
    if max_events is not None:
        records = records[:max_events]
    train_records, val_records = split_records(records, train_fraction=train_fraction, seed=seed)

    coords = load_station_coordinates(find_reference_h5(event_class, raw_root=raw_root))
    n_stations = coords.shape[0]

    train_matrix = assemble_snapshot_matrix(train_records, band=band)
    full_basis, _ = compute_pod_basis(train_matrix, rank=rank)
    basis = full_basis[:, :rank]

    mapped_station_indices = load_mapped_station_indices(station_mapping_csv)

    rng_station = np.random.default_rng(seed + n_observed)
    random_stations = select_stations_random(n_stations, n_observed, rng_station)
    ncsn_stations = select_stations_ncsn(mapped_station_indices, n_observed, n_stations)

    layouts = [
        ("random", random_stations),
        ("ncsn", ncsn_stations),
    ]

    val_snapshots = [
        (record, load_bandpassed_snapshots(record, band=band)) for record in val_records
    ]

    rows: list[dict] = []
    for delta in delta_levels:
        for layout_name, observed_stations in layouts:
            rng_noise = np.random.default_rng(seed=7919 + int(delta * 1e4))
            event_results: list[dict[str, float]] = []
            for record, snapshots in val_snapshots:
                res = evaluate_noise_event(
                    basis=basis,
                    snapshots=snapshots,
                    observed_stations=observed_stations,
                    dt=record.dt,
                    delta=delta,
                    lambda_fixed=lambda_fixed,
                    n_noise_seeds=n_noise_seeds,
                    rng=rng_noise,
                )
                event_results.append(res)

            mean_lam_dp = float(np.mean([r["lambda_dp"] for r in event_results]))
            row = {
                "class": event_class,
                "frequency_band": band_name,
                "layout": layout_name,
                "rank": rank,
                "n_observed_stations": n_observed,
                "n_val_events": len(val_records),
                "delta": delta,
                "lambda_fixed": lambda_fixed,
                "lambda_dp": mean_lam_dp,
                "n_noise_seeds": n_noise_seeds if delta > 0 else 1,
                "mean_wf_fixed": float(np.mean([r["wf_fixed"] for r in event_results])),
                "std_wf_fixed": float(np.std([r["wf_fixed"] for r in event_results])),
                "mean_pgv_fixed": float(np.mean([r["pgv_fixed"] for r in event_results])),
                "std_pgv_fixed": float(np.std([r["pgv_fixed"] for r in event_results])),
                "mean_wf_dp": float(np.mean([r["wf_dp"] for r in event_results])),
                "std_wf_dp": float(np.std([r["wf_dp"] for r in event_results])),
                "mean_pgv_dp": float(np.mean([r["pgv_dp"] for r in event_results])),
                "std_pgv_dp": float(np.std([r["pgv_dp"] for r in event_results])),
                "mean_noise_std": float(np.mean([r["noise_std"] for r in event_results])),
                "mean_rms_obs": float(np.mean([r["rms_obs"] for r in event_results])),
            }
            rows.append(row)
            print(
                f"delta={delta:.3f} layout={layout_name}: "
                f"wf_fixed={row['mean_wf_fixed']:.3f} wf_dp={row['mean_wf_dp']:.3f} "
                f"pgv_fixed={row['mean_pgv_fixed']:.3f} pgv_dp={row['mean_pgv_dp']:.3f} "
                f"lam_dp={mean_lam_dp:.2e}"
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {output_csv}")
    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_PROCESSED / "events")
    parser.add_argument("--raw-root", type=Path, default=DATA_RAW)
    parser.add_argument("--event-class", default=DEFAULT_EVENT_CLASS)
    parser.add_argument("--band", choices=sorted(FREQUENCY_BANDS), default=DEFAULT_BAND)
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--n-observed", type=int, default=DEFAULT_N_OBSERVED)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-events", type=int, default=100)
    parser.add_argument("--lambda-fixed", type=float, default=DEFAULT_LAMBDA_FIXED)
    parser.add_argument("--n-noise-seeds", type=int, default=DEFAULT_N_NOISE_SEEDS)
    parser.add_argument(
        "--station-mapping-csv",
        type=Path,
        default=DEFAULT_STATION_MAPPING,
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=RESULTS_TABLES / "noise_robustness_gappy_pod_mw7.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_noise_study(
        data_root=args.data_root,
        event_class=args.event_class,
        band_name=args.band,
        rank=args.rank,
        n_observed=args.n_observed,
        train_fraction=args.train_fraction,
        seed=args.seed,
        max_events=args.max_events,
        lambda_fixed=args.lambda_fixed,
        n_noise_seeds=args.n_noise_seeds,
        output_csv=args.output_csv,
        raw_root=args.raw_root,
        station_mapping_csv=args.station_mapping_csv,
    )


if __name__ == "__main__":
    main()
