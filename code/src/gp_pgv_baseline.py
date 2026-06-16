"""Gaussian Process (kriging) PGV baseline for sparse station reconstruction.

Fits a Matérn-3/2 GP on scalar PGV at observed stations and predicts PGV at
all held-out stations.  Results are directly comparable to the gappy POD PGV
error reported in Table 3 of the manuscript.

The GP is fitted and evaluated on the same event-normalised PGV values used
throughout the gappy POD experiments (same train/val split, same seeds).

Usage
-----
    python -m src.gp_pgv_baseline
    python -m src.gp_pgv_baseline --n-observed 128 --n-seeds 10 --max-events 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .balanced_pod_analysis import FREQUENCY_BANDS
from .config import DATA_PROCESSED, DATA_RAW, RESULTS_TABLES
from .gappy_pod_reconstruction import (
    DEFAULT_EVENT_CLASS,
    N_COMPONENTS,
    find_reference_h5,
    load_station_coordinates,
    pgv_from_snapshots,
    relative_frobenius_error,
    select_stations_random,
)
from .train_lowfreq_opinf import (
    DEFAULT_BAND,
    DEFAULT_DT_SECONDS,
    DEFAULT_EVENT_CLASS,
    discover_processed_events,
    load_bandpassed_snapshots,
    split_records,
)

_BATCH = 8000  # prediction batch size to bound peak memory


def _gp_predict_pgv(
    coords_obs: np.ndarray,
    pgv_obs: np.ndarray,
    coords_hidden: np.ndarray,
) -> np.ndarray:
    """Fit Matérn-3/2 GP on observed PGV, predict at hidden stations.

    Coordinates are normalised internally; PGV is log-transformed to enforce
    positivity and then back-transformed.
    """
    # Normalise coordinates to unit scale
    mu_c = coords_obs.mean(axis=0)
    sd_c = coords_obs.std(axis=0).clip(1e-8)
    X_obs = (coords_obs - mu_c) / sd_c
    X_hid = (coords_hidden - mu_c) / sd_c

    # Log-transform PGV (always positive; stabilises variance)
    log_pgv = np.log(pgv_obs.clip(1e-12))

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=1.0, length_scale_bounds=(0.05, 20.0), nu=1.5)
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1.0))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=3,
        normalize_y=True,
        alpha=0.0,
    )
    gp.fit(X_obs, log_pgv)

    # Predict in batches to bound memory for large hidden sets
    log_pred = np.empty(len(X_hid))
    for start in range(0, len(X_hid), _BATCH):
        log_pred[start : start + _BATCH] = gp.predict(X_hid[start : start + _BATCH])

    return np.exp(log_pred)


def run(
    data_root: Path,
    raw_root: Path,
    band_name: str,
    n_observed: int,
    seeds: list[int],
    train_fraction: float,
    max_events: int,
    output_csv: Path,
) -> None:
    band = FREQUENCY_BANDS[band_name]
    records = discover_processed_events(data_root, event_class=DEFAULT_EVENT_CLASS)
    if max_events:
        records = records[:max_events]

    # Fixed train/val split — same as gappy POD experiments
    train_records, val_records = split_records(
        records, train_fraction=train_fraction, seed=42
    )
    print(f"Val events: {len(val_records)}")

    ref_h5 = find_reference_h5(DEFAULT_EVENT_CLASS, raw_root=raw_root)
    coords = load_station_coordinates(ref_h5)          # (n_stations, 2)
    n_stations = len(coords)

    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        obs_stations = select_stations_random(n_stations, n_observed, rng)
        hidden_mask = np.ones(n_stations, dtype=bool)
        hidden_mask[obs_stations] = False
        hidden_stations = np.where(hidden_mask)[0]

        coords_obs = coords[obs_stations]
        coords_hid = coords[hidden_stations]

        pgv_errors = []
        for rec in val_records:
            snaps = load_bandpassed_snapshots(rec, band=band)
            pgv_ref = pgv_from_snapshots(snaps, dt=DEFAULT_DT_SECONDS)   # (n_stations,)

            pgv_obs_vals = pgv_ref[obs_stations]
            pgv_hid_ref  = pgv_ref[hidden_stations]

            pgv_hid_gp = _gp_predict_pgv(coords_obs, pgv_obs_vals, coords_hid)
            err = relative_frobenius_error(pgv_hid_ref, pgv_hid_gp)
            pgv_errors.append(err)

        mean_err = float(np.mean(pgv_errors))
        print(f"  seed={seed}: GP PGV error = {mean_err:.4f}  (n={len(pgv_errors)} events)")
        rows.append({"seed": seed, "n_observed": n_observed, "gp_pgv_error": mean_err})

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\nGP PGV error  mean={df['gp_pgv_error'].mean():.4f}  "
          f"std={df['gp_pgv_error'].std():.4f}")
    print(f"Saved: {output_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DATA_PROCESSED / "events")
    p.add_argument("--raw-root",  type=Path, default=DATA_RAW)
    p.add_argument("--band", choices=sorted(FREQUENCY_BANDS), default=DEFAULT_BAND)
    p.add_argument("--n-observed", type=int, default=128)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--max-events", type=int, default=100)
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument("--output-csv", type=Path,
                   default=RESULTS_TABLES / "gp_pgv_baseline_random_128.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = list(range(args.n_seeds))
    run(
        data_root=args.data_root,
        raw_root=args.raw_root,
        band_name=args.band,
        n_observed=args.n_observed,
        seeds=seeds,
        train_fraction=args.train_fraction,
        max_events=args.max_events,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
