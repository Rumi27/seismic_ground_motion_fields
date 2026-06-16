"""Geostatistical GP baselines for PGV reconstruction: isotropic, anisotropic, Loth–Baker.

Three kriging variants evaluated at 128 observed stations (10 random seeds, same
train/val split as gappy POD experiments):

  iso   — Matérn-3/2, isotropic, length scale optimised (working in km)
  aniso — Matérn-3/2, geometric anisotropy aligned to Hayward Fault strike
           (fault-parallel length scale free ~20–200 km,
            fault-perpendicular free ~1–80 km)
  lb    — Loth & Baker (2013) two-component exponential kernel
           B1=0.367 exp(-h/b1) + B2=0.305 exp(-h/b2), T=1 s proxy for PGV,
           b1 and b2 free within (2,50) km and (30,500) km respectively

All variants work in km coordinates (NOT normalised) so length scales are
physically meaningful.  PGV is log-transformed before fitting to enforce
positivity and stabilise variance.

Usage
-----
    python -m src.gp_pgv_baseline_v2
    python -m src.gp_pgv_baseline_v2 --n-observed 128 --n-seeds 10 --max-events 20
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .balanced_pod_analysis import FREQUENCY_BANDS
from .config import DATA_PROCESSED, RESULTS_TABLES
from .gappy_pod_reconstruction import pgv_from_snapshots, relative_frobenius_error
from .plot_spatial_error_map import N_I, N_J, select_stations_random, station_coords_km
from .train_lowfreq_opinf import (
    DEFAULT_BAND,
    DEFAULT_DT_SECONDS,
    DEFAULT_EVENT_CLASS,
    discover_processed_events,
    load_bandpassed_snapshots,
    split_records,
)

_BATCH = 8000

# Hayward Fault strike ≈ 20° from the local x-axis (x = N-S, y = W direction)
_FAULT_THETA = math.radians(20.0)
_FAULT_R = np.array([
    [ math.cos(_FAULT_THETA), math.sin(_FAULT_THETA)],
    [-math.sin(_FAULT_THETA), math.cos(_FAULT_THETA)],
], dtype=np.float64)

# Loth & Baker (2013) T=1 s model parameters (proxy for PGV)
_LB_B1, _LB_b1_km = 0.367, 10.8    # short-range: amplitude, length scale (km)
_LB_B2, _LB_b2_km = 0.305, 111.0   # long-range: amplitude, length scale (km)


def _build_iso_kernel():
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=15.0, length_scale_bounds=(1.0, 150.0), nu=1.5)
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1.0))
    )


def _build_aniso_kernel():
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(
            length_scale=[30.0, 10.0],
            length_scale_bounds=[(5.0, 200.0), (1.0, 80.0)],
            nu=1.5,
        )
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1.0))
    )


def _build_lb_kernel():
    """Loth & Baker (2013) two-component correlation kernel (km coordinates)."""
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * (
            ConstantKernel(_LB_B1, "fixed")
            * Matern(length_scale=_LB_b1_km, length_scale_bounds=(2.0, 50.0), nu=0.5)
            + ConstantKernel(_LB_B2, "fixed")
            * Matern(length_scale=_LB_b2_km, length_scale_bounds=(30.0, 500.0), nu=0.5)
        )
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 2.0))
    )


_KERNEL_BUILDERS = {
    "iso":   _build_iso_kernel,
    "aniso": _build_aniso_kernel,
    "lb":    _build_lb_kernel,
}


def _gp_predict(
    kernel,
    coords_obs: np.ndarray,
    log_pgv_obs: np.ndarray,
    coords_hid: np.ndarray,
    n_restarts: int = 2,
) -> np.ndarray:
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        normalize_y=True,
        alpha=0.0,
    )
    gp.fit(coords_obs, log_pgv_obs)
    log_pred = np.empty(len(coords_hid))
    for start in range(0, len(coords_hid), _BATCH):
        log_pred[start:start + _BATCH] = gp.predict(
            coords_hid[start:start + _BATCH]
        )
    return np.exp(log_pred)


def run(
    data_root: Path,
    band_name: str,
    n_observed: int,
    seeds: list[int],
    train_fraction: float,
    max_events: int,
    n_restarts: int,
    output_csv: Path,
) -> None:
    band = FREQUENCY_BANDS[band_name]
    records = discover_processed_events(data_root, event_class=DEFAULT_EVENT_CLASS)
    if max_events:
        records = records[:max_events]

    _, val_records = split_records(records, train_fraction=train_fraction, seed=42)
    print(f"Val events: {len(val_records)}  n_observed: {n_observed}")

    n_stations = N_I * N_J
    all_coords = station_coords_km(n_stations)          # (n_stations, 2) km
    all_coords_fa = all_coords @ _FAULT_R.T             # fault-aligned rotation

    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        obs_idx = select_stations_random(n_stations, n_observed, rng)
        hidden_mask = np.ones(n_stations, dtype=bool)
        hidden_mask[obs_idx] = False
        hid_idx = np.where(hidden_mask)[0]

        coords_obs    = all_coords[obs_idx]
        coords_hid    = all_coords[hid_idx]
        coords_obs_fa = all_coords_fa[obs_idx]
        coords_hid_fa = all_coords_fa[hid_idx]

        variant_errors: dict[str, list[float]] = {v: [] for v in _KERNEL_BUILDERS}

        for rec in val_records:
            snaps = load_bandpassed_snapshots(rec, band=band)
            pgv_ref = pgv_from_snapshots(snaps, dt=DEFAULT_DT_SECONDS)
            pgv_obs = pgv_ref[obs_idx].clip(1e-12)
            pgv_hid_ref = pgv_ref[hid_idx]
            log_pgv_obs = np.log(pgv_obs)

            for variant, kernel_fn in _KERNEL_BUILDERS.items():
                kernel = kernel_fn()
                if variant == "aniso":
                    pgv_pred = _gp_predict(
                        kernel, coords_obs_fa, log_pgv_obs, coords_hid_fa, n_restarts
                    )
                else:
                    pgv_pred = _gp_predict(
                        kernel, coords_obs, log_pgv_obs, coords_hid, n_restarts
                    )
                err = relative_frobenius_error(pgv_hid_ref, pgv_pred)
                variant_errors[variant].append(err)

        for variant, errors in variant_errors.items():
            mean_err = float(np.mean(errors))
            rows.append({
                "seed":               seed,
                "n_observed":         n_observed,
                "variant":            variant,
                "mean_pgv_error":     mean_err,
                "n_val_events":       len(errors),
            })
            print(f"  seed={seed:2d}  {variant:5s}  pgv_err={mean_err:.4f}")

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("variant")["mean_pgv_error"]
        .agg(mean="mean", std="std")
        .round(4)
    )
    print("\nSummary (mean ± std over seeds):")
    print(summary.to_string())

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root", type=Path, default=DATA_PROCESSED / "events",
        help="Path to processed event data directory",
    )
    p.add_argument(
        "--band", choices=sorted(FREQUENCY_BANDS), default=DEFAULT_BAND,
    )
    p.add_argument("--n-observed",     type=int,   default=128)
    p.add_argument("--n-seeds",        type=int,   default=10)
    p.add_argument("--max-events",     type=int,   default=20)
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument("--n-restarts",     type=int,   default=2)
    p.add_argument(
        "--output-csv", type=Path,
        default=RESULTS_TABLES / "gp_pgv_baseline_v2_128.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(
        data_root=args.data_root,
        band_name=args.band,
        n_observed=args.n_observed,
        seeds=list(range(args.n_seeds)),
        train_fraction=args.train_fraction,
        max_events=args.max_events,
        n_restarts=args.n_restarts,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
