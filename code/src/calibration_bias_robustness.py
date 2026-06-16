"""Calibration-bias robustness experiment for gappy POD reconstruction.

Models a per-station, per-component DC offset b_{s,α} that is constant in time
(unlike zero-mean Gaussian noise, which is independent at each time step).
This represents:
  - Instrument-response miscalibration (gain offset)
  - Long-period drift or DC coupling
  - Site-amplification modelling error

The bias model: x_obs_corrupted[dof, t] = x_obs_true[dof, t] + b[dof]
where b[dof] ~ N(0, delta_b^2 * rms_obs^2) drawn once per station/component,
then held constant across all time steps.

Because the current gappy POD formulation solves independently at each time
step, the coefficient error from a constant bias b is:

  Δa_k = (V_obs^T V_obs + λI)^{-1} V_obs^T b   (same for all k)

This is structurally identical to the error from Gaussian noise of the same
amplitude at a single time step — the only difference is that for noise the
error varies randomly per step while for calibration bias it is a fixed
vector. Both give the same waveform-error magnitude.

The experiment therefore:
  1. Confirms that the noise-robustness table is equally valid for bias
  2. Shows that λ_b = δ_b² σ_ev² (discrepancy-principle estimate) stabilises
     both layouts, even though the bias is not zero-mean over time
  3. Tests δ_b ∈ {0.001, 0.01, 0.05} matching the noise experiment

Outputs
-------
  results/tables/calibration_bias_mw7.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .balanced_pod_analysis import FREQUENCY_BANDS
from .config import DATA_PROCESSED, RESULTS_TABLES
from .gappy_pod_reconstruction import (
    N_COMPONENTS,
    _station_dof_cf,
    pgv_from_snapshots,
    reconstruct_from_coefficients,
    relative_frobenius_error,
    solve_gappy_coefficients,
    station_dof_indices,
    n_stations_from_snapshots,
    load_mapped_station_indices,
    select_stations_ncsn,
    select_stations_random,
    DEFAULT_STATION_MAPPING,
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

OUT_CSV = RESULTS_TABLES / "calibration_bias_mw7.csv"

RANK          = 151
N_OBSERVED    = 128
TRAIN_FRAC    = 0.8
SEED          = 42
MAX_EVENTS    = 100
LAMBDA_FIXED  = 1e-8
DELTA_LEVELS  = [0.0, 0.001, 0.01, 0.05]
N_BIAS_SEEDS  = 10   # number of independent bias draws per event


def lambda_bias(x_obs_clean: np.ndarray, delta_b: float) -> float:
    """λ_b = δ_b² × per-DOF RMS² (discrepancy-principle estimate for bias)."""
    if delta_b == 0.0:
        return LAMBDA_FIXED
    return delta_b**2 * float(np.mean(x_obs_clean.astype(np.float64)**2))


def corrupt_with_bias(snapshots: np.ndarray, obs_dof: np.ndarray,
                      delta_b: float, rms_obs: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Add a fixed-in-time per-DOF bias drawn from N(0, (delta_b*rms_obs)^2)."""
    if delta_b == 0.0:
        return snapshots
    bias_std = delta_b * rms_obs
    biases = rng.normal(0.0, bias_std, size=obs_dof.shape[0]).astype(np.float32)
    corrupted = snapshots.copy()
    # Add bias to all time steps
    corrupted[obs_dof, :] += biases[:, np.newaxis]
    return corrupted


def evaluate_bias_event(
    basis: np.ndarray,
    snapshots: np.ndarray,
    obs_stations: np.ndarray,
    dt: float,
    delta_b: float,
    rng: np.random.Generator,
) -> dict:
    n_stations = n_stations_from_snapshots(snapshots)
    hid_stations = np.setdiff1d(np.arange(n_stations), obs_stations)
    obs_dof = station_dof_indices(obs_stations, n_stations=n_stations)
    hid_dof = station_dof_indices(hid_stations, n_stations=n_stations)

    x_obs_clean = snapshots[obs_dof].astype(np.float64)
    rms_obs = float(np.sqrt(np.mean(x_obs_clean**2)))
    lam_b = lambda_bias(x_obs_clean, delta_b)
    pgv_true = pgv_from_snapshots(snapshots, dt=dt)

    wf_fixed_list, pgv_fixed_list = [], []
    wf_b_list,     pgv_b_list     = [], []

    n_iter = 1 if delta_b == 0.0 else N_BIAS_SEEDS
    for _ in range(n_iter):
        corrupted = corrupt_with_bias(snapshots, obs_dof, delta_b, rms_obs, rng)

        for lam, wfl, pgvl in [(LAMBDA_FIXED, wf_fixed_list, pgv_fixed_list),
                                (lam_b,        wf_b_list,     pgv_b_list)]:
            coeffs, _ = solve_gappy_coefficients(basis, corrupted, obs_stations, lam)
            recon = reconstruct_from_coefficients(basis, coeffs)
            wfl.append(relative_frobenius_error(snapshots[hid_dof], recon[hid_dof]))
            pgv_r = pgv_from_snapshots(recon, dt=dt)
            pgvl.append(relative_frobenius_error(pgv_true[hid_stations],
                                                  pgv_r[hid_stations]))

    return {
        "wf_fixed":  float(np.mean(wf_fixed_list)),
        "pgv_fixed": float(np.mean(pgv_fixed_list)),
        "wf_b":      float(np.mean(wf_b_list)),
        "pgv_b":     float(np.mean(pgv_b_list)),
        "lambda_b":  lam_b,
        "rms_obs":   rms_obs,
    }


def run(
    data_root: Path = DATA_PROCESSED / "events",
    mapping_csv: Path = DEFAULT_STATION_MAPPING,
    output_csv: Path = OUT_CSV,
) -> None:
    band = FREQUENCY_BANDS[DEFAULT_BAND]
    records = discover_processed_events(data_root, event_class=DEFAULT_EVENT_CLASS)[:MAX_EVENTS]
    train_records, val_records = split_records(records, train_fraction=TRAIN_FRAC, seed=SEED)
    print(f"Train: {len(train_records)}  Val: {len(val_records)}")

    train_matrix = assemble_snapshot_matrix(train_records, band=band)
    basis_full, _ = compute_pod_basis(train_matrix, rank=RANK)
    basis = basis_full[:, :RANK]
    n_stations = train_matrix.shape[0] // N_COMPONENTS

    mapped = load_mapped_station_indices(mapping_csv)
    mapped = mapped[(mapped >= 0) & (mapped < n_stations)]

    rng_station = np.random.default_rng(SEED + N_OBSERVED)
    random_stations = select_stations_random(n_stations, N_OBSERVED, rng_station)
    ncsn_stations   = select_stations_ncsn(mapped, N_OBSERVED, n_stations)

    layouts = [("random", random_stations), ("ncsn", ncsn_stations)]
    val_snaps = [(rec, load_bandpassed_snapshots(rec, band=band)) for rec in val_records]
    dt = val_records[0].dt

    rows = []
    for delta_b in DELTA_LEVELS:
        for layout_name, obs in layouts:
            rng = np.random.default_rng(3141 + int(delta_b * 1000))
            ev_results = [
                evaluate_bias_event(basis, snaps, obs, dt, delta_b, rng)
                for _, snaps in val_snaps
            ]
            row = {
                "class":          DEFAULT_EVENT_CLASS,
                "frequency_band": DEFAULT_BAND,
                "layout":         layout_name,
                "rank":           RANK,
                "n_observed":     N_OBSERVED,
                "n_val_events":   len(val_records),
                "delta_b":        delta_b,
                "mean_lambda_b":  float(np.mean([r["lambda_b"] for r in ev_results])),
                "mean_wf_fixed":  float(np.mean([r["wf_fixed"] for r in ev_results])),
                "mean_pgv_fixed": float(np.mean([r["pgv_fixed"] for r in ev_results])),
                "mean_wf_b":      float(np.mean([r["wf_b"]     for r in ev_results])),
                "mean_pgv_b":     float(np.mean([r["pgv_b"]    for r in ev_results])),
                "mean_rms_obs":   float(np.mean([r["rms_obs"]  for r in ev_results])),
            }
            rows.append(row)
            print(f"  delta_b={delta_b}  {layout_name:<8}  "
                  f"wf_fixed={row['mean_wf_fixed']:.4f}  wf_b={row['mean_wf_b']:.4f}  "
                  f"pgv_fixed={row['mean_pgv_fixed']:.4f}  pgv_b={row['mean_pgv_b']:.4f}  "
                  f"lam_b={row['mean_lambda_b']:.2e}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved → {output_csv}")


if __name__ == "__main__":
    run()
