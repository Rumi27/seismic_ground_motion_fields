"""Timing-jitter sensitivity of gappy POD reconstruction.

Models per-station timing errors by applying an independent integer-sample
time shift drawn from Uniform(-J_max, J_max) to each observed station's
full waveform before passing it to gappy POD.  Shifts use boundary-fill
padding (edge-hold) to avoid wrap-around artefacts.

The effective observation error introduced by a J-sample shift is
approximately delta_x_i(t) ≈ J_i * dt * (dx/dt)_i(t), so timing jitter
acts like coloured noise with variance:

  sigma²_jitter ≈ J_max² / 3 * mean( (x[t+1] - x[t])² )

This variance defines a jitter-aware Tikhonov parameter:

  lambda_J = J_max² / 3 * mean( (x_obs[t+1] - x_obs[t])² )

Both fixed lambda = 1e-8 and lambda_J are tested.
Jitter levels tested: J_max in {0, 1, 2, 5} samples.
Layouts: random (well-conditioned) and NCSN-derived (ill-conditioned).

Usage
-----
    python -m src.timing_jitter_gappy_pod
    python -m src.timing_jitter_gappy_pod --max-events 40
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
    n_stations_from_snapshots,
    pgv_from_snapshots,
    reconstruct_from_coefficients,
    relative_frobenius_error,
    station_dof_indices,
)
from .plot_spatial_error_map import (
    DEFAULT_STATION_MAPPING,
    load_mapped_station_indices,
    select_stations_ncsn,
    select_stations_random,
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

JITTER_LEVELS = [0, 1, 2, 5]          # integer samples
DEFAULT_RANK = 151
DEFAULT_N_OBSERVED = 128
DEFAULT_LAMBDA_FIXED = 1e-8
DEFAULT_N_JITTER_SEEDS = 10


def _solve_gappy(
    basis: np.ndarray,
    x_obs_noisy: np.ndarray,
    observed_stations: np.ndarray,
    regularization: float,
) -> np.ndarray:
    obs_idx = station_dof_indices(observed_stations)
    v_obs = basis[obs_idx].astype(np.float64)
    x = x_obs_noisy[obs_idx].astype(np.float64)
    gram = v_obs.T @ v_obs + regularization * np.eye(v_obs.shape[1], dtype=np.float64)
    return np.linalg.solve(gram, v_obs.T @ x).astype(np.float32)


def _apply_jitter(
    snapshots: np.ndarray,
    station_ids: np.ndarray,
    n_stations: int,
    j_max: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return copy of snapshots with per-station time shifts on observed stations.

    Positive shift: waveform advanced (future values moved earlier).
    Edge-hold boundary: no wrap-around, edge value repeated for filled samples.
    """
    if j_max == 0:
        return snapshots.copy()
    out = snapshots.copy()
    shifts = rng.integers(-j_max, j_max + 1, size=len(station_ids))
    for comp in range(N_COMPONENTS):
        for s_id, shift in zip(station_ids, shifts):
            if shift == 0:
                continue
            row = comp * n_stations + s_id
            trace = out[row].copy()
            if shift > 0:
                out[row, :-shift] = trace[shift:]
                out[row, -shift:] = trace[-1]
            else:
                lag = -shift
                out[row, lag:] = trace[:-lag]
                out[row, :lag] = trace[0]
    return out


def _lambda_jitter(x_obs: np.ndarray, j_max: int) -> float:
    """Jitter-aware Tikhonov parameter based on effective noise variance.

    For small jitter, the error per observation is approximately
    J * (dx/dt) * dt = J * diff(x).  Variance is J^2/3 * mean(diff^2).
    """
    if j_max == 0:
        return DEFAULT_LAMBDA_FIXED
    diffs = np.diff(x_obs.astype(np.float64), axis=-1)
    return float(j_max**2 / 3.0 * np.mean(diffs**2))


def evaluate_jitter_event(
    basis: np.ndarray,
    snapshots: np.ndarray,
    observed_stations: np.ndarray,
    dt: float,
    j_max: int,
    lambda_fixed: float,
    n_seeds: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    n_stations = n_stations_from_snapshots(snapshots)
    hidden_stations = np.setdiff1d(np.arange(n_stations), observed_stations)
    obs_idx = station_dof_indices(observed_stations)
    hid_idx = station_dof_indices(hidden_stations)
    pgv_true = pgv_from_snapshots(snapshots, dt=dt)

    lam_J = _lambda_jitter(snapshots[obs_idx], j_max)

    wf_fixed, pgv_fixed, wf_J, pgv_J = [], [], [], []
    n_iters = 1 if j_max == 0 else n_seeds
    for _ in range(n_iters):
        jittered = _apply_jitter(snapshots, observed_stations, n_stations, j_max, rng)
        for lam, wf_list, pgv_list in [
            (lambda_fixed, wf_fixed, pgv_fixed),
            (lam_J,        wf_J,     pgv_J),
        ]:
            coeffs = _solve_gappy(basis, jittered, observed_stations, lam)
            recon  = reconstruct_from_coefficients(basis, coeffs)
            wf_list.append(relative_frobenius_error(snapshots[hid_idx], recon[hid_idx]))
            pgv_list.append(relative_frobenius_error(
                pgv_true[hidden_stations],
                pgv_from_snapshots(recon, dt=dt)[hidden_stations],
            ))

    return {
        "wf_fixed":  float(np.mean(wf_fixed)),
        "pgv_fixed": float(np.mean(pgv_fixed)),
        "wf_J":      float(np.mean(wf_J)),
        "pgv_J":     float(np.mean(pgv_J)),
        "lambda_J":  lam_J,
    }


def run(
    data_root: Path,
    mapping_csv: Path,
    rank: int,
    n_observed: int,
    train_fraction: float,
    seed: int,
    max_events: int | None,
    lambda_fixed: float,
    n_jitter_seeds: int,
    output_csv: Path,
    jitter_levels: list[int] = JITTER_LEVELS,
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

    mapped_indices = load_mapped_station_indices(mapping_csv)
    rng_station = np.random.default_rng(seed + n_observed)
    random_stations = select_stations_random(n_stations, n_observed, rng_station)
    ncsn_stations   = select_stations_ncsn(mapped_indices, n_observed, n_stations)

    layouts = [("random", random_stations), ("ncsn", ncsn_stations)]
    val_snapshots = [
        (rec, load_bandpassed_snapshots(rec, band=band)) for rec in val_records
    ]
    dt = val_records[0].dt

    rows: list[dict] = []
    for j_max in jitter_levels:
        for layout_name, obs_stations in layouts:
            rng = np.random.default_rng(3141 + j_max * 100)
            ev_results = [
                evaluate_jitter_event(
                    basis=basis, snapshots=snaps, observed_stations=obs_stations,
                    dt=dt, j_max=j_max, lambda_fixed=lambda_fixed,
                    n_seeds=n_jitter_seeds, rng=rng,
                )
                for _, snaps in val_snapshots
            ]
            row = {
                "class":               DEFAULT_EVENT_CLASS,
                "frequency_band":      DEFAULT_BAND,
                "layout":              layout_name,
                "rank":                rank,
                "n_observed_stations": n_observed,
                "n_val_events":        len(val_records),
                "j_max_samples":       j_max,
                "j_max_seconds":       round(j_max * dt, 4),
                "mean_lambda_J":       float(np.mean([r["lambda_J"]  for r in ev_results])),
                "mean_wf_fixed":       float(np.mean([r["wf_fixed"]  for r in ev_results])),
                "mean_pgv_fixed":      float(np.mean([r["pgv_fixed"] for r in ev_results])),
                "mean_wf_J":           float(np.mean([r["wf_J"]      for r in ev_results])),
                "mean_pgv_J":          float(np.mean([r["pgv_J"]     for r in ev_results])),
            }
            rows.append(row)
            print(
                f"J_max={j_max}samp ({j_max*dt:.3f}s)  {layout_name:<8s}  "
                f"wf_fixed={row['mean_wf_fixed']:.4f}  pgv_fixed={row['mean_pgv_fixed']:.4f}  "
                f"wf_J={row['mean_wf_J']:.4f}  pgv_J={row['mean_pgv_J']:.4f}  "
                f"lam_J={row['mean_lambda_J']:.2e}",
                flush=True,
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved → {output_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",           type=Path, default=DATA_PROCESSED / "events")
    p.add_argument("--mapping-csv",         type=Path, default=DEFAULT_STATION_MAPPING)
    p.add_argument("--rank",                type=int,  default=DEFAULT_RANK)
    p.add_argument("--n-observed",          type=int,  default=DEFAULT_N_OBSERVED)
    p.add_argument("--train-fraction",      type=float, default=0.8)
    p.add_argument("--seed",                type=int,  default=42)
    p.add_argument("--max-events",          type=int,  default=100)
    p.add_argument("--lambda-fixed",        type=float, default=DEFAULT_LAMBDA_FIXED)
    p.add_argument("--n-jitter-seeds",      type=int,  default=DEFAULT_N_JITTER_SEEDS)
    p.add_argument("--output-csv",          type=Path,
                   default=RESULTS_TABLES / "timing_jitter_gappy_pod_mw7.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(
        data_root=args.data_root,
        mapping_csv=args.mapping_csv,
        rank=args.rank,
        n_observed=args.n_observed,
        train_fraction=args.train_fraction,
        seed=args.seed,
        max_events=args.max_events,
        lambda_fixed=args.lambda_fixed,
        n_jitter_seeds=args.n_jitter_seeds,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
