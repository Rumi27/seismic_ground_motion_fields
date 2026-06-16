"""Station-count sweep comparing 3-component vs 2-component (horizontal-only) observations.

For each observed station set the script solves the gappy POD coefficient problem
with either all three displacement components (x, y, z) or only the two horizontal
components (x, y), matching conditions common in strong-motion networks.

Quantities reported per experiment:
  mean_rel_error_heldout  — waveform error at held-out stations (all 3 components)
  mean_pgv_error_heldout  — PGV error at held-out stations
  mean_cond_vobs          — condition number of the observation submatrix V_obs
                            (Σ_max / Σ_min from SVD; large ≡ ill-conditioned)

Layouts evaluated: random (5 seeds) and farthest-point.
NCSN and POD-QR are omitted because their qualitative behaviour (clustering /
non-monotone pivot) is independent of the component-count question.

Results: results/tables/station_count_sweep_components.csv

Usage
-----
    python -m src.station_count_sweep_components
    python -m src.station_count_sweep_components --max-events 40
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
    select_stations_farthest,
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
N_HORIZ = 2   # horizontal component count


def _horiz_dof_idx(station_ids: np.ndarray) -> np.ndarray:
    """DOF row indices for horizontal (x, y) components only: [3s, 3s+1] per station."""
    bases = np.asarray(station_ids, dtype=np.int64) * N_COMPONENTS
    return np.stack([bases, bases + 1], axis=1).ravel()


def _cond_vobs(basis: np.ndarray, dof_idx: np.ndarray) -> float:
    v = basis[dof_idx].astype(np.float64)
    sv = np.linalg.svd(v, compute_uv=False)
    if sv[-1] < 1e-14:
        return float("inf")
    return float(sv[0] / sv[-1])


def _solve_gappy(
    basis: np.ndarray,
    snapshots: np.ndarray,
    obs_stations: np.ndarray,
    regularization: float,
    n_obs_comp: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generalised gappy solve supporting n_obs_comp ∈ {2, 3}.

    Returns (coefficients, hidden_stations, v_obs, cond_vobs).
    """
    n_st = n_stations_from_snapshots(snapshots)
    if n_obs_comp == N_COMPONENTS:
        obs_dof = station_dof_indices(obs_stations)
    else:
        obs_dof = _horiz_dof_idx(obs_stations)

    hid_st = np.setdiff1d(np.arange(n_st), obs_stations, assume_unique=True)

    v_obs = basis[obs_dof].astype(np.float64)
    x_obs = snapshots[obs_dof].astype(np.float64)
    gram = v_obs.T @ v_obs
    reg = regularization * np.eye(gram.shape[0], dtype=np.float64)
    coeffs = np.linalg.solve(gram + reg, v_obs.T @ x_obs)

    sv = np.linalg.svd(v_obs, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 1e-14 else float("inf")

    return coeffs.astype(np.float32), hid_st, cond


def _evaluate(
    basis: np.ndarray,
    snapshots: np.ndarray,
    obs_stations: np.ndarray,
    dt: float,
    regularization: float,
    n_obs_comp: int,
) -> dict:
    coeffs, hid_st, cond = _solve_gappy(
        basis, snapshots, obs_stations, regularization, n_obs_comp
    )
    recon = reconstruct_from_coefficients(basis, coeffs)

    hid_dof = station_dof_indices(hid_st)
    wf_err = relative_frobenius_error(snapshots[hid_dof], recon[hid_dof])

    pgv_true = pgv_from_snapshots(snapshots, dt=dt)
    pgv_pred = pgv_from_snapshots(recon, dt=dt)
    pgv_err = relative_frobenius_error(pgv_true[hid_st], pgv_pred[hid_st])

    return {"wf_err": wf_err, "pgv_err": pgv_err, "cond_vobs": cond}


def _aggregate(
    layout: str,
    seed_used: int,
    n_obs: int,
    n_obs_comp: int,
    n_stations: int,
    val_records,
    metrics: list[dict],
) -> dict:
    return {
        "n_obs_comp":             n_obs_comp,
        "layout":                 layout,
        "seed":                   seed_used,
        "rank":                   DEFAULT_RANK,
        "n_observed_stations":    n_obs,
        "n_heldout_stations":     n_stations - n_obs,
        "n_val_events":           len(val_records),
        "mean_rel_error_heldout": float(np.mean([m["wf_err"]   for m in metrics])),
        "mean_pgv_error_heldout": float(np.mean([m["pgv_err"]  for m in metrics])),
        "mean_cond_vobs":         float(np.mean([m["cond_vobs"] for m in metrics])),
    }


def run(
    data_root: Path,
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
    dt = train_records[0].dt

    val_data = [
        (rec, load_bandpassed_snapshots(rec, band=band)) for rec in val_records
    ]

    rows: list[dict] = []

    for n_obs in n_obs_list:
        if n_obs >= n_stations:
            print(f"skip n_obs={n_obs}")
            continue

        for n_comp in (3, 2):
            tag = f"n_comp={n_comp}"

            # Random: 5 seeds (same seed → same station positions for both comp modes)
            for s in range(n_random_seeds):
                rng = np.random.default_rng(seed * 1000 + s * 17 + n_obs)
                obs = select_stations_random(n_stations, n_obs, rng)
                metrics = [
                    _evaluate(basis, snaps, obs, dt=rec.dt,
                              regularization=regularization, n_obs_comp=n_comp)
                    for rec, snaps in val_data
                ]
                rows.append(_aggregate("random", s, n_obs, n_comp, n_stations, val_records, metrics))
                print(
                    f"n_obs={n_obs:4d}  {tag}  random s={s}  "
                    f"wf={rows[-1]['mean_rel_error_heldout']:.4f}  "
                    f"pgv={rows[-1]['mean_pgv_error_heldout']:.4f}  "
                    f"cond={rows[-1]['mean_cond_vobs']:.1e}",
                    flush=True,
                )

            # Farthest-point
            rng_fp = np.random.default_rng(seed + 9999)
            obs_fp = select_stations_farthest(coords, n_obs, rng_fp)
            metrics_fp = [
                _evaluate(basis, snaps, obs_fp, dt=rec.dt,
                          regularization=regularization, n_obs_comp=n_comp)
                for rec, snaps in val_data
            ]
            rows.append(_aggregate("farthest", 0, n_obs, n_comp, n_stations, val_records, metrics_fp))
            print(
                f"n_obs={n_obs:4d}  {tag}  farthest       "
                f"wf={rows[-1]['mean_rel_error_heldout']:.4f}  "
                f"pgv={rows[-1]['mean_pgv_error_heldout']:.4f}  "
                f"cond={rows[-1]['mean_cond_vobs']:.1e}",
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
    p.add_argument("--data-root",      type=Path, default=DATA_PROCESSED / "events")
    p.add_argument("--rank",           type=int,   default=DEFAULT_RANK)
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--max-events",     type=int,   default=100)
    p.add_argument("--regularization", type=float, default=DEFAULT_REG)
    p.add_argument(
        "--output-csv", type=Path,
        default=RESULTS_TABLES / "station_count_sweep_components.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(
        data_root=args.data_root,
        rank=args.rank,
        train_fraction=args.train_fraction,
        seed=args.seed,
        max_events=args.max_events,
        regularization=args.regularization,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
