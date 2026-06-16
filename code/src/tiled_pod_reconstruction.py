"""2×2 tiled Gappy POD reconstruction for mid-band (0.25–0.5 Hz).

Domain: 367 rows × 171 cols stations (73.4 × 38.5 km).
Split: 2×2 tiles with raised-cosine blending in overlap zones.
Local POD: randomised SVD, rank ``--rank`` per tile (default 20).
Reconstruction: Tikhonov-regularised Gappy POD per tile, blended at boundaries.

Tile numbering (row-major): 0=(top-left), 1=(top-right), 2=(bottom-left), 3=(bottom-right).

Usage
-----
    python -m src.tiled_pod_reconstruction
    python -m src.tiled_pod_reconstruction --n-seeds 10 --rank 20 --n-observed 128
    python -m src.tiled_pod_reconstruction --band mid_025_05hz --rank 20 --overlap-fraction 0.25
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.utils.extmath import randomized_svd

from .balanced_pod_analysis import DEFAULT_DT_SECONDS, FREQUENCY_BANDS
from .config import DATA_PROCESSED, DATA_RAW, RESULTS_TABLES
from .gappy_pod_reconstruction import (
    N_COMPONENTS,
    baseline_idw_hidden,
    find_reference_h5,
    load_station_coordinates,
    pgv_from_snapshots,
    pgv_from_station_tensor,
    relative_frobenius_error,
    select_stations_random,
    station_dof_indices,
)
from .train_lowfreq_opinf import (
    DEFAULT_EVENT_CLASS,
    assemble_snapshot_matrix,
    discover_processed_events,
    load_bandpassed_snapshots,
    split_records,
)

DEFAULT_MID_BAND = "mid_025_05hz"
DEFAULT_TILE_RANK = 20
DEFAULT_OVERLAP_FRACTION = 0.25
DEFAULT_N_OBSERVED = 128
DEFAULT_N_SEEDS = 10


# ---------------------------------------------------------------------------
# Partition-of-unity blend weights (raised-cosine, 2×2 tiles)
# ---------------------------------------------------------------------------


def compute_blend_weights(
    coords: np.ndarray,
    overlap_fraction: float = DEFAULT_OVERLAP_FRACTION,
) -> np.ndarray:
    """Raised-cosine partition-of-unity blend weights for a 2×2 domain split.

    Parameters
    ----------
    coords:
        (n_stations, 2) of (row, col) grid indices.
    overlap_fraction:
        Fraction of each half-extent used for the cosine-blended transition zone.
        0.25 → overlap spans ±25% of half-width on each side of the split line.

    Returns
    -------
    weights : (n_stations, 4) float64
        Column tile_id = ti*2 + tj for ti, tj in {0, 1}.
        Rows sum to 1.0 for every station (partition of unity).
    """
    rows = coords[:, 0].astype(np.float64)
    cols = coords[:, 1].astype(np.float64)

    r_mid = 0.5 * (rows.min() + rows.max())
    c_mid = 0.5 * (cols.min() + cols.max())
    ov_r = overlap_fraction * 0.5 * (rows.max() - rows.min())
    ov_c = overlap_fraction * 0.5 * (cols.max() - cols.min())

    def _hann_weights(x: np.ndarray, x_mid: float, ov: float) -> np.ndarray:
        """Return (n, 2) weights: w[:, 0] for lower half, w[:, 1] for upper half."""
        w = np.empty((len(x), 2), dtype=np.float64)
        lo = x < x_mid - ov
        hi = x > x_mid + ov
        mid = ~(lo | hi)
        w[lo, 0] = 1.0;  w[lo, 1] = 0.0
        w[hi, 0] = 0.0;  w[hi, 1] = 1.0
        if mid.any():
            t = (x[mid] - (x_mid - ov)) / (2.0 * ov)  # 0 → 1 across overlap zone
            w[mid, 0] = 0.5 * (1.0 + np.cos(np.pi * t))
            w[mid, 1] = 1.0 - w[mid, 0]
        return w

    w_row = _hann_weights(rows, r_mid, ov_r)  # (n, 2): [lower-half, upper-half]
    w_col = _hann_weights(cols, c_mid, ov_c)  # (n, 2): [left-half,  right-half]

    return np.column_stack([
        w_row[:, 0] * w_col[:, 0],  # tile 0: top-left
        w_row[:, 0] * w_col[:, 1],  # tile 1: top-right
        w_row[:, 1] * w_col[:, 0],  # tile 2: bottom-left
        w_row[:, 1] * w_col[:, 1],  # tile 3: bottom-right
    ])


# ---------------------------------------------------------------------------
# Local POD basis via randomised SVD
# ---------------------------------------------------------------------------


def build_tile_pod_basis(
    train_matrix: np.ndarray,   # (N_C * n_stations, n_samples) float32
    tile_stations: np.ndarray,  # sorted global station indices
    n_stations: int,
    rank: int,
    random_state: int = 42,
) -> np.ndarray:                # (n_tile_dofs, rank) float32
    """Top-rank POD modes for a spatial tile via randomised SVD."""
    tile_dofs = station_dof_indices(tile_stations, n_stations)
    X_t = np.ascontiguousarray(train_matrix[tile_dofs])  # (n_tile_dofs, n_samples) float32
    k = min(rank, X_t.shape[0] - 1, X_t.shape[1] - 1)
    U, _, _ = randomized_svd(X_t, n_components=k, n_iter=4, random_state=random_state)
    return U.astype(np.float32)


# ---------------------------------------------------------------------------
# Tiled Gappy POD reconstruction
# ---------------------------------------------------------------------------


def reconstruct_tiled(
    tile_bases: list[tuple[np.ndarray, np.ndarray]],
    blend_weights: np.ndarray,
    snapshots: np.ndarray,
    obs_stations: np.ndarray,
    n_stations: int,
    regularization: float,
    train_mean: np.ndarray,
) -> np.ndarray:
    """Blend 4 local Gappy POD reconstructions into a global field.

    Parameters
    ----------
    tile_bases:
        List of 4 (tile_stations, V_tile) pairs. ``tile_stations`` is a sorted
        array of global station indices; ``V_tile`` has shape (n_tile_dofs, rank).
    blend_weights:
        (n_stations, 4) partition-of-unity weights from ``compute_blend_weights``.
    snapshots:
        (N_C * n_stations, n_time) ground-truth field (observed values extracted inside).
    obs_stations:
        Sorted array of global station indices that are observed.
    n_stations:
        Total number of stations in the domain.
    regularization:
        Tikhonov regularisation coefficient λ for the per-tile Gappy solve.
    train_mean:
        (N_C * n_stations,) training-set mean, used as fallback when a tile has
        fewer than 2 observed stations.

    Returns
    -------
    recon : (N_C * n_stations, n_time) float32
    """
    n_time = snapshots.shape[1]
    recon = np.zeros((N_COMPONENTS * n_stations, n_time), dtype=np.float32)

    for tile_id, (tile_stations, V_tile) in enumerate(tile_bases):
        n_tile = len(tile_stations)
        tile_dofs_global = station_dof_indices(tile_stations, n_stations)
        # Per-station blend weight replicated for each component (DOF ordering)
        w_tile = np.tile(blend_weights[tile_stations, tile_id], N_COMPONENTS)  # (n_tile_dofs,)

        obs_in_tile = np.intersect1d(obs_stations, tile_stations, assume_unique=True)

        if obs_in_tile.size < 2:
            # Degenerate tile: blend in training mean
            recon[tile_dofs_global] += w_tile[:, np.newaxis] * train_mean[tile_dofs_global, np.newaxis]
            continue

        # Map observed global station indices → local row positions in tile_stations
        obs_local = np.searchsorted(tile_stations, obs_in_tile)   # sorted, exact match guaranteed
        obs_local_dofs = station_dof_indices(obs_local, n_tile)   # local DOF indices into V_tile
        obs_global_dofs = station_dof_indices(obs_in_tile, n_stations)

        # Tikhonov-regularised Gappy POD solve
        V_obs = V_tile[obs_local_dofs].astype(np.float64)          # (n_obs_dofs, r)
        x_obs = snapshots[obs_global_dofs].astype(np.float64)      # (n_obs_dofs, n_time)
        gram = V_obs.T @ V_obs                                      # (r, r)
        reg = regularization * np.eye(gram.shape[0], dtype=np.float64)
        alpha = np.linalg.solve(gram + reg, V_obs.T @ x_obs)       # (r, n_time)

        # Full tile reconstruction and weighted accumulation
        q_hat = (V_tile.astype(np.float64) @ alpha).astype(np.float32)  # (n_tile_dofs, n_time)
        recon[tile_dofs_global] += w_tile[:, np.newaxis] * q_hat

    return recon


# ---------------------------------------------------------------------------
# Observability diagnostics per tile
# ---------------------------------------------------------------------------


def tile_observability(
    V_tile: np.ndarray,          # (n_tile_dofs, rank)
    obs_local_dofs: np.ndarray,  # local DOF indices of observed stations
    regularization: float,
) -> dict[str, float]:
    """Condition number and sigma_min of the tile observation matrix."""
    V_obs = V_tile[obs_local_dofs].astype(np.float64)
    svs = np.linalg.svd(V_obs, compute_uv=False)
    sigma_min = float(svs[-1]) if svs.size >= V_tile.shape[1] else 0.0
    sigma_max = float(svs[0]) if svs.size > 0 else float("nan")
    cond = sigma_max / sigma_min if sigma_min > 0 else float("inf")
    gram = V_obs.T @ V_obs
    reg_gram = gram + regularization * np.eye(gram.shape[0])
    _, logdet = np.linalg.slogdet(reg_gram)
    return {"cond_vobs": cond, "sigma_min": sigma_min, "logdet_gram": float(logdet)}


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run(
    data_root: Path,
    raw_root: Path,
    band_name: str,
    n_observed: int,
    seeds: list[int],
    train_fraction: float,
    max_events: int,
    rank: int,
    overlap_fraction: float,
    regularization: float,
    output_csv: Path,
) -> None:
    band = FREQUENCY_BANDS[band_name]
    records = discover_processed_events(data_root, event_class=DEFAULT_EVENT_CLASS)
    if max_events:
        records = records[:max_events]

    train_records, val_records = split_records(records, train_fraction=train_fraction, seed=42)
    print(f"Band: {band_name}  Train: {len(train_records)}  Val: {len(val_records)}")

    ref_h5 = find_reference_h5(DEFAULT_EVENT_CLASS, raw_root=raw_root)
    coords = load_station_coordinates(ref_h5)   # (n_stations, 2) row/col
    n_stations = coords.shape[0]

    # ---- One-time setup ------------------------------------------------
    print("Assembling training snapshot matrix...")
    train_matrix = assemble_snapshot_matrix(train_records, band=band)   # (N_C*n, n_train)
    train_mean = train_matrix.mean(axis=1).astype(np.float32)

    print(f"Computing 2×2 blend weights (overlap_fraction={overlap_fraction})...")
    blend_weights = compute_blend_weights(coords, overlap_fraction=overlap_fraction)
    tile_sizes = (blend_weights > 0).sum(axis=0).tolist()
    r_mid = 0.5 * (coords[:, 0].min() + coords[:, 0].max())
    c_mid = 0.5 * (coords[:, 1].min() + coords[:, 1].max())
    print(f"  Split: row≈{r_mid:.0f}, col≈{c_mid:.0f}  |  tile station counts: {tile_sizes}")

    print(f"Building 4 local POD bases (rank={rank} per tile, n_iter=4 randomised SVD)...")
    tile_bases: list[tuple[np.ndarray, np.ndarray]] = []
    for tile_id in range(4):
        tile_stations = np.where(blend_weights[:, tile_id] > 0)[0]
        V_tile = build_tile_pod_basis(train_matrix, tile_stations, n_stations, rank)
        tile_bases.append((tile_stations, V_tile))
        print(f"  tile {tile_id}: {len(tile_stations)} stations → basis {V_tile.shape}")

    print("Loading validation snapshots...")
    val_snapshots = [
        (rec, load_bandpassed_snapshots(rec, band=band)) for rec in val_records
    ]

    # ---- Multi-seed experiment loop ------------------------------------
    rows: list[dict] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        obs_stations = select_stations_random(n_stations, n_observed, rng)
        hid_stations = np.setdiff1d(np.arange(n_stations), obs_stations)
        hid_dofs = station_dof_indices(hid_stations, n_stations)
        n_time_ref = val_snapshots[0][1].shape[1]

        # Observed stations per tile (those with w > 0 in tile)
        obs_per_tile = [(blend_weights[obs_stations, t] > 0).sum() for t in range(4)]

        # Per-tile observability at this seed (use first tile basis as representative)
        obs_metrics_per_tile: list[dict] = []
        for tile_id, (tile_stations, V_tile) in enumerate(tile_bases):
            obs_in_t = np.intersect1d(obs_stations, tile_stations, assume_unique=True)
            if obs_in_t.size >= 2:
                obs_local = np.searchsorted(tile_stations, obs_in_t)
                obs_local_dofs = station_dof_indices(obs_local, len(tile_stations))
                obs_metrics_per_tile.append(
                    tile_observability(V_tile, obs_local_dofs, regularization)
                )
            else:
                obs_metrics_per_tile.append(
                    {"cond_vobs": float("inf"), "sigma_min": 0.0, "logdet_gram": float("nan")}
                )

        wf_errors, pgv_errors = [], []
        idw_wf_errors, idw_pgv_errors = [], []

        for rec, snaps in val_snapshots:
            dt = rec.dt
            recon = reconstruct_tiled(
                tile_bases, blend_weights, snaps, obs_stations,
                n_stations, regularization, train_mean,
            )

            wf_errors.append(relative_frobenius_error(snaps[hid_dofs], recon[hid_dofs]))

            pgv_ref = pgv_from_snapshots(snaps, dt=dt)
            pgv_hat = pgv_from_snapshots(recon, dt=dt)
            pgv_errors.append(relative_frobenius_error(pgv_ref[hid_stations], pgv_hat[hid_stations]))

            # IDW baseline (hidden-stations only)
            idw_hid = baseline_idw_hidden(snaps, obs_stations, hid_stations, coords)
            idw_wf_errors.append(relative_frobenius_error(snaps[hid_dofs], idw_hid))

            idw_arr = idw_hid.reshape(N_COMPONENTS, len(hid_stations), snaps.shape[1])
            idw_pgv_hid = pgv_from_station_tensor(idw_arr, dt=dt)
            idw_pgv_errors.append(
                relative_frobenius_error(pgv_ref[hid_stations], idw_pgv_hid)
            )

        # Aggregate observability (mean sigma_min and cond across 4 tiles)
        mean_sigma_min = float(np.mean([m["sigma_min"] for m in obs_metrics_per_tile]))
        mean_cond = float(np.mean([m["cond_vobs"] for m in obs_metrics_per_tile
                                    if np.isfinite(m["cond_vobs"])]))

        row: dict = {
            "seed": seed,
            "band": band_name,
            "n_observed": n_observed,
            "rank_per_tile": rank,
            "overlap_fraction": overlap_fraction,
            "regularization": regularization,
            "n_val_events": len(val_snapshots),
            "obs_tile0": obs_per_tile[0],
            "obs_tile1": obs_per_tile[1],
            "obs_tile2": obs_per_tile[2],
            "obs_tile3": obs_per_tile[3],
            "sigma_min_mean": mean_sigma_min,
            "cond_vobs_mean": mean_cond,
            # Per-tile observability
            "sigma_min_tile0": obs_metrics_per_tile[0]["sigma_min"],
            "sigma_min_tile1": obs_metrics_per_tile[1]["sigma_min"],
            "sigma_min_tile2": obs_metrics_per_tile[2]["sigma_min"],
            "sigma_min_tile3": obs_metrics_per_tile[3]["sigma_min"],
            "cond_tile0": obs_metrics_per_tile[0]["cond_vobs"],
            "cond_tile1": obs_metrics_per_tile[1]["cond_vobs"],
            "cond_tile2": obs_metrics_per_tile[2]["cond_vobs"],
            "cond_tile3": obs_metrics_per_tile[3]["cond_vobs"],
            "overdetermination_tile0": obs_per_tile[0] * N_COMPONENTS / rank,
            "overdetermination_tile1": obs_per_tile[1] * N_COMPONENTS / rank,
            "overdetermination_tile2": obs_per_tile[2] * N_COMPONENTS / rank,
            "overdetermination_tile3": obs_per_tile[3] * N_COMPONENTS / rank,
            "tiled_wf_error": float(np.mean(wf_errors)),
            "tiled_pgv_error": float(np.mean(pgv_errors)),
            "idw_wf_error": float(np.mean(idw_wf_errors)),
            "idw_pgv_error": float(np.mean(idw_pgv_errors)),
        }
        rows.append(row)
        conds = [f"{obs_metrics_per_tile[t]['cond_vobs']:.1f}" for t in range(4)]
        print(
            f"  seed={seed:2d}: tiled_wf={row['tiled_wf_error']:.4f} "
            f"tiled_pgv={row['tiled_pgv_error']:.4f}  "
            f"idw_wf={row['idw_wf_error']:.4f} "
            f"obs/tile={obs_per_tile}  cond/tile={conds}"
        )

    # ---- Summary and output -------------------------------------------
    wf_mean  = float(np.mean([r["tiled_wf_error"]  for r in rows]))
    wf_std   = float(np.std( [r["tiled_wf_error"]  for r in rows]))
    pgv_mean = float(np.mean([r["tiled_pgv_error"] for r in rows]))
    pgv_std  = float(np.std( [r["tiled_pgv_error"] for r in rows]))
    idw_wf   = float(np.mean([r["idw_wf_error"]    for r in rows]))
    idw_pgv  = float(np.mean([r["idw_pgv_error"]   for r in rows]))
    print(
        f"\nTiled POD  wf={wf_mean:.4f}±{wf_std:.4f}  pgv={pgv_mean:.4f}±{pgv_std:.4f}"
        f"\nIDW        wf={idw_wf:.4f}                 pgv={idw_pgv:.4f}"
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved → {output_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DATA_PROCESSED / "events")
    p.add_argument("--raw-root", type=Path, default=DATA_RAW)
    p.add_argument("--band", choices=sorted(FREQUENCY_BANDS), default=DEFAULT_MID_BAND)
    p.add_argument("--n-observed", type=int, default=DEFAULT_N_OBSERVED)
    p.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    p.add_argument("--rank", type=int, default=DEFAULT_TILE_RANK,
                   help="POD rank per tile (default 20; use ≤ n_observed//4 for stability)")
    p.add_argument("--overlap-fraction", type=float, default=DEFAULT_OVERLAP_FRACTION,
                   help="Fraction of half-extent used for raised-cosine overlap (default 0.25)")
    p.add_argument("--regularization", type=float, default=1e-6)
    p.add_argument("--max-events", type=int, default=100)
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument(
        "--output-csv",
        type=Path,
        default=RESULTS_TABLES / "tiled_pod_midband_mw7.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(
        data_root=args.data_root,
        raw_root=args.raw_root,
        band_name=args.band,
        n_observed=args.n_observed,
        seeds=list(range(args.n_seeds)),
        train_fraction=args.train_fraction,
        max_events=args.max_events,
        rank=args.rank,
        overlap_fraction=args.overlap_fraction,
        regularization=args.regularization,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
