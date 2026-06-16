"""Rank × station-count scaling sweep for gappy POD (low-frequency Mw7).

Sweeps ranks r ∈ {50, 100, 151, 200, 250, 300, 336} and
n_obs ∈ {32, 64, 128, 256, 512} for three layouts (random, farthest, pod_qr)
to support network-design trade-off analysis.

The rank-336 basis (V.npy) is loaded once and sliced per rank.
Validation events are pre-loaded to avoid repeated file I/O.

Outputs:
  results/tables/rank_station_scaling_mw7.csv
  results/figures/fig_rank_station_scaling.pdf

Usage
-----
    python -m src.rank_station_scaling
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .config import DATA_PROCESSED, RESULTS_FIGURES, RESULTS_TABLES
from .gappy_pod_reconstruction import (
    N_COMPONENTS,
    evaluate_event,
    pod_qr_station_order,
    select_stations_farthest,
    select_stations_pod_qr,
    select_stations_random,
    station_coords_km,
)
from .train_lowfreq_opinf import (
    DEFAULT_BAND,
    DEFAULT_EVENT_CLASS,
    discover_processed_events,
    load_bandpassed_snapshots,
    split_records,
)
from .balanced_pod_analysis import FREQUENCY_BANDS

BASIS_PATH  = DATA_PROCESSED.parents[1] / "outputs/lowfreq_opinf_mw7_rank336_r999/V.npy"
OUT_CSV     = RESULTS_TABLES / "rank_station_scaling_mw7.csv"
OUT_FIG     = RESULTS_FIGURES / "fig_rank_station_scaling.pdf"

RANKS       = [50, 100, 151, 200, 250, 300, 336]
N_OBS_LIST  = [32, 64, 128, 256, 512]
N_RANDOM_SEEDS = 5
REGULARIZATION = 1e-8
MAX_EVENTS  = 100
TRAIN_FRAC  = 0.8
SPLIT_SEED  = 42


# ── sweep ─────────────────────────────────────────────────────────────────────

def run_sweep(
    basis_full: np.ndarray,
    val_data: list,
    n_stations: int,
    coords: np.ndarray,
    dt: float,
) -> list[dict]:
    rows: list[dict] = []
    total = len(RANKS) * len(N_OBS_LIST)
    done = 0

    for rank in RANKS:
        basis = basis_full[:, :rank]
        # POD-QR pivot order is rank-dependent
        qr_order = pod_qr_station_order(basis, n_stations)

        for n_obs in N_OBS_LIST:
            t0 = time.perf_counter()
            done += 1

            # --- Random: N_RANDOM_SEEDS seeds ---
            for s in range(N_RANDOM_SEEDS):
                rng = np.random.default_rng(SPLIT_SEED * 1000 + s * 17 + n_obs + rank)
                obs = select_stations_random(n_stations, n_obs, rng)
                metrics = [
                    evaluate_event(basis, snaps, obs, dt=dt,
                                   regularization=REGULARIZATION)
                    for _, snaps in val_data
                ]
                rows.append(_agg("random", s, rank, n_obs, n_stations, metrics))

            # --- Farthest-point ---
            rng_fp = np.random.default_rng(SPLIT_SEED + rank * 13 + 9999)
            obs_fp = select_stations_farthest(coords, n_obs, rng_fp)
            metrics_fp = [
                evaluate_event(basis, snaps, obs_fp, dt=dt,
                               regularization=REGULARIZATION)
                for _, snaps in val_data
            ]
            rows.append(_agg("farthest", 0, rank, n_obs, n_stations, metrics_fp))

            # --- POD-QR ---
            obs_qr = select_stations_pod_qr(qr_order, n_obs, n_stations)
            metrics_qr = [
                evaluate_event(basis, snaps, obs_qr, dt=dt,
                               regularization=REGULARIZATION)
                for _, snaps in val_data
            ]
            rows.append(_agg("pod_qr", 0, rank, n_obs, n_stations, metrics_qr))

            elapsed = time.perf_counter() - t0
            rand_wf = np.mean([r["wf_error"] for r in rows
                               if r["rank"] == rank and r["n_obs"] == n_obs
                               and r["layout"] == "random"])
            print(
                f"[{done:3d}/{total}] r={rank:3d} n={n_obs:3d}  "
                f"random_wf={rand_wf:.3f}  elapsed={elapsed:.1f}s",
                flush=True,
            )

    return rows


def _agg(
    layout: str,
    seed: int,
    rank: int,
    n_obs: int,
    n_stations: int,
    metrics: list[dict],
) -> dict:
    wf_vals  = [m["relative_error_heldout"] for m in metrics]
    pgv_vals = [m["pgv_error_heldout"]      for m in metrics]
    od = (N_COMPONENTS * n_obs) / rank        # overdetermination ratio
    return {
        "layout":   layout,
        "seed":     seed,
        "rank":     rank,
        "n_obs":    n_obs,
        "od_ratio": round(od, 4),
        "wf_error": round(float(np.mean(wf_vals)),  4),
        "wf_std":   round(float(np.std(wf_vals)),   4),
        "pgv_error": round(float(np.mean(pgv_vals)), 4),
        "pgv_std":  round(float(np.std(pgv_vals)),  4),
        "n_val_events": len(metrics),
    }


# ── figure ────────────────────────────────────────────────────────────────────

# colours for rank curves (dark-to-light blue → green → orange → red)
RANK_COLORS = {
    50:  "#d4e6f1",
    100: "#85c1e9",
    151: "#2980b9",
    200: "#1a5276",
    250: "#196f3d",
    300: "#f39c12",
    336: "#c0392b",
}
# colours for n_obs curves (light-to-dark teal)
NOBS_COLORS = {
    32:  "#b2dfdb",
    64:  "#4db6ac",
    128: "#00897b",
    256: "#00695c",
    512: "#004d40",
}

FONT_SIZE = 18
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 2,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
})


def make_figure(df_all: list[dict], out_path: Path) -> None:
    import pandas as pd
    df = pd.DataFrame(df_all)

    rand = (
        df[df["layout"] == "random"]
        .groupby(["rank", "n_obs"])
        .agg(wf_mean=("wf_error", "mean"), wf_std=("wf_error", "std"))
        .reset_index()
        .rename(columns={"wf_mean": "wf_error"})
    )
    qr = df[df["layout"] == "pod_qr"].copy()

    # 2-panel heatmap: much cleaner than 10+ line legends
    rand_grid = (
        rand.pivot(index="rank", columns="n_obs", values="wf_error")
        .reindex(index=RANKS, columns=N_OBS_LIST)
        .to_numpy()
    )
    qr_grid = (
        qr.pivot(index="rank", columns="n_obs", values="wf_error")
        .reindex(index=RANKS, columns=N_OBS_LIST)
        .to_numpy()
    )

    vmin = float(np.nanmin([rand_grid.min(), qr_grid.min()]))
    vmax = float(np.nanmax([rand_grid.max(), qr_grid.max()]))

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 7.6), constrained_layout=True)
    cmap = "viridis_r"

    for ax, grid, title in [
        (axes[0], rand_grid, "(a) Random layout"),
        (axes[1], qr_grid, "(b) POD-QR layout"),
    ]:
        im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(N_OBS_LIST)))
        ax.set_xticklabels([str(n) for n in N_OBS_LIST])
        ax.set_yticks(np.arange(len(RANKS)))
        ax.set_yticklabels([str(r) for r in RANKS])
        ax.set_xlabel("Observed stations, n")
        ax.set_ylabel("POD rank, r")
        ax.set_title(title, loc="left", pad=8)
        # subtle cell boundaries
        ax.set_xticks(np.arange(-0.5, len(N_OBS_LIST), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(RANKS), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8, alpha=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=axes, shrink=0.95, pad=0.02)
    cbar.set_label("Relative waveform error")
    cbar.ax.tick_params(labelsize=FONT_SIZE - 1)

    fig.suptitle(
        "Rank-station scaling summary (Mw7, 0-0.25 Hz)\nLower color = lower error",
        fontsize=14,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved figure → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    band = FREQUENCY_BANDS[DEFAULT_BAND]
    print("Loading basis …")
    basis_full = np.load(BASIS_PATH)   # (n_dof, 336)
    print(f"  Basis shape: {basis_full.shape}")

    print("Loading events …")
    records = discover_processed_events(
        DATA_PROCESSED / "events", event_class=DEFAULT_EVENT_CLASS
    )[:MAX_EVENTS]
    _, val_records = split_records(records, train_fraction=TRAIN_FRAC, seed=SPLIT_SEED)
    n_stations = basis_full.shape[0] // N_COMPONENTS
    coords = station_coords_km(n_stations)
    dt = val_records[0].dt

    print(f"  Pre-loading {len(val_records)} val events …")
    val_data = [
        (rec, load_bandpassed_snapshots(rec, band=band)) for rec in val_records
    ]
    print("  Done.")

    print(f"\nRunning {len(RANKS)} ranks × {len(N_OBS_LIST)} n_obs "
          f"× 3 layouts × (5/1/1 seeds) …")
    t_start = time.perf_counter()
    rows = run_sweep(basis_full, val_data, n_stations, coords, dt)
    print(f"\nSweep done in {(time.perf_counter()-t_start)/60:.1f} min")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved CSV → {OUT_CSV}")

    make_figure(rows, OUT_FIG)


if __name__ == "__main__":
    main()
