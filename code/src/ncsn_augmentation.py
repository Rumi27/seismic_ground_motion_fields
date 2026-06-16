"""NCSN network augmentation experiment.

Starting from the 144-station NCSN footprint (fixed), selects k additional
stations from the remaining 62,613 grid nodes by three strategies:

  pod_qr   — QR pivot on the POD basis rows restricted to non-NCSN nodes
              (finds the k non-NCSN stations with highest POD-mode leverage)
  farthest — greedy farthest-point from the existing NCSN footprint
              (fills the largest spatial gaps first)
  random   — uniform random from non-NCSN nodes (5 seeds)

For each k in K_LIST the combined (NCSN + k augmentation) station set is
evaluated with gappy POD at rank 151, reporting waveform error, PGV error,
sigma_min(V_obs), and cond(V_obs).

Outputs:
  results/tables/ncsn_augmentation_mw7.csv
  results/figures/fig_ncsn_augmentation.pdf

Usage
-----
    python -m src.ncsn_augmentation
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import qr

from .config import DATA_PROCESSED, RESULTS_FIGURES, RESULTS_TABLES
from .gappy_pod_reconstruction import (
    DEFAULT_STATION_MAPPING,
    N_COMPONENTS,
    N_I, N_J,
    _station_dof_cf,
    load_mapped_station_indices,
    pgv_from_snapshots,
    reconstruct_from_coefficients,
    relative_frobenius_error,
    solve_gappy_coefficients,
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

BASIS_PATH  = DATA_PROCESSED.parents[1] / "outputs/lowfreq_opinf_mw7_rank151_baseline/V.npy"
OUT_CSV     = RESULTS_TABLES / "ncsn_augmentation_mw7.csv"
OUT_FIG     = RESULTS_FIGURES / "fig_ncsn_augmentation.pdf"

RANK            = 151
N_QR_MODES      = 50       # modes used for POD-QR pivot
LAMBDA          = 1e-8
MAX_EVENTS      = 100
TRAIN_FRAC      = 0.8
SPLIT_SEED      = 42
N_RANDOM_SEEDS  = 5
K_LIST          = [0, 16, 32, 64, 128, 192, 256]   # augmentation counts

FONT_SIZE = 13
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE, "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE, "legend.fontsize": FONT_SIZE - 1,
})


# ── station selection helpers ─────────────────────────────────────────────────

def podqr_augmentation_order(
    basis: np.ndarray,
    avail_stations: np.ndarray,
    n_stations: int,
    n_modes: int = N_QR_MODES,
) -> np.ndarray:
    """QR-pivot order over non-NCSN stations (station indices, not DOFs)."""
    dofs = _station_dof_cf(avail_stations, n_stations)
    v_avail = basis[dofs, :n_modes].astype(np.float64)
    # block to (n_modes * N_COMPONENTS, n_avail) for joint pivot
    n_av = len(avail_stations)
    v_blk = v_avail.reshape(N_COMPONENTS, n_av, n_modes)
    v_mat = v_blk.transpose(2, 0, 1).reshape(n_modes * N_COMPONENTS, n_av)
    _, _, piv = qr(v_mat, mode="economic", pivoting=True, check_finite=False)
    return avail_stations[piv]


def farthest_augmentation_order(
    avail_stations: np.ndarray,
    fixed_stations: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    """Greedy farthest-point from the fixed (NCSN) footprint into available."""
    fixed_c  = coords[fixed_stations]    # (n_fixed, 2)
    avail_c  = coords[avail_stations]    # (n_avail, 2)
    selected: list[int] = []
    # Initial min-distances from NCSN
    min_dists = np.min(
        np.linalg.norm(avail_c[:, None, :] - fixed_c[None, :, :], axis=2),
        axis=1,
    )
    for _ in range(len(avail_stations)):
        idx = int(np.argmax(min_dists))
        selected.append(idx)
        min_dists[idx] = -np.inf
        d_new = np.linalg.norm(avail_c - avail_c[idx], axis=1)
        min_dists = np.minimum(min_dists, d_new)
    return avail_stations[np.array(selected)]


# ── observability metrics ──────────────────────────────────────────────────────

def obs_metrics(basis: np.ndarray, obs: np.ndarray, n_stations: int) -> dict:
    dof = _station_dof_cf(obs, n_stations)
    sv  = np.linalg.svd(basis[dof].astype(np.float64), compute_uv=False)
    return {
        "sigma_min": float(sv[-1]),
        "cond":      float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf,
    }


# ── reconstruction evaluation ─────────────────────────────────────────────────

def evaluate_layout(
    obs: np.ndarray,
    basis: np.ndarray,
    val_data: list,
    n_stations: int,
) -> dict:
    hid     = np.setdiff1d(np.arange(n_stations), obs)
    hid_dof = _station_dof_cf(hid, n_stations)
    wf_list, pgv_list = [], []
    for rec, snaps in val_data:
        coeffs, _ = solve_gappy_coefficients(basis, snaps, obs, LAMBDA)
        recon = reconstruct_from_coefficients(basis, coeffs)
        wf_list.append(relative_frobenius_error(snaps[hid_dof], recon[hid_dof]))
        pgv_t = pgv_from_snapshots(snaps, dt=rec.dt)
        pgv_r = pgv_from_snapshots(recon, dt=rec.dt)
        pgv_list.append(relative_frobenius_error(pgv_t[hid], pgv_r[hid]))
    m = obs_metrics(basis, obs, n_stations)
    return {
        "wf_error":  float(np.mean(wf_list)),
        "pgv_error": float(np.mean(pgv_list)),
        **m,
    }


# ── main sweep ────────────────────────────────────────────────────────────────

def main() -> None:
    band       = FREQUENCY_BANDS[DEFAULT_BAND]
    n_stations = N_I * N_J
    coords     = station_coords_km(n_stations)

    print("Loading basis …")
    basis = np.load(BASIS_PATH)[:, :RANK]

    print("Loading NCSN mapping …")
    ncsn_all = load_mapped_station_indices(DEFAULT_STATION_MAPPING)
    ncsn_all = np.sort(ncsn_all[(ncsn_all >= 0) & (ncsn_all < n_stations)])
    print(f"  NCSN pool: {len(ncsn_all)} stations")

    avail = np.setdiff1d(np.arange(n_stations), ncsn_all)
    print(f"  Available for augmentation: {len(avail)} stations")

    print("Computing POD-QR augmentation order …")
    qr_order   = podqr_augmentation_order(basis, avail, n_stations)
    print("Computing farthest-point augmentation order …")
    fp_order   = farthest_augmentation_order(avail, ncsn_all, coords)

    print("Loading validation events …")
    records = discover_processed_events(
        DATA_PROCESSED / "events", event_class=DEFAULT_EVENT_CLASS
    )[:MAX_EVENTS]
    _, val_records = split_records(records, TRAIN_FRAC, SPLIT_SEED)
    val_data = [(rec, load_bandpassed_snapshots(rec, band=band))
                for rec in val_records]
    print(f"  {len(val_data)} val events loaded.")

    rows: list[dict] = []
    total_k = len(K_LIST)

    for i, k in enumerate(K_LIST):
        print(f"\n[{i+1}/{total_k}] k={k} augmentation stations "
              f"(total obs = {len(ncsn_all) + k})")

        # POD-QR augmentation
        obs_qr = np.sort(np.concatenate([ncsn_all, qr_order[:k]])) if k > 0 else ncsn_all
        r = evaluate_layout(obs_qr, basis, val_data, n_stations)
        rows.append({"strategy": "pod_qr", "seed": 0, "k_aug": k,
                     "n_total": len(obs_qr), **r})
        print(f"  pod_qr   wf={r['wf_error']:.4f}  pgv={r['pgv_error']:.4f}"
              f"  cond={r['cond']:.2e}")

        # Farthest-point augmentation
        obs_fp = np.sort(np.concatenate([ncsn_all, fp_order[:k]])) if k > 0 else ncsn_all
        r = evaluate_layout(obs_fp, basis, val_data, n_stations)
        rows.append({"strategy": "farthest", "seed": 0, "k_aug": k,
                     "n_total": len(obs_fp), **r})
        print(f"  farthest wf={r['wf_error']:.4f}  pgv={r['pgv_error']:.4f}"
              f"  cond={r['cond']:.2e}")

        # Random augmentation (5 seeds)
        for s in range(N_RANDOM_SEEDS):
            rng     = np.random.default_rng(SPLIT_SEED * 100 + s * 7 + k)
            aug_idx = rng.choice(len(avail), size=k, replace=False) if k > 0 else np.array([], dtype=int)
            obs_rnd = np.sort(np.concatenate([ncsn_all, avail[aug_idx]])) if k > 0 else ncsn_all
            r = evaluate_layout(obs_rnd, basis, val_data, n_stations)
            rows.append({"strategy": "random", "seed": s, "k_aug": k,
                         "n_total": len(obs_rnd), **r})
        rnd_wf = np.mean([r["wf_error"] for r in rows
                          if r["strategy"] == "random" and r["k_aug"] == k])
        print(f"  random   wf={rnd_wf:.4f}  (mean over {N_RANDOM_SEEDS} seeds)")

    # Write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved CSV → {OUT_CSV}")

    make_figure(rows, OUT_FIG, len(ncsn_all))


# ── figure ────────────────────────────────────────────────────────────────────

def make_figure(rows: list[dict], out_path: Path, n_ncsn: int) -> None:
    import pandas as pd
    df = pd.DataFrame(rows)

    qr_df  = df[df["strategy"] == "pod_qr"].sort_values("k_aug")
    fp_df  = df[df["strategy"] == "farthest"].sort_values("k_aug")
    rnd_df = df[df["strategy"] == "random"].groupby("k_aug").agg(
        wf_mean=("wf_error",  "mean"), wf_std=("wf_error",  "std"),
        pgv_mean=("pgv_error", "mean"), pgv_std=("pgv_error", "std"),
        cond_mean=("cond",     "mean"),
    ).reset_index()

    fig, axes = plt.subplots(3, 1, figsize=(10, 9.0), sharex=True)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.97, bottom=0.08, hspace=0.28)

    k_vals = sorted(df["k_aug"].unique())
    panel_specs = [
        ("wf_error",  "wf_error",  "wf_mean",  "wf_std",
         "Relative waveform error",  "(a)  Waveform error"),
        ("pgv_error", "pgv_error", "pgv_mean", "pgv_std",
         "Relative PGV error",       "(b)  PGV error"),
        ("cond",      "cond",      "cond_mean", None,
         r"$\mathrm{cond}(\mathbf{V}_\mathrm{obs})$",
         r"(c)  Condition number"),
    ]

    for ax, (ycol_qr, ycol_fp, ycol_rnd, ycol_std, ylabel, title) in zip(
        axes, panel_specs
    ):
        ax.plot(qr_df["k_aug"], qr_df[ycol_qr], "o-",
                color="#2ca02c", lw=2, ms=6, label="POD-QR augment.")
        ax.plot(fp_df["k_aug"], fp_df[ycol_fp], "s--",
                color="#1f77b4", lw=2, ms=6, label="Farthest-point augment.")
        ax.plot(rnd_df["k_aug"], rnd_df[ycol_rnd], "^:",
                color="#d62728", lw=2, ms=6, label="Random augment. (mean)")
        if ycol_std:
            ax.fill_between(
                rnd_df["k_aug"],
                rnd_df[ycol_rnd] - rnd_df[ycol_std],
                rnd_df[ycol_rnd] + rnd_df[ycol_std],
                color="#d62728", alpha=0.15,
            )
        if ycol_qr != "cond":
            ax.axhline(0.989 if "waveform" in ylabel.lower() else 1.0,
                       color="#888", ls="--", lw=1.0, label="Mean-field baseline")
            # reference: pure random layout at n=128
            ref_val = 0.268 if "waveform" in ylabel.lower() else 0.093
            ax.axhline(ref_val, color="#555", ls=":", lw=1.2,
                       label=f"Random-128 ref ({ref_val})")
        else:
            ax.set_yscale("log")
            ax.axhline(50, color="#555", ls=":", lw=1.2,
                       label="Random-128 cond (≈50)")

        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=6)
        ax.set_xticks(k_vals)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(f"Augmentation stations $k$ (total = {n_ncsn} + $k$)")
    axes[0].legend(loc="upper right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    print(f"Saved figure → {out_path}")


if __name__ == "__main__":
    main()
