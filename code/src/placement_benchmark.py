"""Placement-strategy benchmark: POD-QR vs D-optimality, leverage-score, and SGR.

Implements and compares five sensor-placement strategies at n=128 stations:

  random          uniform random (5 seeds, existing)
  farthest        greedy farthest-point geometric (existing)
  pod_qr          QR pivot on rank-50 POD basis rows (existing paper default)
  leverage        non-greedy: rank stations by ||V_r[dofs(s)]||_F² (block Frobenius norm)
  d_optimal       greedy D-optimal: maximise det(Gram) via tr(B_s A_k⁻¹ B_s^T) criterion
  e_optimal       greedy E-optimal / SGR: maximise σ_min via min-eigenvector projection

All methods use rank-151 for the Gram construction; POD-QR uses the paper's
default of rank-50 for pivot ordering (matching the existing result).

For each selection, computes:
  waveform error, PGV error (gappy POD reconstruction, 20 val events)
  σ_min(V_obs), cond(V_obs)

Outputs:
  results/tables/placement_benchmark_mw7.csv
  results/figures/fig_placement_benchmark.pdf

Usage
-----
    python -m src.placement_benchmark
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import qr as sp_qr

from .config import DATA_PROCESSED, RESULTS_FIGURES, RESULTS_TABLES
from .gappy_pod_reconstruction import (
    N_COMPONENTS,
    N_I, N_J,
    _station_dof_cf,
    load_mapped_station_indices,
    pgv_from_snapshots,
    pod_qr_station_order,
    reconstruct_from_coefficients,
    relative_frobenius_error,
    select_stations_farthest,
    select_stations_random,
    solve_gappy_coefficients,
    station_coords_km,
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

BASIS_PATH  = DATA_PROCESSED.parents[1] / "outputs/lowfreq_opinf_mw7_rank151_baseline/V.npy"
OUT_CSV     = RESULTS_TABLES / "placement_benchmark_mw7.csv"
OUT_FIG     = RESULTS_FIGURES / "fig_placement_benchmark.pdf"

RANK          = 151
N_OBSERVED    = 128
N_RANDOM_SEEDS = 5
TRAIN_FRAC    = 0.8
SEED          = 42
MAX_EVENTS    = 100
LAMBDA        = 1e-8
GRAM_LAMBDA   = 1e-8   # regularisation for Gramian in D-opt / E-opt
N_QR_MODES    = 50     # modes for POD-QR pivot (paper default)

FONT_SIZE = 13
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE, "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
})


# ── placement algorithms ───────────────────────────────────────────────────────

def leverage_score(basis: np.ndarray, n_obs: int, n_stations: int) -> np.ndarray:
    """Non-greedy: rank stations by ||V_r[dofs(s)]||_F² (block Frobenius norm)."""
    B = basis.astype(np.float32)
    Bx = B[:n_stations]
    By = B[n_stations:2 * n_stations]
    Bz = B[2 * n_stations:]
    scores = (Bx ** 2).sum(1) + (By ** 2).sum(1) + (Bz ** 2).sum(1)
    return np.sort(np.argsort(scores)[-n_obs:].astype(np.int64))


def d_optimal_greedy(basis: np.ndarray, n_obs: int, n_stations: int,
                     lam: float = GRAM_LAMBDA) -> np.ndarray:
    """Greedy D-optimal: maximise det(Gram) via ||B_s A_k^{-1/2}||_F² criterion.

    Equivalent to maximising the first-order determinant increase
    tr(B_s A_k^{-1} B_s^T) at each step, where A_k is the regularised
    Gramian after k selections.
    """
    r = basis.shape[1]
    basis_f = basis.astype(np.float32)
    A = lam * np.eye(r, dtype=np.float64)
    remaining = np.arange(n_stations, dtype=np.int64)
    selected: list[int] = []

    print(f"    d_optimal: ", end="", flush=True)
    for k in range(n_obs):
        # A_k^{-1/2} via eigen-decomposition
        eigvals, eigvecs = np.linalg.eigh(A)
        A_invsqrt = (eigvecs * (1.0 / np.sqrt(eigvals))[np.newaxis, :]) @ eigvecs.T
        A_invsqrt_f = A_invsqrt.astype(np.float32)

        # Z = basis @ A_k^{-1/2}, shape (n_dof, r)
        Z = basis_f @ A_invsqrt_f

        # Per-station score: ||Z[dofs(s)]||_F²
        rem = remaining
        Zx = Z[rem]
        Zy = Z[rem + n_stations]
        Zz = Z[rem + 2 * n_stations]
        scores = (Zx ** 2).sum(1) + (Zy ** 2).sum(1) + (Zz ** 2).sum(1)

        best_local = int(np.argmax(scores))
        best_global = int(remaining[best_local])
        selected.append(best_global)
        remaining = np.delete(remaining, best_local)

        # Update Gramian with the chosen station's block
        dofs = _station_dof_cf(np.array([best_global]), n_stations)
        Bs = basis[dofs].astype(np.float64)  # (3, r)
        A += Bs.T @ Bs

        if (k + 1) % 16 == 0:
            print(f"{k+1}..", end="", flush=True)
    print(" done")
    return np.sort(np.array(selected, dtype=np.int64))


def e_optimal_sgr(basis: np.ndarray, n_obs: int, n_stations: int,
                  lam: float = GRAM_LAMBDA) -> np.ndarray:
    """Greedy E-optimal / SGR: maximise σ_min via min-eigenvector projection.

    At each step selects the station whose block has the largest projection
    onto the current minimum eigenvector of the Gramian, directly targeting
    the worst-observed POD direction.
    """
    r = basis.shape[1]
    basis_f = basis.astype(np.float32)
    A = lam * np.eye(r, dtype=np.float64)
    remaining = np.arange(n_stations, dtype=np.int64)
    selected: list[int] = []

    print(f"    e_optimal: ", end="", flush=True)
    for k in range(n_obs):
        eigvals, eigvecs = np.linalg.eigh(A)
        u_min = eigvecs[:, 0].astype(np.float32)  # min-eigenvalue direction, (r,)

        rem = remaining
        px = (basis_f[rem] @ u_min) ** 2
        py = (basis_f[rem + n_stations] @ u_min) ** 2
        pz = (basis_f[rem + 2 * n_stations] @ u_min) ** 2
        scores = px + py + pz

        best_local = int(np.argmax(scores))
        best_global = int(remaining[best_local])
        selected.append(best_global)
        remaining = np.delete(remaining, best_local)

        dofs = _station_dof_cf(np.array([best_global]), n_stations)
        Bs = basis[dofs].astype(np.float64)
        A += Bs.T @ Bs

        if (k + 1) % 16 == 0:
            print(f"{k+1}..", end="", flush=True)
    print(" done")
    return np.sort(np.array(selected, dtype=np.int64))


# ── observability and reconstruction ──────────────────────────────────────────

def obs_metrics(basis: np.ndarray, obs: np.ndarray, n_stations: int) -> dict:
    dof = _station_dof_cf(obs, n_stations)
    sv = np.linalg.svd(basis[dof].astype(np.float64), compute_uv=False)
    return {"sigma_min": float(sv[-1]),
            "sigma_max": float(sv[0]),
            "cond":      float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf}


def evaluate(label: str, obs: np.ndarray, basis: np.ndarray,
             val_records: list, n_stations: int) -> dict:
    hid = np.setdiff1d(np.arange(n_stations), obs)
    hid_dof = _station_dof_cf(hid, n_stations)

    wf_list, pgv_list = [], []
    for rec in val_records:
        snaps = load_bandpassed_snapshots(rec, band=DEFAULT_BAND_TUPLE)
        coeffs, _ = solve_gappy_coefficients(basis, snaps, obs, LAMBDA)
        recon = reconstruct_from_coefficients(basis, coeffs)
        wf_list.append(relative_frobenius_error(snaps[hid_dof], recon[hid_dof]))
        pgv_t = pgv_from_snapshots(snaps, dt=rec.dt)
        pgv_r = pgv_from_snapshots(recon, dt=rec.dt)
        pgv_list.append(relative_frobenius_error(pgv_t[hid], pgv_r[hid]))

    m = obs_metrics(basis, obs, n_stations)
    return {
        "method":      label,
        "n_observed":  len(obs),
        "n_val_events": len(val_records),
        "wf_error":    float(np.mean(wf_list)),
        "pgv_error":   float(np.mean(pgv_list)),
        **m,
    }


# ── figure ─────────────────────────────────────────────────────────────────────

def make_figure(rows: list[dict], out_path: Path) -> None:
    # Sort by wf error for display
    ordered = sorted(rows, key=lambda r: r["wf_error"])
    labels = [r["method"] for r in ordered]
    wf     = [r["wf_error"] for r in ordered]
    pgv    = [r["pgv_error"] for r in ordered]
    smin   = [r["sigma_min"] for r in ordered]
    cond   = [r["cond"] for r in ordered]
    wf_std = [float(r.get("wf_std", np.nan)) for r in ordered]
    pgv_std = [float(r.get("pgv_std", np.nan)) for r in ordered]

    COLORS = {
        "random":     "#74add1",
        "farthest":   "#4575b4",
        "leverage":   "#fee090",
        "d_optimal":  "#f46d43",
        "e_optimal":  "#a50026",
        "pod_qr":     "#4dac26",
    }
    DISPLAY = {
        "random": "Random",
        "farthest": "Farthest",
        "leverage": "Leverage",
        "d_optimal": "D-opt",
        "e_optimal": "E-opt",
        "pod_qr": "POD-QR",
    }
    bar_colors = [COLORS.get(m.split("_seed")[0], "#888") for m in labels]
    label_text = [DISPLAY.get(m.split("_seed")[0], m) for m in labels]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.6))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.16, wspace=0.28, hspace=0.36)

    panel_specs = [
        (axes[0, 0], wf, "wf_std", "Relative waveform error", "(a)  Waveform error", False, 0.989),
        (axes[0, 1], pgv, "pgv_std", "Relative PGV error", "(b)  PGV error", False, 1.0),
        (axes[1, 0], smin, None, r"$\sigma_\mathrm{min}(\mathbf{V}_\mathrm{obs})$", r"(c)  $\sigma_\mathrm{min}$", True, None),
        (axes[1, 1], cond, None, r"$\mathrm{cond}(\mathbf{V}_\mathrm{obs})$", r"(d)  Condition number", True, None),
    ]

    for ax, vals, yerr_key, ylabel, title, logy, baseline in panel_specs:
        yerr = None
        if yerr_key == "wf_std":
            yerr = np.array([0.0 if np.isnan(v) else v for v in wf_std], dtype=float)
        elif yerr_key == "pgv_std":
            yerr = np.array([0.0 if np.isnan(v) else v for v in pgv_std], dtype=float)

        ax.bar(
            x, vals, color=bar_colors, zorder=3, edgecolor="white", lw=0.5,
            yerr=yerr, capsize=3 if yerr is not None and np.any(yerr > 0) else 0,
            ecolor="#333333",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(label_text, rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=6)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if logy:
            ax.set_yscale("log")
            ax.set_ylim(bottom=min(vals) * 0.6, top=max(vals) * 1.8)
        else:
            vmin = float(min(vals))
            vmax = float(max(vals))
            span = vmax - vmin
            pad = max(span * 0.25, max(abs(vmin), abs(vmax)) * 0.005)
            ymin = vmin - pad
            ymax = vmax + pad
            ax.set_ylim(ymin, ymax)
            if baseline is not None:
                if ymin <= baseline <= ymax:
                    ax.axhline(baseline, color="#888", ls="--", lw=0.9, alpha=0.7)
                else:
                    ax.text(
                        0.02, 0.95, f"Mean-field={baseline:.3f} (off-scale)",
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=FONT_SIZE - 3, color="#666666",
                    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved figure → {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

DEFAULT_BAND_TUPLE = None  # set in main


def main() -> None:
    global DEFAULT_BAND_TUPLE
    from .balanced_pod_analysis import FREQUENCY_BANDS
    DEFAULT_BAND_TUPLE = FREQUENCY_BANDS[DEFAULT_BAND]

    n_stations = N_I * N_J
    coords_km  = station_coords_km(n_stations)

    print("Loading basis …")
    basis = np.load(BASIS_PATH)[:, :RANK]

    print("Loading events …")
    records = discover_processed_events(DATA_PROCESSED / "events",
                                        event_class=DEFAULT_EVENT_CLASS)[:MAX_EVENTS]
    _, val_records = split_records(records, train_fraction=TRAIN_FRAC, seed=SEED)
    print(f"  Val events: {len(val_records)}")

    # ── compute station selections ───────────────────────────────────────────
    print("\nComputing station selections …")

    # leverage-score (non-greedy)
    t0 = time.time()
    obs_leverage = leverage_score(basis, N_OBSERVED, n_stations)
    print(f"  leverage:   {time.time()-t0:.2f}s")

    # D-optimal greedy
    t0 = time.time()
    obs_d_opt = d_optimal_greedy(basis, N_OBSERVED, n_stations)
    print(f"  d_optimal:  {time.time()-t0:.1f}s total")

    # E-optimal / SGR
    t0 = time.time()
    obs_e_opt = e_optimal_sgr(basis, N_OBSERVED, n_stations)
    print(f"  e_optimal:  {time.time()-t0:.1f}s total")

    # POD-QR (paper default, rank-50 pivot)
    pod_qr_order = pod_qr_station_order(basis, n_stations, n_modes=N_QR_MODES)
    obs_pod_qr = np.sort(pod_qr_order[:N_OBSERVED])

    # random (5 seeds)
    random_selections = []
    for s in range(N_RANDOM_SEEDS):
        rng_seed = SEED * 1000 + s * 17 + N_OBSERVED
        obs_r = select_stations_random(n_stations, N_OBSERVED, np.random.default_rng(rng_seed))
        random_selections.append((f"random_s{s}", obs_r))

    # farthest (1 seed)
    obs_fp = select_stations_farthest(
        coords_km, N_OBSERVED, np.random.default_rng(SEED + 9999))

    # ── observability metrics ────────────────────────────────────────────────
    print("\nObservability diagnostics:")
    for label, obs in [("leverage",  obs_leverage),
                        ("d_optimal", obs_d_opt),
                        ("e_optimal", obs_e_opt),
                        ("pod_qr",    obs_pod_qr),
                        ("farthest",  obs_fp)] + \
                       [(lb, o) for lb, o in random_selections]:
        m = obs_metrics(basis, obs, n_stations)
        print(f"  {label:14s}: sigma_min={m['sigma_min']:.3e}  cond={m['cond']:.3e}")

    # ── reconstruction evaluation ────────────────────────────────────────────
    print("\nRunning gappy POD reconstruction …")
    all_rows: list[dict] = []

    for label, obs in [("leverage",  obs_leverage),
                        ("d_optimal", obs_d_opt),
                        ("e_optimal", obs_e_opt),
                        ("pod_qr",    obs_pod_qr),
                        ("farthest",  obs_fp)] + \
                       [(lb, o) for lb, o in random_selections]:
        print(f"  evaluating {label} …", flush=True)
        row = evaluate(label, obs, basis, val_records, n_stations)
        print(f"    wf={row['wf_error']:.4f}  pgv={row['pgv_error']:.4f}"
              f"  σ_min={row['sigma_min']:.2e}  cond={row['cond']:.2e}")
        all_rows.append(row)

    # Aggregate random across seeds
    rand_rows = [r for r in all_rows if r["method"].startswith("random")]
    rand_wf  = np.mean([r["wf_error"]  for r in rand_rows])
    rand_wf_s = np.std([r["wf_error"]  for r in rand_rows])
    rand_pgv = np.mean([r["pgv_error"] for r in rand_rows])
    rand_pgv_s = np.std([r["pgv_error"] for r in rand_rows])
    rand_sm  = np.mean([r["sigma_min"] for r in rand_rows])
    rand_cond = np.mean([r["cond"]     for r in rand_rows])

    summary_rows = [r for r in all_rows if not r["method"].startswith("random")]
    summary_rows.insert(0, {
        "method": "random", "n_observed": N_OBSERVED,
        "n_val_events": len(val_records),
        "wf_error": rand_wf, "wf_std": rand_wf_s,
        "pgv_error": rand_pgv, "pgv_std": rand_pgv_s,
        "sigma_min": rand_sm, "sigma_max": rand_rows[0]["sigma_max"],
        "cond": rand_cond,
    })
    for r in summary_rows:
        r.setdefault("wf_std", np.nan)
        r.setdefault("pgv_std", np.nan)

    # Write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    all_fields = ["method", "n_observed", "n_val_events",
                  "wf_error", "wf_std", "pgv_error", "pgv_std",
                  "sigma_min", "sigma_max", "cond"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=all_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nSaved table → {OUT_CSV}")

    print("\n=== Summary ===")
    print(f"{'Method':<14} {'WF':>11}  {'PGV':>11}  {'σ_min':>10}  {'cond':>10}")
    print("-" * 65)
    for r in sorted(summary_rows, key=lambda x: x["wf_error"]):
        wf_s = f"±{r['wf_std']:.3f}" if not np.isnan(r.get('wf_std', np.nan)) else ""
        pgv_s = f"±{r['pgv_std']:.3f}" if not np.isnan(r.get('pgv_std', np.nan)) else ""
        print(f"{r['method']:<14} {r['wf_error']:.3f}{wf_s:>6}  "
              f"{r['pgv_error']:.3f}{pgv_s:>6}  "
              f"{r['sigma_min']:.2e}  {r['cond']:.2e}")

    # Figure (exclude per-seed random rows; use aggregated)
    make_figure(summary_rows, OUT_FIG)


if __name__ == "__main__":
    main()
