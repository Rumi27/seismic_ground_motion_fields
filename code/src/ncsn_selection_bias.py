"""NCSN selection-bias experiment.

Tests four ways to choose 128 (or all 143) observed stations from the
143-station NCSN-derived pool:

  ncsn_all_143     — all 143 mapped stations
  ncsn_first_128   — first 128 by catalogue order (current manuscript default)
  ncsn_farthest_128 — greedy farthest-point thinning from the 143 pool
  ncsn_podqr_128   — QR pivot on POD basis rows restricted to the 143 pool
                     (leverage-score / constrained POD-QR)

For each selection, reports waveform error, PGV error, σ_min(V_obs), cond(V_obs),
averaged over all 20 validation events.

Outputs:
  results/tables/ncsn_selection_bias.csv
  results/figures/fig_ncsn_selection_bias.pdf          (bar chart)
  results/figures/fig_ncsn_selection_bias_lines.pdf    (line chart)

Usage
-----
    python -m src.ncsn_selection_bias
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
    N_COMPONENTS,
    N_I, N_J,
    _station_dof_cf,
    load_mapped_station_indices,
    pgv_from_snapshots,
    reconstruct_from_coefficients,
    relative_frobenius_error,
    solve_gappy_coefficients,
    station_coords_km,
    DEFAULT_STATION_MAPPING,
)
from .train_lowfreq_opinf import (
    DEFAULT_BAND,
    DEFAULT_EVENT_CLASS,
    compute_pod_basis,
    assemble_snapshot_matrix,
    discover_processed_events,
    load_bandpassed_snapshots,
    split_records,
)

BASIS_PATH  = DATA_PROCESSED.parents[1] / "outputs/lowfreq_opinf_mw7_rank151_baseline/V.npy"
OUT_CSV      = RESULTS_TABLES / "ncsn_selection_bias.csv"
OUT_FIG      = RESULTS_FIGURES / "fig_ncsn_selection_bias.pdf"
OUT_FIG_LINE = RESULTS_FIGURES / "fig_ncsn_selection_bias_lines.pdf"

RANK            = 151
TRAIN_FRACTION  = 0.8
SEED            = 42
MAX_EVENTS      = 100
LAMBDA          = 1e-8
N_QR_MODES      = 50  # modes used for POD-QR pivot (same as global POD-QR)

FONT_SIZE = 13
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE, "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE, "legend.fontsize": FONT_SIZE - 1,
    "font.weight": "normal",
    "axes.titleweight": "normal",
})


# ── station selection ─────────────────────────────────────────────────────────

def ncsn_farthest_128(ncsn_all: np.ndarray, coords_km: np.ndarray) -> np.ndarray:
    """Greedy farthest-point thinning: select 128 of 143 by max-min distance."""
    n_select = 128
    ncsn_coords = coords_km[ncsn_all]          # (143, 2)
    sel = [0]                                  # start from first in catalogue
    while len(sel) < n_select:
        sel_c = ncsn_coords[sel]
        dists = np.min(
            np.linalg.norm(ncsn_coords[:, None, :] - sel_c[None, :, :], axis=2),
            axis=1,
        )
        dists[sel] = -np.inf
        sel.append(int(np.argmax(dists)))
    return np.sort(ncsn_all[np.array(sel)])


def ncsn_podqr_128(ncsn_all: np.ndarray, basis: np.ndarray, n_stations: int) -> np.ndarray:
    """QR pivot on POD basis rows restricted to the 144 NCSN stations.

    Selects the 128 of the 144 stations that maximally span the first
    N_QR_MODES POD modes — a leverage-score / constrained-QR selection.
    """
    dof_all = _station_dof_cf(ncsn_all, n_stations)
    v_ncsn  = basis[dof_all, :N_QR_MODES].astype(np.float64)     # (3*143, N_QR_MODES)
    n_ncsn  = len(ncsn_all)
    # Block-station form: (N_QR_MODES * N_COMPONENTS, n_ncsn)
    v_blk = v_ncsn.reshape(N_COMPONENTS, n_ncsn, N_QR_MODES)
    v_mat = v_blk.transpose(2, 0, 1).reshape(N_QR_MODES * N_COMPONENTS, n_ncsn)
    _, _, piv = qr(v_mat, mode="economic", pivoting=True, check_finite=False)
    return np.sort(ncsn_all[piv[:128]])


# ── observability metrics ─────────────────────────────────────────────────────

def obs_metrics(basis: np.ndarray, obs: np.ndarray, n_stations: int) -> dict:
    dof = _station_dof_cf(obs, n_stations)
    sv  = np.linalg.svd(basis[dof].astype(np.float64), compute_uv=False)
    return {"sigma_min": float(sv[-1]),
            "cond":      float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf}


# ── reconstruction evaluation ─────────────────────────────────────────────────

def evaluate_layout(label: str, obs: np.ndarray, basis: np.ndarray,
                    val_records: list, band, n_stations: int) -> dict:
    hid = np.setdiff1d(np.arange(n_stations), obs)
    hid_dof = _station_dof_cf(hid, n_stations)

    wf_list, pgv_list = [], []
    for rec in val_records:
        snaps = load_bandpassed_snapshots(rec, band=band)
        coeffs, _ = solve_gappy_coefficients(basis, snaps, obs, LAMBDA)
        recon = reconstruct_from_coefficients(basis, coeffs)
        wf_list.append(relative_frobenius_error(snaps[hid_dof], recon[hid_dof]))
        pgv_t = pgv_from_snapshots(snaps, dt=rec.dt)
        pgv_r = pgv_from_snapshots(recon, dt=rec.dt)
        pgv_list.append(relative_frobenius_error(pgv_t[hid], pgv_r[hid]))

    m = obs_metrics(basis, obs, n_stations)
    row = {
        "layout":      label,
        "n_observed":  len(obs),
        "n_val_events": len(val_records),
        "wf_error":    float(np.mean(wf_list)),
        "pgv_error":   float(np.mean(pgv_list)),
        **m,
    }
    print(f"  {label:<22s}  n={len(obs):3d}  "
          f"wf={row['wf_error']:.4f}  pgv={row['pgv_error']:.4f}  "
          f"σ_min={row['sigma_min']:.2e}  cond={row['cond']:.2e}")
    return row


# ── figure ────────────────────────────────────────────────────────────────────

LABEL_TEXT = ["All143", "First128", "Far128", "QR128"]
BAR_COLORS = ["#8B0000", "#d6604d", "#f4a582", "#fddbc7"]
LINE_COLOR = "#333333"


def _annotate_values(ax, x: np.ndarray, vals: list, logy: bool) -> None:
    for xi, v in zip(x, vals):
        if logy:
            ax.text(xi, v * 1.08, f"{v:.1e}",
                    ha="center", va="bottom", fontsize=FONT_SIZE - 2)
        else:
            ax.text(xi, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=FONT_SIZE - 2)


def _plot_bars(
    ax, x: np.ndarray, vals: list, logy: bool, annotate_values: bool = True
) -> None:
    ax.bar(x, vals, color=BAR_COLORS, zorder=3, edgecolor="white", lw=0.5)
    if annotate_values:
        _annotate_values(ax, x, vals, logy)


def _plot_lines(
    ax, x: np.ndarray, vals: list, logy: bool, annotate_values: bool = True
) -> None:
    ax.plot(x, vals, "-", color=LINE_COLOR, lw=2.0, zorder=2, alpha=0.85)
    for xi, v, c in zip(x, vals, BAR_COLORS):
        ax.plot(xi, v, "o", color=c, ms=9, zorder=4,
                markeredgecolor="white", markeredgewidth=0.8)
    if annotate_values:
        _annotate_values(ax, x, vals, logy)


def _set_tight_linear_ylim(ax, vals: list[float]) -> tuple[float, float]:
    vmin = float(min(vals))
    vmax = float(max(vals))
    span = vmax - vmin
    pad = max(span * 0.25, max(abs(vmin), abs(vmax)) * 0.005)
    ymin = vmin - pad
    ymax = vmax + pad
    ax.set_ylim(ymin, ymax)
    return ymin, ymax


def make_figure(rows: list[dict], out_path: Path, *, style: str = "bar") -> None:
    if style not in {"bar", "line"}:
        raise ValueError(f"style must be 'bar' or 'line', got {style!r}")

    wf   = [r["wf_error"]  for r in rows]
    pgv  = [r["pgv_error"] for r in rows]
    smin = [r["sigma_min"] for r in rows]
    cond = [r["cond"]      for r in rows]

    x = np.arange(len(rows))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.96, bottom=0.18,
                        wspace=0.38, hspace=0.55)

    panel_specs = [
        (axes[0, 0], wf,   "Relative waveform error",
         "(a)  Waveform error",    False),
        (axes[0, 1], pgv,  "Relative PGV error",
         "(b)  PGV error",         False),
        (axes[1, 0], smin, r"$\sigma_\mathrm{min}(\mathbf{V}_\mathrm{obs})$",
         r"(c)  $\sigma_\mathrm{min}$", True),
        (axes[1, 1], cond, r"$\mathrm{cond}(\mathbf{V}_\mathrm{obs})$",
         r"(d)  Condition number",  True),
    ]

    for ax, vals, ylabel, title, logy in panel_specs:
        annotate_values = False
        if style == "bar":
            _plot_bars(ax, x, vals, logy, annotate_values=annotate_values)
        else:
            _plot_lines(ax, x, vals, logy, annotate_values=annotate_values)

        ax.set_xticks(x)
        ax.set_xticklabels(LABEL_TEXT, rotation=0, ha="center")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=6)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(bottom=min(vals) * 0.5, top=max(vals) * 1.5)
        else:
            ymin, ymax = _set_tight_linear_ylim(ax, vals)
            baseline = 0.989 if "Waveform" in ylabel else 1.0
            if ymin <= baseline <= ymax:
                ax.axhline(
                    baseline, color="#888", ls="--", lw=1.0, label="Mean-field"
                )
            else:
                ax.text(
                    0.02,
                    0.95,
                    f"Mean-field={baseline:.3f} (off-scale)",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=FONT_SIZE - 3,
                    color="#666",
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(loc="upper right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved figure → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    from .balanced_pod_analysis import FREQUENCY_BANDS
    band = FREQUENCY_BANDS[DEFAULT_BAND]
    n_stations = N_I * N_J
    coords_km  = station_coords_km(n_stations)

    print("Loading basis …")
    basis = np.load(BASIS_PATH)[:, :RANK]

    print("Loading NCSN mapping …")
    ncsn_all = load_mapped_station_indices(DEFAULT_STATION_MAPPING)
    ncsn_all = ncsn_all[(ncsn_all >= 0) & (ncsn_all < n_stations)]
    print(f"  NCSN pool: {len(ncsn_all)} stations")

    print("Building principled subsets …")
    ncsn_first128   = np.sort(ncsn_all[:128])
    ncsn_far128     = ncsn_farthest_128(ncsn_all, coords_km)
    ncsn_qr128      = ncsn_podqr_128(ncsn_all, basis, n_stations)

    print("Loading validation events …")
    records = discover_processed_events(DATA_PROCESSED / "events",
                                        event_class=DEFAULT_EVENT_CLASS)[:MAX_EVENTS]
    _, val_records = split_records(records, train_fraction=TRAIN_FRACTION, seed=SEED)
    print(f"  Val events: {len(val_records)}")

    print("\nRunning reconstruction experiments …")
    configs = [
        ("ncsn_all_143",      ncsn_all),
        ("ncsn_first_128",    ncsn_first128),
        ("ncsn_farthest_128", ncsn_far128),
        ("ncsn_podqr_128",    ncsn_qr128),
    ]

    rows = []
    for label, obs in configs:
        row = evaluate_layout(label, obs, basis, val_records, band, n_stations)
        rows.append(row)

    # Write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved table  → {OUT_CSV}")

    make_figure(rows, OUT_FIG, style="bar")
    make_figure(rows, OUT_FIG_LINE, style="line")


if __name__ == "__main__":
    main()
