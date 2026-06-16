"""Compute σ_min and cond(V_obs) across station counts and layouts.

Reads the existing station_count_sweep_mw7_lowfreq.csv for wf/PGV, then
recomputes the same station selections deterministically and evaluates the
observability metrics (σ_min, cond) of the observed POD submatrix.

Outputs:
  results/tables/station_count_observability_mw7.csv
  results/figures/fig_station_count_observability.pdf

Usage
-----
    python -m src.station_count_observability
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd

from .config import DATA_PROCESSED, RESULTS_FIGURES, RESULTS_TABLES
from .gappy_pod_reconstruction import (
    DEFAULT_STATION_MAPPING,
    N_COMPONENTS,
    N_I, N_J,
    _station_dof_cf,
    load_mapped_station_indices,
    pod_qr_station_order,
    select_stations_farthest,
    select_stations_ncsn,
    select_stations_random,
    station_coords_km,
)

BASIS_PATH   = DATA_PROCESSED.parents[1] / "outputs/lowfreq_opinf_mw7_rank151_baseline/V.npy"
SWEEP_CSV    = RESULTS_TABLES / "station_count_sweep_mw7_lowfreq.csv"
OUT_CSV      = RESULTS_TABLES / "station_count_observability_mw7.csv"
OUT_FIG      = RESULTS_FIGURES / "fig_station_count_observability.pdf"

RANK         = 151
OUTER_SEED   = 42       # train/val split seed (same as sweep)
N_RANDOM_SEEDS = 5      # match station_count_sweep.py
N_OBS_LIST   = [32, 64, 96, 128, 192, 256, 384, 512]

FONT_SIZE = 18
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 2, "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE, "legend.fontsize": FONT_SIZE - 1,
    "font.weight": "normal",
    "axes.titleweight": "normal",
})


def observability_metrics(basis: np.ndarray, obs_stations: np.ndarray, n_stations: int) -> dict:
    obs_dof = _station_dof_cf(obs_stations, n_stations)
    v_obs = basis[obs_dof].astype(np.float64)
    sv = np.linalg.svd(v_obs, compute_uv=False)
    sigma_min = float(sv[-1])
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
    return {"sigma_min": sigma_min, "cond": cond}


def build_observability_table(basis: np.ndarray, n_stations: int,
                               mapping_csv: Path) -> list[dict]:
    coords = station_coords_km(n_stations)
    mapped = load_mapped_station_indices(mapping_csv)
    mapped = mapped[(mapped >= 0) & (mapped < n_stations)]
    pod_qr_order = pod_qr_station_order(basis, n_stations)

    rows = []
    for n_obs in N_OBS_LIST:
        # --- random: 5 seeds matching station_count_sweep.py RNG schedule ---
        for s in range(N_RANDOM_SEEDS):
            rng_seed = OUTER_SEED * 1000 + s * 17 + n_obs
            rng = np.random.default_rng(rng_seed)
            obs = select_stations_random(n_stations, n_obs, rng)
            m = observability_metrics(basis, obs, n_stations)
            rows.append({"layout": "random", "seed": s,
                         "n_observed_stations": n_obs, **m})

        # --- pod_qr ---
        obs_qr = np.sort(pod_qr_order[:n_obs])
        m = observability_metrics(basis, obs_qr, n_stations)
        rows.append({"layout": "pod_qr", "seed": 0,
                     "n_observed_stations": n_obs, **m})

        # --- farthest-point: replicate sweep seed schedule (seed+9999 per n_obs) ---
        rng_fp = np.random.default_rng(OUTER_SEED + 9999)
        obs_fp = select_stations_farthest(coords, n_obs, rng_fp)
        m = observability_metrics(basis, obs_fp, n_stations)
        rows.append({"layout": "farthest", "seed": 0,
                     "n_observed_stations": n_obs, **m})

        # --- ncsn: capped at len(mapped) ---
        n_ncsn = min(n_obs, len(mapped))
        obs_ncsn = select_stations_ncsn(mapped, n_ncsn, n_stations)
        m = observability_metrics(basis, obs_ncsn, n_stations)
        rows.append({"layout": "ncsn", "seed": 0,
                     "n_observed_stations": n_obs, **m})

        print(f"  n_obs={n_obs:4d}  done", flush=True)

    return rows


def merge_with_sweep(obs_rows: list[dict]) -> pd.DataFrame:
    sweep = pd.read_csv(SWEEP_CSV)
    sweep = sweep[sweep["rank"] == RANK]
    obs_df = pd.DataFrame(obs_rows)

    merged = pd.merge(
        sweep[["layout", "seed", "n_observed_stations",
               "mean_rel_error_heldout", "mean_pgv_error_heldout"]],
        obs_df,
        on=["layout", "seed", "n_observed_stations"],
        how="inner",
    )
    return merged


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std over seeds per (layout, n_obs)."""
    g = df.groupby(["layout", "n_observed_stations"])
    agg = g.agg(
        wf_mean=("mean_rel_error_heldout", "mean"),
        wf_std=("mean_rel_error_heldout", "std"),
        pgv_mean=("mean_pgv_error_heldout", "mean"),
        pgv_std=("mean_pgv_error_heldout", "std"),
        sigma_min_mean=("sigma_min", "mean"),
        sigma_min_std=("sigma_min", "std"),
        cond_mean=("cond", "mean"),
        cond_std=("cond", "std"),
    ).reset_index()
    return agg


def make_figure(agg: pd.DataFrame, out_path: Path) -> None:
    layouts = ["random", "farthest", "pod_qr", "ncsn"]
    colors  = {"random": "#2166ac", "farthest": "#d62728",
               "pod_qr": "#4dac26",  "ncsn": "#9467bd"}
    labels  = {"random": "Random (5 seeds)", "farthest": "Farthest-point",
               "pod_qr": "POD-QR",           "ncsn": "NCSN-derived"}
    lstyles = {"random": "-", "farthest": "--", "pod_qr": "-.", "ncsn": ":"}

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.93, bottom=0.10,
                        hspace=0.38, wspace=0.30)

    panels = [
        (axes[0, 0], "wf_mean",        "wf_std",        "(a)  Waveform error",
         "Relative waveform error",     False, (0, 1.4)),
        (axes[0, 1], "pgv_mean",       "pgv_std",        "(b)  PGV error",
         "Relative PGV error",          False, (0, 1.1)),
        (axes[1, 0], "sigma_min_mean", "sigma_min_std",  r"(c)  $\sigma_\mathrm{min}(\mathbf{V}_\mathrm{obs})$",
         r"$\sigma_\mathrm{min}$",      True,  None),
        (axes[1, 1], "cond_mean",      "cond_std",       r"(d)  $\mathrm{cond}(\mathbf{V}_\mathrm{obs})$",
         r"Condition number",           True,  None),
    ]

    for ax, col_m, col_s, title, ylabel, logy, ylim in panels:
        for lay in layouts:
            sub = agg[agg["layout"] == lay].sort_values("n_observed_stations")
            x   = sub["n_observed_stations"].values
            y_m = sub[col_m].values
            y_s = sub[col_s].fillna(0).values

            ax.plot(x, y_m, color=colors[lay], ls=lstyles[lay],
                    lw=2.6, marker="o", ms=7, zorder=3, label=labels[lay])
            if lay == "random" and y_s.any():
                ax.fill_between(x, y_m - y_s, y_m + y_s,
                                color=colors[lay], alpha=0.15, zorder=2)

        # OD thresholds and mean-field baseline for error panels
        if col_m in ("wf_mean", "pgv_mean"):
            y_top = ylim[1] if ylim else 1.4
            ax.axvline(51,  color="grey", ls=":", lw=1.0, alpha=0.7)
            ax.axvline(128, color="grey", ls=":", lw=1.0, alpha=0.7)
            ax.text(53,  y_top * 0.97, "OD=1",   fontsize=FONT_SIZE - 3, color="grey", va="top")
            ax.text(130, y_top * 0.97, "OD≈2.5", fontsize=FONT_SIZE - 3, color="grey", va="top")
        if col_m == "wf_mean":
            ax.axhline(0.989, color="#888", ls="--", lw=0.9, alpha=0.7,
                       label="Mean-field baseline")

        ax.set_xscale("log", base=2)
        ax.set_xticks([32, 64, 128, 256, 512])
        ax.set_xticklabels(["32", "64", "128", "256", "512"])
        ax.set_xlabel("Number of observed stations")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=6)
        ax.grid(alpha=0.25, zorder=0)
        if logy:
            ax.set_yscale("log")
        elif ylim is not None:
            ax.set_ylim(ylim)

    # legend on panel (a)
    handles = [mlines.Line2D([], [], color=colors[l], ls=lstyles[l],
                             lw=2.6, marker="o", ms=7, label=labels[l])
               for l in layouts]
    handles.append(mlines.Line2D([], [], color="#888", ls="--", lw=0.9,
                                 label="Mean-field baseline"))
    axes[0, 0].legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.78),
        frameon=False,
        ncol=1,
        labelspacing=0.2,
        borderpad=0.1,
        handletextpad=0.35,
        handlelength=1.0,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved figure → {out_path}")


def make_summary_table(agg: pd.DataFrame, n_vals: list[int]) -> pd.DataFrame:
    rows = []
    for lay in ["random", "farthest", "pod_qr", "ncsn"]:
        for n in n_vals:
            r = agg[(agg["layout"] == lay) & (agg["n_observed_stations"] == n)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append({
                "layout": lay, "n_obs": n,
                "wf": f"{r.wf_mean:.3f}",
                "wf_std": f"±{r.wf_std:.3f}" if not np.isnan(r.wf_std) else "—",
                "pgv": f"{r.pgv_mean:.3f}",
                "pgv_std": f"±{r.pgv_std:.3f}" if not np.isnan(r.pgv_std) else "—",
                "sigma_min": f"{r.sigma_min_mean:.2e}",
                "cond": f"{r.cond_mean:.1e}",
            })
    return pd.DataFrame(rows)


def main() -> None:
    print("Loading POD basis …")
    basis_full = np.load(BASIS_PATH)
    basis = basis_full[:, :RANK]
    n_stations = N_I * N_J

    print("Computing observability metrics …")
    obs_rows = build_observability_table(basis, n_stations, DEFAULT_STATION_MAPPING)

    print("Merging with sweep CSV …")
    merged = merge_with_sweep(obs_rows)
    merged.to_csv(OUT_CSV, index=False)
    print(f"Saved table  → {OUT_CSV}")

    agg = aggregate(merged)

    make_figure(agg, OUT_FIG)

    summary = make_summary_table(agg, [32, 64, 128, 256, 512])
    print("\n=== Summary at reviewer-requested n values ===")
    print(summary.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
