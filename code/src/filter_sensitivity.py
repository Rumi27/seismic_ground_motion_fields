"""Filter sensitivity study: brick-wall vs. tapered, ±1-bin band-edge variation.

The current pipeline uses an FFT brick-wall filter (boolean mask on DFT bins).
With N=118 samples at Δt=0.2535 s the DFT bin width is Δf=0.03342 Hz, so the
nominal low-band edge at 0.25 Hz falls between bin 7 (0.234 Hz) and bin 8
(0.268 Hz).  This script tests four filter variants:

  narrow     bins 0–6   0–0.201 Hz  (remove top boundary bin)
  current    bins 0–7   0–0.234 Hz  paper default
  wide       bins 0–8   0–0.268 Hz  (add one boundary bin)
  hann_edge  bins 0–8   0–0.268 Hz  with raised-cosine taper on bins 6–8

For each variant the script reports:
  * POD energy ranks r_99, r_999 (training set)
  * Gappy POD waveform and PGV error at rank 151, 128 observed stations
    (10 seeds, same random draws across variants for direct comparison)

Usage
-----
    python -m src.filter_sensitivity
    python -m src.filter_sensitivity --n-seeds 10 --rank 151
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .balanced_pod_analysis import DEFAULT_DT_SECONDS
from .config import DATA_PROCESSED, DATA_RAW, RESULTS_TABLES
from .gappy_pod_reconstruction import (
    N_COMPONENTS,
    find_reference_h5,
    load_station_coordinates,
    pgv_from_snapshots,
    relative_frobenius_error,
    select_stations_random,
    solve_gappy_coefficients,
    reconstruct_from_coefficients,
    station_dof_indices,
)
from .pod_feasibility import (
    cumulative_energy,
    event_to_snapshots,
    load_event_array,
    rank_for_energy,
)
from .train_lowfreq_opinf import (
    DEFAULT_EVENT_CLASS,
    assemble_snapshot_matrix,
    compute_pod_basis,
    discover_processed_events,
    split_records,
)

DEFAULT_RANK = 151
DEFAULT_N_OBSERVED = 128
DEFAULT_N_SEEDS = 10


# ---------------------------------------------------------------------------
# Filter variants
# ---------------------------------------------------------------------------

def _rfft_bin_frequencies(n_samples: int, dt: float) -> np.ndarray:
    return np.fft.rfftfreq(n_samples, d=dt)


def bandpass_weighted(
    event: np.ndarray,
    dt: float,
    weights: np.ndarray,  # (n_rfft_bins,) non-negative weights, 0 = zeroed
) -> np.ndarray:
    """Apply a weighted FFT bandpass along the last axis."""
    arr = np.asarray(event, dtype=np.float32)
    spectrum = np.fft.rfft(arr, axis=-1)
    spectrum *= weights.reshape((1,) * (spectrum.ndim - 1) + (weights.size,))
    return np.fft.irfft(spectrum, n=arr.shape[-1], axis=-1).astype(np.float32, copy=False)


def make_filter_weights(
    n_samples: int,
    dt: float,
    variant: str,
) -> tuple[np.ndarray, dict]:
    """Return (weights, info_dict) for a named filter variant.

    Parameters
    ----------
    variant : one of 'narrow', 'current', 'wide', 'hann_edge'
    """
    freqs = _rfft_bin_frequencies(n_samples, dt)
    df = freqs[1] - freqs[0]          # bin spacing (Hz)
    weights = np.zeros(len(freqs), dtype=np.float64)

    if variant == "narrow":
        # bins with freq <= 0.25 - df  (remove the boundary bin just below 0.25 Hz)
        cutoff = 0.25 - df
        mask = freqs <= cutoff
        weights[mask] = 1.0
        actual_freqs = freqs[mask]
        info = {
            "variant": variant,
            "n_bins": int(mask.sum()),
            "f_lo_hz": float(actual_freqs.min()) if mask.any() else 0.0,
            "f_hi_hz": float(actual_freqs.max()) if mask.any() else 0.0,
        }

    elif variant == "current":
        # paper default: strict [0, 0.25] Hz boolean mask
        mask = (freqs >= 0.0) & (freqs <= 0.25)
        weights[mask] = 1.0
        actual_freqs = freqs[mask]
        info = {
            "variant": variant,
            "n_bins": int(mask.sum()),
            "f_lo_hz": float(actual_freqs.min()),
            "f_hi_hz": float(actual_freqs.max()),
        }

    elif variant == "wide":
        # add the first bin above 0.25 Hz
        cutoff = 0.25 + df
        mask = freqs <= cutoff
        weights[mask] = 1.0
        actual_freqs = freqs[mask]
        info = {
            "variant": variant,
            "n_bins": int(mask.sum()),
            "f_lo_hz": float(actual_freqs.min()),
            "f_hi_hz": float(actual_freqs.max()),
        }

    elif variant == "hann_edge":
        # brick-wall up to 0.25 Hz (same as 'current'), then a raised-cosine
        # taper over the two bins immediately above 0.25 Hz.
        # Bin exactly at 0.25 stays at full weight; the next two bins above
        # receive cos²(π k / 4) for k = 1, 2 (weights ≈ 0.854, 0.5, then 0).
        brick = (freqs >= 0.0) & (freqs <= 0.25)
        weights[brick] = 1.0
        taper_bins = np.where((freqs > 0.25) & (freqs <= 0.25 + 2 * df))[0]
        for step, idx in enumerate(taper_bins, start=1):
            weights[idx] = np.cos(np.pi * step / (2 * (len(taper_bins) + 1))) ** 2
        active = weights > 0
        actual_freqs = freqs[active]
        info = {
            "variant": variant,
            "n_bins": int(active.sum()),
            "f_lo_hz": float(actual_freqs.min()),
            "f_hi_hz": float(actual_freqs.max()),
        }

    else:
        raise ValueError(f"Unknown filter variant: {variant!r}")

    return weights.astype(np.float32), info


# ---------------------------------------------------------------------------
# Filtered snapshot loading
# ---------------------------------------------------------------------------

def load_filtered_snapshots(
    record,
    weights: np.ndarray,
) -> np.ndarray:
    """Load, filter with weights, normalise by σ_ev → (N_C*n_stations, n_time)."""
    raw = load_event_array(record.path)
    filtered = bandpass_weighted(raw, dt=record.dt, weights=weights)
    return event_to_snapshots(filtered)


def assemble_filtered_matrix(records: list, weights: np.ndarray) -> np.ndarray:
    blocks = [load_filtered_snapshots(rec, weights) for rec in records]
    return np.concatenate(blocks, axis=1)


# ---------------------------------------------------------------------------
# Single-variant experiment
# ---------------------------------------------------------------------------

def run_variant(
    variant: str,
    train_records: list,
    val_records: list,
    coords: np.ndarray,
    n_stations: int,
    rank: int,
    n_observed: int,
    seeds: list[int],
    regularization: float,
) -> list[dict]:
    dt = train_records[0].dt
    n_samples = 118  # fixed for GMFlow dataset

    weights, filt_info = make_filter_weights(n_samples, dt, variant)
    print(
        f"\n--- {variant}: {filt_info['n_bins']} bins "
        f"({filt_info['f_lo_hz']:.3f}–{filt_info['f_hi_hz']:.3f} Hz) ---"
    )

    # POD rank analysis on training set
    train_matrix = assemble_filtered_matrix(train_records, weights)
    basis_full, singular_values = compute_pod_basis(train_matrix, rank=max(rank, 400))
    energy = cumulative_energy(singular_values)
    r_99 = rank_for_energy(energy, 0.99)
    r_999 = rank_for_energy(energy, 0.999)
    print(f"  r_99={r_99}  r_999={r_999}")

    train_mean = train_matrix.mean(axis=1).astype(np.float32)
    basis = basis_full[:, :rank]

    # Load validation snapshots with this filter
    val_snapshots = [
        (rec, load_filtered_snapshots(rec, weights)) for rec in val_records
    ]

    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        obs_stations = select_stations_random(n_stations, n_observed, rng)
        hid_stations = np.setdiff1d(np.arange(n_stations), obs_stations)
        hid_dofs = station_dof_indices(hid_stations, n_stations)

        wf_errors, pgv_errors = [], []
        for rec, snaps in val_snapshots:
            coeffs, _ = solve_gappy_coefficients(
                basis, snaps, obs_stations, regularization=regularization
            )
            recon = reconstruct_from_coefficients(basis, coeffs)
            wf_errors.append(relative_frobenius_error(snaps[hid_dofs], recon[hid_dofs]))
            pgv_ref = pgv_from_snapshots(snaps, dt=rec.dt)
            pgv_hat = pgv_from_snapshots(recon, dt=rec.dt)
            pgv_errors.append(
                relative_frobenius_error(pgv_ref[hid_stations], pgv_hat[hid_stations])
            )

        rows.append({
            "variant": variant,
            "n_bins": filt_info["n_bins"],
            "f_lo_hz": filt_info["f_lo_hz"],
            "f_hi_hz": filt_info["f_hi_hz"],
            "r_99": r_99,
            "r_999": r_999,
            "rank": rank,
            "n_observed": n_observed,
            "seed": seed,
            "wf_error": float(np.mean(wf_errors)),
            "pgv_error": float(np.mean(pgv_errors)),
        })

    mean_wf  = np.mean([r["wf_error"]  for r in rows])
    mean_pgv = np.mean([r["pgv_error"] for r in rows])
    print(f"  wf={mean_wf:.4f}  pgv={mean_pgv:.4f}  (rank={rank}, n_obs={n_observed})")
    return rows


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    data_root: Path,
    raw_root: Path,
    rank: int,
    n_observed: int,
    seeds: list[int],
    train_fraction: float,
    max_events: int,
    regularization: float,
    output_csv: Path,
) -> None:
    records = discover_processed_events(data_root, event_class=DEFAULT_EVENT_CLASS)
    if max_events:
        records = records[:max_events]
    train_records, val_records = split_records(records, train_fraction=train_fraction, seed=42)
    print(f"Train: {len(train_records)}  Val: {len(val_records)}")

    ref_h5 = find_reference_h5(DEFAULT_EVENT_CLASS, raw_root=raw_root)
    coords = load_station_coordinates(ref_h5)
    n_stations = coords.shape[0]

    variants = ["narrow", "current", "wide", "hann_edge"]
    all_rows: list[dict] = []
    for v in variants:
        all_rows.extend(
            run_variant(
                variant=v,
                train_records=train_records,
                val_records=val_records,
                coords=coords,
                n_stations=n_stations,
                rank=rank,
                n_observed=n_observed,
                seeds=seeds,
                regularization=regularization,
            )
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved → {output_csv}")

    # Summary table
    print("\nSummary (mean over seeds):")
    print(f"{'variant':<12} {'bins':>4} {'f_hi':>6} {'r_99':>5} {'r_999':>6} {'wf':>7} {'pgv':>7}")
    for v in variants:
        vrows = [r for r in all_rows if r["variant"] == v]
        print(
            f"{v:<12} {vrows[0]['n_bins']:>4} {vrows[0]['f_hi_hz']:>6.3f}"
            f" {vrows[0]['r_99']:>5} {vrows[0]['r_999']:>6}"
            f" {np.mean([r['wf_error'] for r in vrows]):>7.4f}"
            f" {np.mean([r['pgv_error'] for r in vrows]):>7.4f}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DATA_PROCESSED / "events")
    p.add_argument("--raw-root", type=Path, default=DATA_RAW)
    p.add_argument("--rank", type=int, default=DEFAULT_RANK)
    p.add_argument("--n-observed", type=int, default=DEFAULT_N_OBSERVED)
    p.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument("--max-events", type=int, default=100)
    p.add_argument("--regularization", type=float, default=1e-8)
    p.add_argument(
        "--output-csv",
        type=Path,
        default=RESULTS_TABLES / "filter_sensitivity_mw7.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(
        data_root=args.data_root,
        raw_root=args.raw_root,
        rank=args.rank,
        n_observed=args.n_observed,
        seeds=list(range(args.n_seeds)),
        train_fraction=args.train_fraction,
        max_events=args.max_events,
        regularization=args.regularization,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
