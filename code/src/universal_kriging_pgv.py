"""Universal kriging with GMPE mean for PGV reconstruction.

Standard seismological approach (ShakeMap-style spatial interpolation):
  1. Calibrate a simple GMPE on training events:
       ln(PGV_i^(e)) = c0^(e) + c1*ln(Rhypo_i) + c2*Rhypo_i
     where Rhypo is hypocentral distance in km, c1 captures geometric
     spreading, and c2 captures anelastic attenuation.
  2. For each test event, estimate the per-event intercept from observed
     stations: c0_hat^(e) = mean_i( ln(PGV_obs_i) - c1*ln(Rhypo_i) - c2*Rhypo_i )
  3. Compute within-event spatial residuals at observed stations:
       dW_i = ln(PGV_obs_i) - c0_hat^(e) - c1*ln(Rhypo_i) - c2*Rhypo_i
  4. Ordinary-krige dW from observed to hidden stations via Matérn-3/2 GP.
  5. Reconstruct hidden PGV:
       PGV_j = exp( c0_hat^(e) + c1*ln(Rhypo_j) + c2*Rhypo_j + kriged_dW_j )

Works entirely in the normalized PGV units used throughout the paper.
The per-event normalization (std-scaling in event_to_snapshots) cancels when
computing within-event residuals, so no physical unit conversion is needed.

Grid-spacing calibration: h_x ≈ 200 m/row, h_y ≈ 225 m/col for the 367×171
SW4 station grid (GMFlow/Hayward-fault SFBA configuration). Derived by
matching peak-PGV station locations to hypocenter coordinates from the rupture
realization summary CSV.

Usage
-----
    python -m src.universal_kriging_pgv
    python -m src.universal_kriging_pgv --n-observed 128 --n-seeds 10 --max-events 100
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .balanced_pod_analysis import FREQUENCY_BANDS
from .config import DATA_PROCESSED, DATA_RAW, RESULTS_TABLES
from .gappy_pod_reconstruction import (
    DEFAULT_EVENT_CLASS,
    find_reference_h5,
    load_station_coordinates,
    pgv_from_snapshots,
    relative_frobenius_error,
    select_stations_random,
)
from .train_lowfreq_opinf import (
    DEFAULT_BAND,
    DEFAULT_DT_SECONDS,
    DEFAULT_EVENT_CLASS,
    discover_processed_events,
    load_bandpassed_snapshots,
    split_records,
)

# ------------------------------------------------------------------
# SW4 simulation grid spacing (metres per row/column increment)
# Calibrated from peak-PGV station positions vs rupture hypocenter
# coordinates across multiple Mw7 events.  Uncertainty ±10%.
# ------------------------------------------------------------------
H_X_M: float = 200.0
H_Y_M: float = 225.0

_BATCH = 8000          # GP prediction batch size
_SAMPLE_PER_EVENT = 3000  # stations sampled per event for GMPE calibration


# ------------------------------------------------------------------
# Rupture metadata
# ------------------------------------------------------------------

_HYPO_RE = re.compile(r"\((-?\d+),\s*(-?\d+),\s*(-?\d+)\)")


def parse_rupture_csv(csv_path: Path) -> dict[str, tuple[float, float, float]]:
    """Return {stem → (x_m, y_m, z_m)} hypocenter in simulation metres."""
    df = pd.read_csv(csv_path)
    out: dict[str, tuple[float, float, float]] = {}
    for _, row in df.iterrows():
        m = _HYPO_RE.search(str(row["Hypocenter"]))
        if m is None:
            continue
        stem = str(row["Filename"]).replace(".h5", ".npy")
        out[stem] = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    return out


# ------------------------------------------------------------------
# Physical station coordinates
# ------------------------------------------------------------------


def load_station_coords_m(
    ref_h5: Path,
    h_x: float = H_X_M,
    h_y: float = H_Y_M,
) -> np.ndarray:
    """(n_stations, 2) physical coordinates in metres derived from grid indices."""
    rc = load_station_coordinates(ref_h5)          # (n, 2): (row, col)
    return np.column_stack([
        (rc[:, 0] - 1.0) * h_x,
        (rc[:, 1] - 1.0) * h_y,
    ])


# ------------------------------------------------------------------
# Distance utility
# ------------------------------------------------------------------


def rhypo_km(
    coords_m: np.ndarray,
    hypo_m: tuple[float, float, float],
) -> np.ndarray:
    """Hypocentral distances in km.  coords_m shape (n, 2); hypo_m = (x,y,z) metres."""
    dx_km = (coords_m[:, 0] - hypo_m[0]) / 1000.0
    dy_km = (coords_m[:, 1] - hypo_m[1]) / 1000.0
    dz_km = hypo_m[2] / 1000.0      # depth (always positive)
    return np.sqrt(dx_km ** 2 + dy_km ** 2 + dz_km ** 2)


# ------------------------------------------------------------------
# GMPE calibration
# ------------------------------------------------------------------


def calibrate_gmpe(
    train_records: list,
    rupture: dict[str, tuple[float, float, float]],
    all_coords_m: np.ndarray,
    band: tuple[float, float],
    dt: float,
    n_sample: int = _SAMPLE_PER_EVENT,
    rng_seed: int = 0,
) -> tuple[float, float]:
    """Fit shared attenuation coefficients (c1, c2) via demeaned OLS.

    Model: ln(PGV_i^(e)) = c0^(e) + c1*ln(Rhypo) + c2*Rhypo
    Demeaning within each event removes c0^(e) and yields a clean
    regression of (demeaned ln PGV) on (demeaned [ln R, R]).

    Returns (c1, c2).
    """
    rng = np.random.default_rng(rng_seed)
    n_total = all_coords_m.shape[0]
    idx = np.sort(rng.choice(n_total, size=min(n_sample, n_total), replace=False))
    samp_m = all_coords_m[idx]

    X_dm_all: list[np.ndarray] = []
    y_dm_all: list[np.ndarray] = []

    for rec in train_records:
        if rec.path.name not in rupture:
            continue
        hypo = rupture[rec.path.name]
        snaps = load_bandpassed_snapshots(rec, band=band)
        pgv = pgv_from_snapshots(snaps, dt=dt)[idx]
        mask = pgv > 1e-12
        if mask.sum() < 20:
            continue

        r_km = rhypo_km(samp_m[mask], hypo)
        lnpgv = np.log(pgv[mask])
        X = np.column_stack([np.log(r_km.clip(0.1)), r_km])  # (n, 2)
        # Demean to remove per-event intercept
        X_dm_all.append(X - X.mean(axis=0))
        y_dm_all.append(lnpgv - lnpgv.mean())

    if not X_dm_all:
        print("Warning: no training events matched rupture CSV; using defaults c1=-1, c2=0")
        return -1.0, 0.0

    Xm = np.vstack(X_dm_all)                    # (N_total, 2)
    ym = np.concatenate(y_dm_all)               # (N_total,)
    reg = 1e-6 * np.eye(2)
    coeffs = np.linalg.solve(Xm.T @ Xm + reg, Xm.T @ ym)
    return float(coeffs[0]), float(coeffs[1])


# ------------------------------------------------------------------
# Universal kriging prediction for one event
# ------------------------------------------------------------------


def _krige_residuals(
    coords_obs_km: np.ndarray,   # (n_obs, 2)
    residuals_obs: np.ndarray,   # (n_obs,)
    coords_hid_km: np.ndarray,   # (n_hid, 2)
) -> np.ndarray:
    """Ordinary kriging of within-event residuals via Matérn-3/2 GP."""
    mu_c = coords_obs_km.mean(axis=0)
    sd_c = coords_obs_km.std(axis=0).clip(1e-8)
    X_obs = (coords_obs_km - mu_c) / sd_c
    X_hid = (coords_hid_km - mu_c) / sd_c

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=1.0, length_scale_bounds=(0.05, 20.0), nu=1.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 0.5))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=3,
        normalize_y=True,
        alpha=0.0,
    )
    gp.fit(X_obs, residuals_obs)

    pred = np.empty(len(X_hid), dtype=np.float64)
    for start in range(0, len(X_hid), _BATCH):
        pred[start : start + _BATCH] = gp.predict(X_hid[start : start + _BATCH])
    return pred


def predict_pgv_uk(
    pgv_obs: np.ndarray,
    rhypo_obs_km: np.ndarray,
    coords_obs_km: np.ndarray,
    rhypo_hid_km: np.ndarray,
    coords_hid_km: np.ndarray,
    c1: float,
    c2: float,
) -> np.ndarray:
    """Universal-kriging prediction: GMPE trend + kriged within-event residuals."""
    eps = 1e-12
    lnpgv_obs = np.log(pgv_obs.clip(eps))

    gmpe_obs = c1 * np.log(rhypo_obs_km.clip(0.1)) + c2 * rhypo_obs_km
    c0_hat = float((lnpgv_obs - gmpe_obs).mean())   # per-event intercept
    residuals_obs = lnpgv_obs - c0_hat - gmpe_obs   # within-event residuals

    kriged = _krige_residuals(coords_obs_km, residuals_obs, coords_hid_km)

    gmpe_hid = c1 * np.log(rhypo_hid_km.clip(0.1)) + c2 * rhypo_hid_km
    return np.exp(c0_hat + gmpe_hid + kriged)


# ------------------------------------------------------------------
# Experiment runner
# ------------------------------------------------------------------


def run(
    data_root: Path,
    raw_root: Path,
    band_name: str,
    n_observed: int,
    seeds: list[int],
    train_fraction: float,
    max_events: int,
    output_csv: Path,
) -> None:
    band = FREQUENCY_BANDS[band_name]
    records = discover_processed_events(data_root, event_class=DEFAULT_EVENT_CLASS)
    if max_events:
        records = records[:max_events]

    train_records, val_records = split_records(records, train_fraction=train_fraction, seed=42)
    print(f"Train: {len(train_records)}  Val: {len(val_records)}")

    rupture_csv = raw_root / "meta" / "M7_Testing_rupture_realization_summary.csv"
    rupture = parse_rupture_csv(rupture_csv)

    ref_h5 = find_reference_h5(DEFAULT_EVENT_CLASS, raw_root=raw_root)
    all_coords_m = load_station_coords_m(ref_h5)       # (n_stations, 2) metres
    all_coords_km = all_coords_m / 1000.0
    n_stations = all_coords_m.shape[0]

    print("Calibrating GMPE on training set...")
    c1, c2 = calibrate_gmpe(
        train_records, rupture, all_coords_m, band=band, dt=DEFAULT_DT_SECONDS
    )
    print(f"  c1={c1:.3f}  c2={c2:.5f}  "
          f"(ln(PGV) = c0_event + {c1:.3f}*ln(Rhypo) + {c2:.5f}*Rhypo)")

    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        obs_idx = select_stations_random(n_stations, n_observed, rng)
        hid_mask = np.ones(n_stations, dtype=bool)
        hid_mask[obs_idx] = False
        hid_idx = np.where(hid_mask)[0]

        coords_obs_km = all_coords_km[obs_idx]
        coords_hid_km = all_coords_km[hid_idx]

        pgv_errors = []
        for rec in val_records:
            if rec.path.name not in rupture:
                continue
            hypo = rupture[rec.path.name]
            snaps = load_bandpassed_snapshots(rec, band=band)
            pgv_ref = pgv_from_snapshots(snaps, dt=DEFAULT_DT_SECONDS)

            r_obs = rhypo_km(all_coords_m[obs_idx], hypo)
            r_hid = rhypo_km(all_coords_m[hid_idx], hypo)

            pgv_pred = predict_pgv_uk(
                pgv_obs=pgv_ref[obs_idx],
                rhypo_obs_km=r_obs,
                coords_obs_km=coords_obs_km,
                rhypo_hid_km=r_hid,
                coords_hid_km=coords_hid_km,
                c1=c1,
                c2=c2,
            )
            err = relative_frobenius_error(pgv_ref[hid_idx], pgv_pred)
            pgv_errors.append(err)

        mean_err = float(np.mean(pgv_errors))
        print(f"  seed={seed}: UK-GMPE PGV = {mean_err:.4f}  (n={len(pgv_errors)} events)")
        rows.append({
            "seed": seed,
            "n_observed": n_observed,
            "uk_gmpe_pgv_error": mean_err,
            "c1": c1,
            "c2": c2,
        })

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nUK-GMPE PGV  mean={df['uk_gmpe_pgv_error'].mean():.4f}  "
          f"std={df['uk_gmpe_pgv_error'].std():.4f}")
    print(f"Saved: {output_csv}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DATA_PROCESSED / "events")
    p.add_argument("--raw-root", type=Path, default=DATA_RAW)
    p.add_argument("--band", choices=sorted(FREQUENCY_BANDS), default=DEFAULT_BAND)
    p.add_argument("--n-observed", type=int, default=128)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--max-events", type=int, default=100)
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument(
        "--output-csv",
        type=Path,
        default=RESULTS_TABLES / "uk_gmpe_pgv_random_128.csv",
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
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
