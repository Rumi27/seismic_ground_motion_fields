"""Distance-binned Gappy POD errors for sparse station layouts.

This experiment asks whether a station layout fails globally because it is poor
everywhere, or because it only constrains the field near observed stations.
Distances are computed in km from waveform station indices using the 250 m
surface-grid spacing documented for the GMFlow/SW4 data.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .balanced_pod_analysis import FREQUENCY_BANDS
from .config import DATA_PROCESSED, DATA_RAW, RESULTS_TABLES
from .gappy_pod_reconstruction import (
    DEFAULT_STATION_MAPPING,
    N_COMPONENTS,
    baseline_idw_hidden,
    baseline_mean_field,
    baseline_nearest_station,
    find_reference_h5,
    load_mapped_station_indices,
    load_station_coordinates,
    pgv_from_snapshots,
    pgv_from_station_tensor,
    pod_qr_station_order,
    reconstruct_from_coefficients,
    relative_frobenius_error,
    select_stations_farthest,
    select_stations_ncsn,
    select_stations_pod_qr,
    select_stations_random,
    solve_gappy_coefficients,
    station_dof_indices,
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


DEFAULT_LAYOUTS = ["ncsn", "pod_qr", "random", "farthest"]
DEFAULT_N_OBSERVED = [16, 32, 64, 128]
DEFAULT_RANKS = [151]
DEFAULT_DISTANCE_BINS_KM = [0.0, 10.0, 25.0, 50.0, float("inf")]
GRID_SPACING_KM = 0.25


def bin_label(low: float, high: float) -> str:
    if np.isinf(high):
        return f">{low:g}km"
    return f"{low:g}-{high:g}km"


def nearest_observed_distance_km(
    coords: np.ndarray, observed: np.ndarray, hidden: np.ndarray
) -> np.ndarray:
    obs_coords = coords[observed]
    hidden_coords = coords[hidden]
    distances = np.linalg.norm(
        hidden_coords[:, np.newaxis, :] - obs_coords[np.newaxis, :, :], axis=2
    )
    return np.min(distances, axis=1) * GRID_SPACING_KM


def select_observed_stations(
    layout: str,
    n_observed: int,
    n_stations: int,
    coords: np.ndarray,
    basis: np.ndarray,
    seed: int,
    mapped_station_indices: np.ndarray | None,
) -> np.ndarray:
    if layout == "random":
        return select_stations_random(
            n_stations, n_observed, np.random.default_rng(seed + n_observed)
        )
    if layout == "farthest":
        return select_stations_farthest(
            coords, n_observed, np.random.default_rng(seed + n_observed)
        )
    if layout == "ncsn":
        if mapped_station_indices is None:
            raise RuntimeError("NCSN mapped station indices were not loaded")
        return select_stations_ncsn(
            mapped_station_indices, n_observed, n_stations=n_stations
        )
    if layout == "pod_qr":
        order = pod_qr_station_order(basis, n_stations=n_stations)
        return select_stations_pod_qr(order, n_observed, n_stations=n_stations)
    raise ValueError(f"Unknown layout: {layout}")


def event_distance_bin_metrics(
    snapshots: np.ndarray,
    reconstruction: np.ndarray,
    observed: np.ndarray,
    hidden: np.ndarray,
    train_mean: np.ndarray,
    coords: np.ndarray,
    dt: float,
    distance_bins_km: list[float],
) -> list[dict]:
    n_stations = coords.shape[0]
    n_time = snapshots.shape[1]
    distances_km = nearest_observed_distance_km(coords, observed, hidden)

    mean_recon = baseline_mean_field(train_mean, n_time=n_time)
    nearest_recon = baseline_nearest_station(snapshots, observed, hidden, coords=coords)
    idw_hidden = baseline_idw_hidden(
        snapshots, observed, hidden, coords=coords, power=2.0
    )

    pgv_true = pgv_from_snapshots(snapshots, dt=dt)
    pgv_recon = pgv_from_snapshots(reconstruction, dt=dt)
    pgv_mean = pgv_from_snapshots(mean_recon, dt=dt)
    pgv_nearest = pgv_from_snapshots(nearest_recon, dt=dt)
    idw_pgv = pgv_from_station_tensor(
        idw_hidden.reshape(N_COMPONENTS, hidden.size, n_time), dt=dt
    )
    hidden_position = {int(station): pos for pos, station in enumerate(hidden)}

    rows = []
    for low, high in zip(distance_bins_km[:-1], distance_bins_km[1:]):
        if np.isinf(high):
            in_bin = distances_km >= low
        else:
            in_bin = (distances_km >= low) & (distances_km < high)
        bin_hidden = hidden[in_bin]
        if bin_hidden.size == 0:
            continue

        full_idx = station_dof_indices(bin_hidden, n_stations=n_stations)
        idw_positions = np.asarray(
            [hidden_position[int(station)] for station in bin_hidden], dtype=np.int64
        )
        idw_true = snapshots.reshape(N_COMPONENTS, n_stations, n_time)[
            :, bin_hidden, :
        ].reshape(N_COMPONENTS * bin_hidden.size, n_time)
        idw_pred = idw_hidden.reshape(N_COMPONENTS, hidden.size, n_time)[
            :, idw_positions, :
        ].reshape(N_COMPONENTS * bin_hidden.size, n_time)
        rows.append(
            {
                "distance_bin_km": bin_label(low, high),
                "distance_low_km": low,
                "distance_high_km": high,
                "n_hidden_stations_bin": int(bin_hidden.size),
                "mean_distance_km": float(np.mean(distances_km[in_bin])),
                "waveform_error": relative_frobenius_error(
                    snapshots[full_idx], reconstruction[full_idx]
                ),
                "pgv_error": relative_frobenius_error(
                    pgv_true[bin_hidden], pgv_recon[bin_hidden]
                ),
                "mean_waveform_error": relative_frobenius_error(
                    snapshots[full_idx], mean_recon[full_idx]
                ),
                "nearest_waveform_error": relative_frobenius_error(
                    snapshots[full_idx], nearest_recon[full_idx]
                ),
                "idw_waveform_error": relative_frobenius_error(idw_true, idw_pred),
                "mean_pgv_error": relative_frobenius_error(
                    pgv_true[bin_hidden], pgv_mean[bin_hidden]
                ),
                "nearest_pgv_error": relative_frobenius_error(
                    pgv_true[bin_hidden], pgv_nearest[bin_hidden]
                ),
                "idw_pgv_error": relative_frobenius_error(
                    pgv_true[bin_hidden], idw_pgv[idw_positions]
                ),
            }
        )
    return rows


def aggregate_bin_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    group_keys = (
        "layout",
        "rank",
        "n_observed_stations",
        "distance_bin_km",
        "distance_low_km",
        "distance_high_km",
    )
    metric_keys = (
        "n_hidden_stations_bin",
        "mean_distance_km",
        "waveform_error",
        "pgv_error",
        "mean_waveform_error",
        "nearest_waveform_error",
        "idw_waveform_error",
        "mean_pgv_error",
        "nearest_pgv_error",
        "idw_pgv_error",
    )
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        groups.setdefault(key, []).append(row)

    out = []
    for key, group in groups.items():
        row = {name: value for name, value in zip(group_keys, key)}
        row["n_val_events"] = len(group)
        for metric in metric_keys:
            values = np.asarray([item[metric] for item in group], dtype=np.float64)
            row[metric] = float(np.mean(values))
        out.append(row)
    return sorted(out, key=lambda r: (r["layout"], r["rank"], r["n_observed_stations"], r["distance_low_km"]))


def write_rows(rows: list[dict], output_csv: Path) -> None:
    if not rows:
        raise ValueError("No distance-bin rows were produced")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_csv}")


def run_distance_bin_experiment(
    data_root: Path,
    raw_root: Path,
    event_class: str,
    band_name: str,
    ranks: list[int],
    n_observed_list: list[int],
    layouts: list[str],
    train_fraction: float,
    seed: int,
    max_events: int | None,
    regularization: float,
    station_mapping_csv: Path,
    output_csv: Path,
) -> Path:
    band = FREQUENCY_BANDS[band_name]
    records = discover_processed_events(data_root, event_class=event_class)
    if max_events is not None:
        records = records[:max_events]
    train_records, val_records = split_records(
        records, train_fraction=train_fraction, seed=seed
    )
    coords = load_station_coordinates(find_reference_h5(event_class, raw_root=raw_root))
    n_stations = coords.shape[0]
    mapped_station_indices = (
        load_mapped_station_indices(station_mapping_csv)
        if "ncsn" in set(layouts)
        else None
    )

    train_matrix = assemble_snapshot_matrix(train_records, band=band)
    train_mean = train_matrix.mean(axis=1).astype(np.float32)
    full_basis, _ = compute_pod_basis(train_matrix, rank=max(ranks))
    val_snapshots = [
        (record, load_bandpassed_snapshots(record, band=band)) for record in val_records
    ]

    rows = []
    for rank in ranks:
        basis = full_basis[:, :rank]
        for layout in layouts:
            for n_observed in n_observed_list:
                observed = select_observed_stations(
                    layout=layout,
                    n_observed=n_observed,
                    n_stations=n_stations,
                    coords=coords,
                    basis=basis,
                    seed=seed,
                    mapped_station_indices=mapped_station_indices,
                )
                for record, snapshots in val_snapshots:
                    coefficients, hidden = solve_gappy_coefficients(
                        basis,
                        snapshots,
                        observed,
                        regularization=regularization,
                    )
                    reconstruction = reconstruct_from_coefficients(basis, coefficients)
                    event_rows = event_distance_bin_metrics(
                        snapshots=snapshots,
                        reconstruction=reconstruction,
                        observed=observed,
                        hidden=hidden,
                        train_mean=train_mean,
                        coords=coords,
                        dt=record.dt,
                        distance_bins_km=DEFAULT_DISTANCE_BINS_KM,
                    )
                    for row in event_rows:
                        row.update(
                            {
                                "layout": layout,
                                "rank": rank,
                                "n_observed_stations": n_observed,
                            }
                        )
                    rows.extend(event_rows)

                print(f"layout={layout} rank={rank} n_obs={n_observed} done")

    aggregate = aggregate_bin_rows(rows)
    write_rows(aggregate, output_csv)
    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_PROCESSED / "events")
    parser.add_argument("--raw-root", type=Path, default=DATA_RAW)
    parser.add_argument("--event-class", default=DEFAULT_EVENT_CLASS)
    parser.add_argument("--band", choices=sorted(FREQUENCY_BANDS), default=DEFAULT_BAND)
    parser.add_argument("--ranks", type=int, nargs="+", default=DEFAULT_RANKS)
    parser.add_argument("--n-observed", type=int, nargs="+", default=DEFAULT_N_OBSERVED)
    parser.add_argument("--layouts", nargs="+", default=DEFAULT_LAYOUTS)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-events", type=int, default=100)
    parser.add_argument("--regularization", type=float, default=1e-8)
    parser.add_argument("--station-mapping-csv", type=Path, default=DEFAULT_STATION_MAPPING)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=RESULTS_TABLES / "gappy_pod_mw7_lowfreq_distance_bins.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_distance_bin_experiment(
        data_root=args.data_root,
        raw_root=args.raw_root,
        event_class=args.event_class,
        band_name=args.band,
        ranks=args.ranks,
        n_observed_list=args.n_observed,
        layouts=args.layouts,
        train_fraction=args.train_fraction,
        seed=args.seed,
        max_events=args.max_events,
        regularization=args.regularization,
        station_mapping_csv=args.station_mapping_csv,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
