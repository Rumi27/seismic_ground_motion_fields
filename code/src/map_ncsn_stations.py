"""Map NCEDC/NCSN station coordinates to the simulation coordinate metadata.

Inputs
------
data/raw/stations/ncsn_sfba_fdsn_station_query.txt
    Text response from the NCEDC FDSN station service.

data/raw/meta/point_coordinates.h5
    Simulation coordinate metadata with fields i, j, lat, lon, x, y.

Outputs
-------
data/raw/stations/ncsn_sfba_stations.csv
data/processed/ncsn_station_mapping.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .config import DATA_PROCESSED, DATA_RAW, NY


DEFAULT_RAW_STATIONS = DATA_RAW / "stations" / "ncsn_sfba_fdsn_station_query.txt"
DEFAULT_STATIONS_CSV = DATA_RAW / "stations" / "ncsn_sfba_stations.csv"
DEFAULT_POINT_COORDS = DATA_RAW / "meta" / "point_coordinates.h5"
DEFAULT_MAPPING_CSV = DATA_PROCESSED / "ncsn_station_mapping.csv"
EARTH_RADIUS_KM = 6371.0088


def haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Great-circle distance for vectorized station-to-grid comparisons."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(
        dlambda / 2.0
    ) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def read_fdsn_station_text(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) < 8:
                continue
            network, station, lat, lon, elevation, site_name, start_time, end_time = (
                parts[:8]
            )
            rows.append(
                {
                    "network": network,
                    "station": station,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "elevation_m": float(elevation),
                    "site_name": site_name,
                    "start_time": start_time,
                    "end_time": end_time,
                }
            )
    if not rows:
        raise ValueError(f"No station rows found in {path}")

    df = pd.DataFrame(rows)
    df = df.sort_values(["station", "start_time", "end_time"]).drop_duplicates(
        "station", keep="last"
    )
    return df.sort_values("station").reset_index(drop=True)


def load_point_coordinates(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        coords = handle["coordinates"][()]
    required = {"i", "j", "lat", "lon"}
    missing = required.difference(coords.dtype.names or ())
    if missing:
        raise ValueError(f"{path} is missing coordinate fields: {sorted(missing)}")
    return coords


def nearest_waveform_station_index(grid_i: int, grid_j: int) -> int:
    """Map zero-based metadata grid indices to sorted S_XX_YY waveform rows."""
    return int(grid_i * NY + grid_j)


def build_mapping(stations: pd.DataFrame, point_coords: np.ndarray) -> pd.DataFrame:
    grid_lat = point_coords["lat"].astype(float)
    grid_lon = point_coords["lon"].astype(float)
    records: list[dict[str, object]] = []
    used_grid_indices: set[int] = set()

    for station in stations.itertuples(index=False):
        distances = haversine_km(
            np.asarray(float(station.latitude)),
            np.asarray(float(station.longitude)),
            grid_lat,
            grid_lon,
        )
        nearest = None
        for candidate in np.argsort(distances):
            candidate = int(candidate)
            if candidate not in used_grid_indices:
                nearest = candidate
                used_grid_indices.add(candidate)
                break
        if nearest is None:
            raise ValueError(
                f"Only {len(point_coords)} simulation points are available for "
                f"{len(stations)} stations"
            )
        grid_row = point_coords[nearest]
        records.append(
            {
                "station": station.station,
                "lat": float(station.latitude),
                "lon": float(station.longitude),
                "nearest_grid_index": nearest,
                "nearest_station_index": nearest_waveform_station_index(
                    int(grid_row["i"]), int(grid_row["j"])
                ),
                "grid_i": int(grid_row["i"]),
                "grid_j": int(grid_row["j"]),
                "grid_lat": float(grid_row["lat"]),
                "grid_lon": float(grid_row["lon"]),
                "distance_km": float(distances[nearest]),
            }
        )

    mapping = pd.DataFrame(records)
    mapping = mapping.sort_values(["station", "distance_km"]).drop_duplicates(
        "station", keep="first"
    )
    return mapping.reset_index(drop=True)


def write_minimum_station_csv(stations: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["station", "latitude", "longitude"]
    stations.loc[:, cols].to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-stations", type=Path, default=DEFAULT_RAW_STATIONS)
    parser.add_argument("--stations-csv", type=Path, default=DEFAULT_STATIONS_CSV)
    parser.add_argument("--point-coords", type=Path, default=DEFAULT_POINT_COORDS)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_MAPPING_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stations = read_fdsn_station_text(args.raw_stations)
    point_coords = load_point_coordinates(args.point_coords)
    mapping = build_mapping(stations, point_coords)

    write_minimum_station_csv(stations, args.stations_csv)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.stations_csv} ({len(stations)} stations)")
    print(
        f"Wrote {args.output_csv} "
        f"({mapping['nearest_station_index'].nunique()} unique mapped waveform stations)"
    )


if __name__ == "__main__":
    main()
