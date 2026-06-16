"""Balanced POD rank analysis across magnitude classes and frequency bands.

This module is the next-stage analysis driver. It expects local event files to be
organized under ``data/raw`` with class-specific folders or any paths listed in a
manifest CSV. The current implementation can run on locally downloaded HDF5 files
using the station-group layout discovered in GMFlow samples.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DATA_RAW, RESULTS_TABLES
from .pod_feasibility import (
    cumulative_energy,
    event_to_snapshots,
    gram_singular_values,
    load_event_array,
    rank_for_energy,
)


FREQUENCY_BANDS = {
    "full_0_1hz": (0.0, 1.0),
    "low_0_025hz": (0.0, 0.25),
    "mid_025_05hz": (0.25, 0.5),
    "high_05_1hz": (0.5, 1.0),
}

DEFAULT_DT_SECONDS = 0.2534854


CLASS_PATTERNS = {
    "point_source": ["point_source", "sample_events", "Batch_1Hz_100_ValidationSet_flat"],
    "mw6": ["mw6", "final_results_M6_Testing"],
    "mw7": ["mw7", "final_results_M7_Testing"],
}


@dataclass(frozen=True)
class EventRecord:
    event_class: str
    path: Path


@dataclass(frozen=True)
class LoadedEvent:
    event: np.ndarray
    dt: float


def discover_balanced_events(data_root: Path = DATA_RAW) -> list[EventRecord]:
    records: list[EventRecord] = []
    candidate_paths = sorted(
        path
        for suffix in ("*.h5", "*.hdf5", "*.npy")
        for path in data_root.rglob(suffix)
    )
    for path in candidate_paths:
        if "meta" in path.relative_to(data_root).parts:
            continue
        rel_parts = set(path.relative_to(data_root).parts)
        event_class = classify_path(rel_parts)
        if event_class is not None:
            records.append(EventRecord(event_class=event_class, path=path))
    return records


def classify_path(rel_parts: set[str]) -> str | None:
    for event_class, patterns in CLASS_PATTERNS.items():
        if any(pattern in rel_parts for pattern in patterns):
            return event_class
    return None


def limit_per_class(records: list[EventRecord], max_per_class: int) -> list[EventRecord]:
    selected: list[EventRecord] = []
    counts: dict[str, int] = {}
    for record in records:
        count = counts.get(record.event_class, 0)
        if count >= max_per_class:
            continue
        selected.append(record)
        counts[record.event_class] = count + 1
    return selected


def bandpass_fft(event: np.ndarray, dt: float, low_hz: float, high_hz: float) -> np.ndarray:
    """Apply an FFT-domain bandpass along the last axis."""
    arr = np.asarray(event, dtype=np.float32)
    freqs = np.fft.rfftfreq(arr.shape[-1], d=dt)
    spectrum = np.fft.rfft(arr, axis=-1)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    spectrum *= mask.reshape((1,) * (spectrum.ndim - 1) + (mask.size,))
    filtered = np.fft.irfft(spectrum, n=arr.shape[-1], axis=-1)
    return filtered.astype(np.float32, copy=False)


def load_dt(path: Path) -> float:
    if path.suffix.lower() == ".npy":
        return DEFAULT_DT_SECONDS

    import h5py

    with h5py.File(path, "r") as handle:
        if "DELTA" in handle:
            return float(handle["DELTA"][0])
        # Finite-rupture M6/M7 files omit root metadata but use the same 118-point
        # 1 Hz downsampled traces as the point-source validation files.
        return DEFAULT_DT_SECONDS


def assemble_matrix(records: list[EventRecord], band: tuple[float, float]) -> np.ndarray:
    loaded = load_events(records)
    return assemble_matrix_from_loaded(loaded, band=band)


def load_events(records: list[EventRecord]) -> list[LoadedEvent]:
    return [
        LoadedEvent(event=load_event_array(record.path), dt=load_dt(record.path))
        for record in records
    ]


def assemble_matrix_from_loaded(
    loaded_events: list[LoadedEvent], band: tuple[float, float]
) -> np.ndarray:
    blocks = []
    for loaded in loaded_events:
        filtered = bandpass_fft(
            loaded.event, dt=loaded.dt, low_hz=band[0], high_hz=band[1]
        )
        blocks.append(event_to_snapshots(filtered))
    return np.concatenate(blocks, axis=1)


def decay_exponent(singular_values: np.ndarray, start: int = 5) -> float:
    if singular_values.size <= start + 2:
        return float("nan")
    end = singular_values.size
    idx = np.arange(start, end + 1, dtype=np.float64)
    values = singular_values[start - 1 : end]
    positive = values > 0
    idx = idx[positive]
    values = values[positive]
    if values.size < 3:
        return float("nan")
    slope, _ = np.polyfit(np.log(idx), np.log(values), deg=1)
    return float(-slope)


def run_analysis(data_root: Path, max_per_class: int, bands: list[str]) -> Path:
    records = limit_per_class(
        discover_balanced_events(data_root=data_root), max_per_class=max_per_class
    )
    by_class: dict[str, list[EventRecord]] = {}
    for record in records:
        by_class.setdefault(record.event_class, []).append(record)

    rows = []
    for event_class, class_records in sorted(by_class.items()):
        loaded_events = load_events(class_records)
        for band_name in bands:
            band = FREQUENCY_BANDS[band_name]
            matrix = assemble_matrix_from_loaded(loaded_events, band=band)
            singular_values = gram_singular_values(matrix, rank=None)
            energy = cumulative_energy(singular_values)
            rows.append(
                {
                    "class": event_class,
                    "frequency_band": band_name,
                    "n_events": len(class_records),
                    "n_spatial_dof": matrix.shape[0],
                    "n_snapshots": matrix.shape[1],
                    "rank_max": len(singular_values),
                    "r_99": rank_for_energy(energy, 0.99),
                    "r_999": rank_for_energy(energy, 0.999),
                    "r_9999": rank_for_energy(energy, 0.9999),
                    "alpha": decay_exponent(singular_values),
                    "energy_at_rank_cap": float(energy[-1]),
                }
            )

    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_TABLES / "balanced_pod_rank_summary.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def print_inventory(records: list[EventRecord]) -> None:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.event_class] = counts.get(record.event_class, 0) + 1
    if not counts:
        print("No balanced-analysis event files found under data/raw")
        return
    for event_class, count in sorted(counts.items()):
        print(f"{event_class}: {count} local files")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_RAW)
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--max-per-class", type=int, default=5)
    parser.add_argument(
        "--bands",
        nargs="+",
        choices=sorted(FREQUENCY_BANDS),
        default=sorted(FREQUENCY_BANDS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = discover_balanced_events(data_root=args.data_root)
    if args.inventory:
        print_inventory(records)
        return
    out_path = run_analysis(
        data_root=args.data_root, max_per_class=args.max_per_class, bands=args.bands
    )
    print(f"Wrote balanced POD rank summary to {out_path}")


if __name__ == "__main__":
    main()
