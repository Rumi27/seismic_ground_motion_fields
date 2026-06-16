"""Preprocess GMFlow HDF5 station-group files into contiguous NumPy arrays.

Raw GMFlow HDF5 files contain tens of thousands of small station groups, which is
slow for repeated POD experiments. This script converts each event once into a
single ``.npy`` array with shape ``(3, n_stations, n_time)``.

Example
-------
    python -m src.preprocess_events --input-root data/raw --output-root data/processed/events
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .balanced_pod_analysis import DEFAULT_DT_SECONDS, classify_path
from .config import DATA_PROCESSED, DATA_RAW
from .pod_feasibility import inspect_array_file, load_event_array


def discover_raw_hdf5(input_root: Path) -> list[Path]:
    return sorted(
        path
        for path in input_root.rglob("*.h5")
        if path.is_file() and "meta" not in path.relative_to(input_root).parts
    )


def class_for_path(path: Path, input_root: Path) -> str | None:
    return classify_path(set(path.relative_to(input_root).parts))


def load_dt_or_default(path: Path) -> float:
    import h5py

    with h5py.File(path, "r") as handle:
        if "DELTA" in handle:
            return float(handle["DELTA"][0])
    return DEFAULT_DT_SECONDS


def preprocess_file(path: Path, input_root: Path, output_root: Path) -> dict | None:
    event_class = class_for_path(path, input_root)
    if event_class is None:
        print(f"skip unclassified {path}")
        return None

    output_dir = output_root / event_class
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{path.stem}.npy"

    if output_path.exists():
        info = inspect_array_file(output_path)
        print(f"skip existing {output_path} shape={info.shape}")
    else:
        print(f"preprocess {path} -> {output_path}")
        event = load_event_array(path)
        np.save(output_path, event.astype(np.float32, copy=False))

    info = inspect_array_file(output_path)
    return {
        "class": event_class,
        "raw_path": str(path),
        "processed_path": str(output_path),
        "shape": "x".join(str(value) for value in (info.shape or ())),
        "dtype": info.dtype,
        "dt_seconds": load_dt_or_default(path),
    }


def run(input_root: Path, output_root: Path) -> Path:
    rows = []
    for path in discover_raw_hdf5(input_root):
        row = preprocess_file(path, input_root=input_root, output_root=output_root)
        if row is not None:
            rows.append(row)

    manifest_path = output_root / "processed_event_manifest.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "class",
                "raw_path",
                "processed_path",
                "shape",
                "dtype",
                "dt_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DATA_RAW)
    parser.add_argument("--output-root", type=Path, default=DATA_PROCESSED / "events")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run(input_root=args.input_root, output_root=args.output_root)
    print(f"Wrote processed manifest to {manifest}")


if __name__ == "__main__":
    main()

