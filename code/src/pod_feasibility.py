"""Phase 1 POD feasibility utilities.

This script is intentionally conservative: it first inspects local files and only
attempts a small in-memory SVD when compatible array files are present. The full
GMFlow dataset is too large for naive loading, so this is meant for subset tests
before implementing the out-of-core pipeline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from .config import DATA_PROCESSED, DATA_RAW, DEFAULT_TARGET_ENERGIES, RESULTS_TABLES


SUPPORTED_SUFFIXES = {".npy", ".npz", ".h5", ".hdf5"}
STATION_GROUP_RE = re.compile(r"^S_(\d+)_(\d+)$")


@dataclass(frozen=True)
class DataFile:
    path: Path
    shape: tuple[int, ...] | None
    dtype: str | None


def discover_data_files(data_dir: Path = DATA_RAW) -> list[Path]:
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        and "meta" not in path.relative_to(data_dir).parts
    )


def inspect_array_file(path: Path) -> DataFile:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
        return DataFile(path=path, shape=arr.shape, dtype=str(arr.dtype))

    if suffix == ".npz":
        with np.load(path) as archive:
            first_key = archive.files[0] if archive.files else None
            if first_key is None:
                return DataFile(path=path, shape=None, dtype=None)
            arr = archive[first_key]
            return DataFile(path=path, shape=arr.shape, dtype=str(arr.dtype))

    if suffix in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "r") as handle:
            station_groups = _station_groups(handle)
            if station_groups:
                first = handle[station_groups[0]]
                npts = int(first["NPTS"][0]) if "NPTS" in first else len(first["X"])
                return DataFile(
                    path=path,
                    shape=(3, len(station_groups), npts),
                    dtype=str(first["X"].dtype),
                )
            dataset = _first_hdf5_dataset(handle)
            if dataset is None:
                return DataFile(path=path, shape=None, dtype=None)
            return DataFile(path=path, shape=dataset.shape, dtype=str(dataset.dtype))

    return DataFile(path=path, shape=None, dtype=None)


def _first_hdf5_dataset(group):
    import h5py

    for value in group.values():
        if isinstance(value, h5py.Dataset):
            return value
        if isinstance(value, h5py.Group):
            found = _first_hdf5_dataset(value)
            if found is not None:
                return found
    return None


def _station_groups(handle) -> list[str]:
    groups = [key for key in handle.keys() if STATION_GROUP_RE.match(key)]
    return sorted(groups, key=_station_group_sort_key)


def _station_group_sort_key(name: str) -> tuple[int, int]:
    match = STATION_GROUP_RE.match(name)
    if match is None:
        return (0, 0)
    return (int(match.group(1)), int(match.group(2)))


def load_event_array(path: Path) -> np.ndarray:
    """Load one event and return shape (3, 256, 128, 96) or compatible."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float32)

    if suffix == ".npz":
        with np.load(path) as archive:
            if not archive.files:
                raise ValueError(f"No arrays found in {path}")
            return np.asarray(archive[archive.files[0]], dtype=np.float32)

    if suffix in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "r") as handle:
            station_groups = _station_groups(handle)
            if station_groups:
                return load_station_event(handle, station_groups)
            dataset = _first_hdf5_dataset(handle)
            if dataset is None:
                raise ValueError(f"No datasets found in {path}")
            return np.asarray(dataset[...], dtype=np.float32)

    raise ValueError(f"Unsupported file type: {path}")


def load_station_event(handle, station_groups: list[str]) -> np.ndarray:
    first = handle[station_groups[0]]
    npts = int(first["NPTS"][0]) if "NPTS" in first else len(first["X"])
    event = np.empty((3, len(station_groups), npts), dtype=np.float32)
    for idx, group_name in enumerate(station_groups):
        group = handle[group_name]
        event[0, idx, :] = group["X"][...]
        event[1, idx, :] = group["Y"][...]
        event[2, idx, :] = group["Z"][...]
    return event


def event_to_snapshots(event: np.ndarray) -> np.ndarray:
    """Convert one event into an (spatial_dof, n_time) snapshot block."""
    arr = np.asarray(event, dtype=np.float32)
    if arr.shape == (3, 256, 128, 96):
        arr = arr.reshape(3, 256 * 128, 96)
    elif arr.shape == (96, 3, 256, 128):
        arr = np.moveaxis(arr, 0, -1)
        arr = arr.reshape(3, 256 * 128, 96)
    elif arr.shape == (256, 128, 96, 3):
        arr = np.moveaxis(arr, -1, 0)
        arr = arr.reshape(3, 256 * 128, 96)
    elif arr.ndim == 3 and arr.shape[0] == 3:
        pass
    else:
        raise ValueError(
            f"Unsupported event shape {arr.shape}; expected compatible GMFlow shape"
        )

    scale = float(np.std(arr))
    if scale == 0.0:
        raise ValueError("Event has zero standard deviation")
    arr = arr / scale
    return arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])


def assemble_subset_matrix(paths: list[Path], max_events: int) -> np.ndarray:
    selected = paths[:max_events]
    if not selected:
        raise ValueError(f"No data files found in {DATA_RAW}")

    blocks = [event_to_snapshots(load_event_array(path)) for path in selected]
    return np.concatenate(blocks, axis=1)


def cumulative_energy(singular_values: np.ndarray) -> np.ndarray:
    squared = singular_values**2
    total = float(np.sum(squared))
    if total == 0.0:
        raise ValueError("Singular values have zero energy")
    return np.cumsum(squared) / total


def rank_for_energy(energy: np.ndarray, target: float) -> int:
    return int(np.searchsorted(energy, target, side="left") + 1)


def run_subset_pod(max_events: int, rank: int) -> Path:
    paths = discover_data_files()
    matrix = assemble_subset_matrix(paths, max_events=max_events)
    singular_values = gram_singular_values(matrix, rank=rank)
    energy = cumulative_energy(singular_values)

    rows = [
        {"target_energy": target, "rank": rank_for_energy(energy, target)}
        for target in DEFAULT_TARGET_ENERGIES
        if target <= float(energy[-1])
    ]

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    np.save(DATA_PROCESSED / "subset_singular_values.npy", singular_values)
    np.save(DATA_PROCESSED / "subset_cumulative_energy.npy", energy)

    out_path = RESULTS_TABLES / "subset_pod_rank_summary.csv"
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("target_energy,rank\n")
        for row in rows:
            handle.write(f"{row['target_energy']},{row['rank']}\n")
    return out_path


def gram_singular_values(matrix: np.ndarray, rank: int | None = None) -> np.ndarray:
    """Compute leading singular values through the small temporal Gram matrix."""
    gram = np.asarray(matrix.T @ matrix, dtype=np.float64)
    eigvals = np.linalg.eigvalsh(gram)
    eigvals = np.maximum(eigvals, 0.0)
    singular_values = np.sqrt(eigvals)[::-1]
    n_components = singular_values.size if rank is None else min(rank, singular_values.size)
    return singular_values[:n_components]


def print_inspection() -> None:
    paths = discover_data_files()
    if not paths:
        print(f"No supported data files found in {DATA_RAW}")
        return

    for path in paths:
        info = inspect_array_file(path)
        rel = info.path.relative_to(DATA_RAW)
        print(f"{rel}: shape={info.shape}, dtype={info.dtype}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true", help="Inspect raw data files")
    parser.add_argument("--max-events", type=int, default=30)
    parser.add_argument("--rank", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.inspect:
        print_inspection()
        return

    out_path = run_subset_pod(max_events=args.max_events, rank=args.rank)
    print(f"Wrote POD rank summary to {out_path}")


if __name__ == "__main__":
    main()
