"""Train a deterministic linear OpInf ROM for low-frequency Mw7 wavefields.

Uses preprocessed event arrays from ``data/processed/events/``, the same
bandpass and snapshot utilities as ``balanced_pod_analysis``, and fits a
discrete-time reduced propagator ``q_{k+1} = A_hat q_k`` via regularized least
squares on POD coefficients.

Example
-------
    python -m src.train_lowfreq_opinf
    python -m src.train_lowfreq_opinf --rank 151 --max-events 100
    python -m src.train_lowfreq_opinf --rank-from-table --energy-level 0.999
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .balanced_pod_analysis import (
    DEFAULT_DT_SECONDS,
    FREQUENCY_BANDS,
    bandpass_fft,
)
from .config import DATA_PROCESSED, OUTPUTS, RESULTS_TABLES
from .pod_feasibility import event_to_snapshots, load_event_array


DEFAULT_EVENT_CLASS = "mw7"
DEFAULT_BAND = "low_0_025hz"
DEFAULT_RANK = 151  # mw7 low-band r_99 at 100 events/class


@dataclass(frozen=True)
class EventPath:
    path: Path
    dt: float


def discover_processed_events(data_root: Path, event_class: str) -> list[EventPath]:
    event_dir = data_root / event_class
    if not event_dir.is_dir():
        raise FileNotFoundError(f"No processed events found at {event_dir}")
    records = []
    for path in sorted(event_dir.glob("*.npy")):
        records.append(EventPath(path=path, dt=DEFAULT_DT_SECONDS))
    return records


def load_bandpassed_snapshots(
    record: EventPath, band: tuple[float, float]
) -> np.ndarray:
    event = load_event_array(record.path)
    filtered = bandpass_fft(event, dt=record.dt, low_hz=band[0], high_hz=band[1])
    return event_to_snapshots(filtered)


def assemble_snapshot_matrix(records: list[EventPath], band: tuple[float, float]) -> np.ndarray:
    blocks = [load_bandpassed_snapshots(record, band=band) for record in records]
    return np.concatenate(blocks, axis=1)


def compute_pod_basis(matrix: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Return spatial POD basis V (n_dof, rank) and singular values."""
    gram = np.asarray(matrix.T @ matrix, dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    positive = eigvals > 0
    eigvals = eigvals[positive]
    eigvecs = eigvecs[:, positive]
    rank = min(rank, eigvals.size)
    eigvals = eigvals[:rank]
    eigvecs = eigvecs[:, :rank]
    singular_values = np.sqrt(eigvals)
    basis = (matrix @ eigvecs) / singular_values
    return basis.astype(np.float32), singular_values.astype(np.float64)


def infer_discrete_operator(states: np.ndarray, regularization: float) -> np.ndarray:
    """Fit one-step propagator q_{k+1} = A_d q_k from snapshot pairs."""
    current = states[:, :-1]
    nxt = states[:, 1:]
    gram = current @ current.T
    rhs = nxt @ current.T
    reg = regularization * np.eye(gram.shape[0], dtype=np.float64)
    operator = np.linalg.solve(gram + reg, rhs)
    return operator.astype(np.float32)


def integrate_discrete_operator(
    operator: np.ndarray, initial_state: np.ndarray, n_steps: int
) -> np.ndarray:
    trajectory = np.empty((operator.shape[0], n_steps), dtype=np.float32)
    trajectory[:, 0] = initial_state
    for step in range(1, n_steps):
        trajectory[:, step] = operator @ trajectory[:, step - 1]
    return trajectory


def relative_frobenius_error(reference: np.ndarray, approximation: np.ndarray) -> float:
    num = np.linalg.norm(reference - approximation)
    den = np.linalg.norm(reference)
    return float(num / den) if den > 0 else float("nan")


def rank_from_summary_table(
    event_class: str,
    band_name: str,
    energy_level: float,
    table_path: Path,
) -> int:
    df = pd.read_csv(table_path)
    row = df[
        (df["class"] == event_class) & (df["frequency_band"] == band_name)
    ].iloc[0]
    if energy_level >= 0.9999:
        return int(row["r_9999"])
    if energy_level >= 0.999:
        return int(row["r_999"])
    return int(row["r_99"])


def split_records(
    records: list[EventPath], train_fraction: float, seed: int
) -> tuple[list[EventPath], list[EventPath]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)
    n_train = max(1, int(round(train_fraction * len(records))))
    train_idx = sorted(indices[:n_train])
    val_idx = sorted(indices[n_train:])
    if not val_idx:
        val_idx = train_idx[-1:]
        train_idx = train_idx[:-1]
    train = [records[i] for i in train_idx]
    val = [records[i] for i in val_idx]
    return train, val


def evaluate_event(
    record: EventPath,
    band: tuple[float, float],
    basis: np.ndarray,
    operator: np.ndarray,
) -> dict:
    snapshots = load_bandpassed_snapshots(record, band=band)
    reduced = basis.T @ snapshots
    one_step_pred = operator @ reduced[:, :-1]
    trajectory = integrate_discrete_operator(
        operator, reduced[:, 0], n_steps=reduced.shape[1]
    )
    reconstructed = basis @ trajectory
    return {
        "event": record.path.name,
        "n_time": snapshots.shape[1],
        "relative_state_error": relative_frobenius_error(snapshots, reconstructed),
        "relative_one_step_error": relative_frobenius_error(
            reduced[:, 1:], one_step_pred
        ),
        "relative_reduced_trajectory_error": relative_frobenius_error(
            reduced, trajectory
        ),
    }


def run_training(
    data_root: Path,
    output_dir: Path,
    event_class: str,
    band_name: str,
    rank: int,
    regularization: float,
    train_fraction: float,
    seed: int,
    max_events: int | None,
) -> Path:
    band = FREQUENCY_BANDS[band_name]
    records = discover_processed_events(data_root, event_class=event_class)
    if max_events is not None:
        records = records[:max_events]
    if len(records) < 3:
        raise ValueError(f"Need at least 3 events, found {len(records)}")

    train_records, val_records = split_records(
        records, train_fraction=train_fraction, seed=seed
    )
    print(f"Training on {len(train_records)} events, validating on {len(val_records)}")

    train_matrix = assemble_snapshot_matrix(train_records, band=band)
    basis, singular_values = compute_pod_basis(train_matrix, rank=rank)
    reduced_train = basis.T @ train_matrix
    operator = infer_discrete_operator(reduced_train, regularization=regularization)

    train_fit = {
        "relative_one_step_error": relative_frobenius_error(
            reduced_train[:, 1:], operator @ reduced_train[:, :-1]
        ),
        "relative_state_error": relative_frobenius_error(
            train_matrix, basis @ reduced_train
        ),
    }

    val_rows = [
        evaluate_event(record, band=band, basis=basis, operator=operator)
        for record in val_records
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "A_hat.npy", operator)
    np.save(output_dir / "V.npy", basis)
    np.save(output_dir / "singular_values.npy", singular_values)

    metadata = {
        "event_class": event_class,
        "frequency_band": band_name,
        "band_hz": band,
        "rank": rank,
        "regularization": regularization,
        "n_train_events": len(train_records),
        "n_val_events": len(val_records),
        "n_spatial_dof": int(train_matrix.shape[0]),
        "n_train_snapshots": int(train_matrix.shape[1]),
        "dt_seconds": train_records[0].dt,
        "train_fit": train_fit,
        "val_mean_relative_state_error": float(
            np.mean([row["relative_state_error"] for row in val_rows])
        ),
        "val_mean_relative_one_step_error": float(
            np.mean([row["relative_one_step_error"] for row in val_rows])
        ),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    metrics_path = output_dir / "validation_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(val_rows[0].keys()))
        writer.writeheader()
        writer.writerows(val_rows)

    print(f"Wrote OpInf model to {output_dir}")
    print(json.dumps(metadata, indent=2))
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_PROCESSED / "events",
        help="Processed event root (class subfolders)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS / "lowfreq_opinf_mw7",
    )
    parser.add_argument("--event-class", default=DEFAULT_EVENT_CLASS)
    parser.add_argument(
        "--band",
        choices=sorted(FREQUENCY_BANDS),
        default=DEFAULT_BAND,
    )
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument(
        "--rank-from-table",
        action="store_true",
        help="Set rank from balanced_pod_rank_summary CSV",
    )
    parser.add_argument(
        "--summary-table",
        type=Path,
        default=RESULTS_TABLES / "balanced_pod_rank_summary_100class.csv",
    )
    parser.add_argument(
        "--energy-level",
        type=float,
        default=0.99,
        help="Used with --rank-from-table (0.99, 0.999, 0.9999)",
    )
    parser.add_argument("--regularization", type=float, default=1e-8)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-events", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank = args.rank
    if args.rank_from_table:
        rank = rank_from_summary_table(
            event_class=args.event_class,
            band_name=args.band,
            energy_level=args.energy_level,
            table_path=args.summary_table,
        )
        print(f"Using rank {rank} from {args.summary_table}")

    run_training(
        data_root=args.data_root,
        output_dir=args.output_dir,
        event_class=args.event_class,
        band_name=args.band,
        rank=rank,
        regularization=args.regularization,
        train_fraction=args.train_fraction,
        seed=args.seed,
        max_events=args.max_events,
    )


if __name__ == "__main__":
    main()
