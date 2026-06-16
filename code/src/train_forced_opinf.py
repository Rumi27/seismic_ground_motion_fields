"""Source-forced linear OpInf ROM for low-frequency Mw7 wavefields.

Extends the autonomous model q_{k+1} = A q_k to the source-forced form
    q_{k+1} = A q_k + B u^(e)
where u^(e) in R^p is a static event-specific source parameter vector
(e.g., [HYP_y, Vel] for Mw7).  The augmented design matrix stacks the
repeated source vector below the reduced-state columns so that A and B are
recovered simultaneously from the same regularised least-squares solve.

Example
-------
    python -m src.train_forced_opinf
    python -m src.train_forced_opinf --rank 151 --max-events 100
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .balanced_pod_analysis import (
    DEFAULT_DT_SECONDS,
    FREQUENCY_BANDS,
    bandpass_fft,
)
from .config import DATA_PROCESSED, OUTPUTS, RESULTS_TABLES
from .pod_feasibility import event_to_snapshots, load_event_array
from .train_lowfreq_opinf import (
    EventPath,
    compute_pod_basis,
    discover_processed_events,
    load_bandpassed_snapshots,
    operator_stability,
    relative_frobenius_error,
    split_records,
)


# ---------------------------------------------------------------------------
# Source parameter extraction (Mw7 filename convention)
# ---------------------------------------------------------------------------

_MW7_PATTERN = re.compile(
    r"HYP_[\d.]+_([-\d.]+)_Vel_([\d.]+)"
)


def parse_source_params_mw7(path: Path) -> np.ndarray:
    """Extract [HYP_y, Vel] from an Mw7 event filename.

    HYP_y: hypocenter y-offset in km, range approximately [-25, +12].
    Vel:   rupture-velocity ratio, range approximately [0.67, 0.94].
    """
    m = _MW7_PATTERN.search(path.name)
    if m is None:
        raise ValueError(f"Cannot parse Mw7 source params from: {path.name}")
    return np.array([float(m.group(1)), float(m.group(2))], dtype=np.float64)


# ---------------------------------------------------------------------------
# Source normalisation
# ---------------------------------------------------------------------------

@dataclass
class SourceScaler:
    """Z-score normaliser fit on training events."""
    mean: np.ndarray  # (p,)
    std: np.ndarray   # (p,)

    def transform(self, u: np.ndarray) -> np.ndarray:
        return (u - self.mean) / self.std

    @classmethod
    def fit(cls, source_vecs: np.ndarray) -> "SourceScaler":
        """source_vecs: (n_events, p)."""
        return cls(mean=source_vecs.mean(axis=0), std=source_vecs.std(axis=0) + 1e-12)


# ---------------------------------------------------------------------------
# Forced operator inference
# ---------------------------------------------------------------------------

def infer_forced_operator(
    states: np.ndarray,       # (r, N_events * T)  concatenated training states
    source_tile: np.ndarray,  # (p, N_events * T)  source params tiled per time step
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit q_{k+1} = A q_k + B u via augmented least squares.

    Augments each column [q_k] with [u^(e)] so the combined (r+p) x (r+p)
    Gram matrix encodes source co-variance. Returns (A_hat, B_hat).
    """
    aug = np.vstack([states[:, :-1], source_tile[:, :-1]])  # (r+p, N*(T-1))
    nxt = np.asarray(states[:, 1:], dtype=np.float64)        # (r,   N*(T-1))
    gram = np.asarray(aug @ aug.T, dtype=np.float64)          # (r+p, r+p)
    rhs  = nxt @ aug.T                                         # (r,   r+p)
    reg  = regularization * np.eye(gram.shape[0], dtype=np.float64)
    # Solve M @ (gram+reg) = rhs  <=>  (gram+reg) M^T = rhs^T
    M_T = np.linalg.solve(gram + reg, rhs.T)  # (r+p, r)
    M = M_T.T                                   # (r,   r+p)
    r = states.shape[0]
    return M[:, :r].astype(np.float32), M[:, r:].astype(np.float32)


def integrate_forced_operator(
    A: np.ndarray,
    B: np.ndarray,
    source_u: np.ndarray,  # (p,) normalised, constant across all time steps
    initial_state: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    Bu = (B @ source_u).astype(np.float32)  # (r,) precomputed constant
    trajectory = np.empty((A.shape[0], n_steps), dtype=np.float32)
    trajectory[:, 0] = initial_state
    for step in range(1, n_steps):
        trajectory[:, step] = A @ trajectory[:, step - 1] + Bu
    return trajectory


# ---------------------------------------------------------------------------
# Assembly helpers
# ---------------------------------------------------------------------------

def build_source_tile(
    records: list[EventPath],
    scaler: SourceScaler,
    n_time: int,
) -> np.ndarray:
    """Build (p, N_events * n_time) tiled normalised source matrix."""
    cols = []
    for record in records:
        u = scaler.transform(parse_source_params_mw7(record.path))
        cols.append(np.tile(u[:, None], (1, n_time)))  # (p, T)
    return np.concatenate(cols, axis=1).astype(np.float64)


def assemble_snapshot_matrix(
    records: list[EventPath], band: tuple[float, float]
) -> np.ndarray:
    blocks = [load_bandpassed_snapshots(record, band=band) for record in records]
    return np.concatenate(blocks, axis=1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_forced_event(
    record: EventPath,
    band: tuple[float, float],
    basis: np.ndarray,
    A_hat: np.ndarray,
    B_hat: np.ndarray,
    scaler: SourceScaler,
) -> dict:
    snapshots = load_bandpassed_snapshots(record, band=band)
    reduced = basis.T @ snapshots            # (r, T)
    source_u = scaler.transform(parse_source_params_mw7(record.path))

    # Forced rollout from true first snapshot
    trajectory = integrate_forced_operator(
        A_hat, B_hat, source_u, reduced[:, 0], n_steps=reduced.shape[1]
    )
    reconstructed = basis @ trajectory

    # One-step prediction (forced)
    Bu = (B_hat @ source_u).astype(np.float32)
    one_step = A_hat @ reduced[:, :-1] + Bu[:, None]

    return {
        "event": record.path.name,
        "n_time": snapshots.shape[1],
        "relative_state_error": relative_frobenius_error(snapshots, reconstructed),
        "relative_one_step_error": relative_frobenius_error(reduced[:, 1:], one_step),
        "relative_reduced_trajectory_error": relative_frobenius_error(reduced, trajectory),
    }


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def run_forced_training(
    data_root: Path,
    output_dir: Path,
    event_class: str,
    band_name: str,
    rank: int,
    regularization: float,
    train_fraction: float,
    seed: int,
    max_events: int | None,
    save_artifacts: bool = True,
) -> dict:
    band = FREQUENCY_BANDS[band_name]
    records = discover_processed_events(data_root, event_class=event_class)
    if max_events is not None:
        records = records[:max_events]
    if len(records) < 3:
        raise ValueError(f"Need at least 3 events, found {len(records)}")

    train_records, val_records = split_records(records, train_fraction, seed)
    print(f"Training on {len(train_records)} events, validating on {len(val_records)}")

    # Build POD basis from training snapshots
    train_matrix = assemble_snapshot_matrix(train_records, band=band)
    basis, singular_values = compute_pod_basis(train_matrix, rank=rank)
    n_time = train_matrix.shape[1] // len(train_records)

    # Fit source scaler from training filenames
    train_source_raw = np.stack([
        parse_source_params_mw7(rec.path) for rec in train_records
    ])  # (n_train, p)
    scaler = SourceScaler.fit(train_source_raw)

    # Build augmented training data
    reduced_train = basis.T @ train_matrix  # (r, N_train*T)
    source_tile_train = build_source_tile(train_records, scaler, n_time=n_time)

    A_hat, B_hat = infer_forced_operator(
        reduced_train, source_tile_train, regularization=regularization
    )

    # Training one-step fit
    Bu_train = B_hat @ source_tile_train[:, :-1]
    one_step_train = A_hat @ reduced_train[:, :-1] + Bu_train
    train_one_step = relative_frobenius_error(reduced_train[:, 1:], one_step_train)

    # Validation
    val_rows = [
        evaluate_forced_event(record, band=band, basis=basis,
                              A_hat=A_hat, B_hat=B_hat, scaler=scaler)
        for record in val_records
    ]

    stability_A = operator_stability(A_hat)
    p = B_hat.shape[1]
    b_frobenius = float(np.linalg.norm(B_hat))

    metadata = {
        "event_class": event_class,
        "frequency_band": band_name,
        "band_hz": band,
        "rank": rank,
        "n_source_params": p,
        "source_param_names": ["HYP_y", "Vel"],
        "source_mean": scaler.mean.tolist(),
        "source_std": scaler.std.tolist(),
        "regularization": regularization,
        "n_train_events": len(train_records),
        "n_val_events": len(val_records),
        "n_spatial_dof": int(train_matrix.shape[0]),
        "train_one_step_error": float(train_one_step),
        "val_one_step_error": float(np.mean([r["relative_one_step_error"] for r in val_rows])),
        "val_rollout_error": float(np.mean([r["relative_state_error"] for r in val_rows])),
        "val_reduced_rollout_error": float(np.mean([r["relative_reduced_trajectory_error"] for r in val_rows])),
        "B_frobenius_norm": b_frobenius,
        **{f"A_{k}": v for k, v in stability_A.items()},
    }

    if save_artifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "A_hat.npy", A_hat)
        np.save(output_dir / "B_hat.npy", B_hat)
        np.save(output_dir / "V.npy", basis)
        np.save(output_dir / "singular_values.npy", singular_values)
        scaler_dict = {"mean": scaler.mean.tolist(), "std": scaler.std.tolist()}
        with (output_dir / "source_scaler.json").open("w") as fh:
            json.dump(scaler_dict, fh, indent=2)
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        metrics_path = output_dir / "validation_metrics.csv"
        with metrics_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(val_rows[0].keys()))
            writer.writeheader()
            writer.writerows(val_rows)
        print(f"Wrote forced OpInf model to {output_dir}")

    print(json.dumps(metadata, indent=2))
    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_PROCESSED / "events")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS / "lowfreq_forced_opinf_mw7")
    parser.add_argument("--event-class", default="mw7")
    parser.add_argument("--band", choices=sorted(FREQUENCY_BANDS), default="low_0_025hz")
    parser.add_argument("--rank", type=int, default=100)
    parser.add_argument("--regularization", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-events", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = run_forced_training(
        data_root=args.data_root,
        output_dir=args.output_dir,
        event_class=args.event_class,
        band_name=args.band,
        rank=args.rank,
        regularization=args.regularization,
        train_fraction=args.train_fraction,
        seed=args.seed,
        max_events=args.max_events,
    )
    print(
        f"forced OpInf: val_one_step={metadata['val_one_step_error']:.4f} "
        f"val_rollout={metadata['val_rollout_error']:.4f} "
        f"A_spectral_radius={metadata['A_spectral_radius']:.4f}"
    )


if __name__ == "__main__":
    main()
