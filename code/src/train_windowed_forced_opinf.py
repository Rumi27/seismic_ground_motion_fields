"""Windowed-forced and piece-wise-A OpInf for seismic wavefields.

Fix 1 – Windowed forcing
    q_{k+1} = A q_k + B (w_k u^(e))
    w_k decays to zero (exp or Hann taper), removing the non-zero steady-state
    attractor that causes rollout divergence in static-forcing OpInf.

Fix 2 – Piece-wise operators (combined with Fix 1)
    q_{k+1} = A_early q_k + B (w_k u^(e)),  k < k_switch
    q_{k+1} = A_late  q_k,                   k >= k_switch
    A_late captures the late-coda propagation regime separately from the
    source-dominated early phase.

Window types
    "static"      – constant 1 (static forcing baseline)
    "exp"         – w_k = exp(-k / tau)
    "hann"        – w_k = 0.5*(1+cos(pi*k/tau)) for k<=tau, else 0

Example
-------
    python -m src.train_windowed_forced_opinf
    python -m src.train_windowed_forced_opinf --rank 151 --max-events 100
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .balanced_pod_analysis import FREQUENCY_BANDS
from .config import DATA_PROCESSED, RESULTS_TABLES
from .train_forced_opinf import (
    SourceScaler,
    assemble_snapshot_matrix,
    build_source_tile,
    parse_source_params_mw7,
)
from .train_lowfreq_opinf import (
    DEFAULT_BAND,
    DEFAULT_EVENT_CLASS,
    compute_pod_basis,
    discover_processed_events,
    load_bandpassed_snapshots,
    operator_stability,
    relative_frobenius_error,
    split_records,
)


# ---------------------------------------------------------------------------
# Window functions
# ---------------------------------------------------------------------------

def make_forcing_window(n_time: int, window_type: str, tau: float) -> np.ndarray:
    """Return w[k] for k = 0, ..., n_time-1.

    Parameters
    ----------
    tau:
        - "exp": decay time constant in time steps
        - "hann": half-width of the one-sided cosine in time steps
        - "static": ignored
    """
    k = np.arange(n_time, dtype=np.float64)
    if window_type == "static":
        return np.ones(n_time, dtype=np.float64)
    elif window_type == "exp":
        return np.exp(-k / tau)
    elif window_type == "hann":
        # One-sided Hann: rises to 1 at k=0, decays to 0 at k=tau.
        # w_k = 0.5*(1 + cos(pi*k/tau))  for k <= tau, else 0
        return np.where(k <= tau, 0.5 * (1.0 + np.cos(np.pi * k / tau)), 0.0)
    else:
        raise ValueError(f"Unknown window_type: {window_type!r}")


def window_equilibrium_value(window_type: str, tau: float, n_time: int) -> float:
    """Value of w at the last time step (proxy for residual attractor strength)."""
    w = make_forcing_window(n_time, window_type, tau)
    return float(w[-1])


# ---------------------------------------------------------------------------
# Windowed source tile
# ---------------------------------------------------------------------------

def build_windowed_source_tile(
    records: list,
    scaler: SourceScaler,
    n_time: int,
    window: np.ndarray,
) -> np.ndarray:
    """(p, N_events * n_time) windowed source matrix: u_k = w_k * u^(e)."""
    cols = []
    for record in records:
        u = scaler.transform(parse_source_params_mw7(record.path))
        u_tiled = u[:, None] * window[None, :]   # (p, T)
        cols.append(u_tiled)
    return np.concatenate(cols, axis=1).astype(np.float64)


# ---------------------------------------------------------------------------
# Index helpers (event-boundary-safe)
# ---------------------------------------------------------------------------

def build_paired_indices(
    n_events: int,
    n_time: int,
    t_start: int,
    t_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Global column index pairs (k, k+1) for steps k in [t_start, t_end-1]
    across all events, without crossing event boundaries.
    """
    curr, nxt = [], []
    for ev in range(n_events):
        base = ev * n_time
        for k in range(t_start, t_end):
            curr.append(base + k)
            nxt.append(base + k + 1)
    return np.array(curr, dtype=np.int64), np.array(nxt, dtype=np.int64)


# ---------------------------------------------------------------------------
# Operator learning
# ---------------------------------------------------------------------------

def infer_windowed_forced(
    states: np.ndarray,               # (r, N*T)
    windowed_source: np.ndarray,      # (p, N*T)
    n_events: int,
    n_time: int,
    regularization: float,
    t_start: int = 0,
    t_end: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit q_{k+1} = A q_k + B (w_k u^(e)) on steps [t_start, t_end).

    Returns (A_hat, B_hat).
    """
    if t_end is None:
        t_end = n_time - 1
    curr_idx, nxt_idx = build_paired_indices(n_events, n_time, t_start, t_end)

    aug = np.vstack([
        states[:, curr_idx].astype(np.float64),
        windowed_source[:, curr_idx].astype(np.float64),
    ])                                               # (r+p, n_pairs)
    target = states[:, nxt_idx].astype(np.float64)  # (r,   n_pairs)

    gram = aug @ aug.T
    reg = regularization * np.eye(gram.shape[0], dtype=np.float64)
    M_T = np.linalg.solve(gram + reg, (target @ aug.T).T)   # (r+p, r)
    M = M_T.T                                                 # (r,   r+p)
    r = states.shape[0]
    return M[:, :r].astype(np.float32), M[:, r:].astype(np.float32)


def infer_autonomous_late(
    states: np.ndarray,
    n_events: int,
    n_time: int,
    regularization: float,
    t_start: int,
) -> np.ndarray:
    """Fit q_{k+1} = A_late q_k on steps [t_start, n_time-1).

    Returns A_late.
    """
    curr_idx, nxt_idx = build_paired_indices(n_events, n_time, t_start, n_time - 1)
    Q_curr = states[:, curr_idx].astype(np.float64)
    Q_nxt  = states[:, nxt_idx].astype(np.float64)
    gram = Q_curr @ Q_curr.T
    reg  = regularization * np.eye(gram.shape[0], dtype=np.float64)
    return np.linalg.solve(gram + reg, (Q_nxt @ Q_curr.T).T).T.astype(np.float32)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def integrate_windowed(
    A: np.ndarray,
    B: np.ndarray,
    source_u: np.ndarray,   # (p,) normalised
    window: np.ndarray,     # (n_time,)
    initial_state: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """Rollout with single A and windowed forcing.

    Training pairs (k, k+1) use w_k as the forcing weight, so during rollout
    the forcing at step `step` (computing q_step from q_{step-1}) uses
    w_{step-1}, i.e. window[step - 1].
    """
    Bu = (B @ source_u).astype(np.float64)
    trajectory = np.empty((A.shape[0], n_steps), dtype=np.float32)
    trajectory[:, 0] = initial_state
    for step in range(1, n_steps):
        trajectory[:, step] = A @ trajectory[:, step - 1] + window[step - 1] * Bu
    return trajectory


def integrate_piecewise_windowed(
    A_early: np.ndarray,
    A_late: np.ndarray,
    B: np.ndarray,
    source_u: np.ndarray,
    window: np.ndarray,
    k_switch: int,
    initial_state: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """Rollout with piece-wise operators and windowed forcing."""
    Bu = (B @ source_u).astype(np.float64)
    trajectory = np.empty((A_early.shape[0], n_steps), dtype=np.float32)
    trajectory[:, 0] = initial_state
    for step in range(1, n_steps):
        A = A_early if step < k_switch else A_late
        trajectory[:, step] = A @ trajectory[:, step - 1] + window[step - 1] * Bu
    return trajectory


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    window_type: str
    tau: float           # steps; ignored for "static"
    k_switch: int | None # None = single operator (Fix 1 only)
    rank: int
    regularization: float

    @property
    def label(self) -> str:
        win = f"{self.window_type}" if self.window_type == "static" else f"{self.window_type}(tau={self.tau:.0f})"
        seg = f"_kswitch{self.k_switch}" if self.k_switch is not None else ""
        return f"r{self.rank}_{win}{seg}_lam{self.regularization:.0e}"


def evaluate_one_event(
    record,
    band: tuple[float, float],
    basis: np.ndarray,
    A_early: np.ndarray,
    A_late: np.ndarray | None,
    B: np.ndarray,
    scaler: SourceScaler,
    window: np.ndarray,
    k_switch: int | None,
) -> dict[str, float]:
    snapshots = load_bandpassed_snapshots(record, band=band)
    reduced = basis.T @ snapshots      # (r, T)
    n_steps = reduced.shape[1]
    source_u = scaler.transform(parse_source_params_mw7(record.path))

    if k_switch is None or A_late is None:
        trajectory = integrate_windowed(
            A_early, B, source_u, window, reduced[:, 0], n_steps
        )
    else:
        trajectory = integrate_piecewise_windowed(
            A_early, A_late, B, source_u, window, k_switch, reduced[:, 0], n_steps
        )

    reconstructed = basis @ trajectory

    # One-step: q_{k+1} = A q_k + w_k B u  (forcing weight is w_k, the current step)
    Bu = (B @ source_u).astype(np.float64)
    A_for_one_step = A_early   # approximation — A_early dominates early pairs
    one_step_pred = A_for_one_step @ reduced[:, :-1] + window[:-1] * Bu[:, None]

    # Late-time error (last 20% of steps)
    late_start = int(0.8 * n_steps)
    late_err = relative_frobenius_error(reduced[:, late_start:], trajectory[:, late_start:])

    return {
        "rollout_error": relative_frobenius_error(snapshots, reconstructed),
        "reduced_rollout_error": relative_frobenius_error(reduced, trajectory),
        "one_step_error": relative_frobenius_error(reduced[:, 1:], one_step_pred),
        "late_rollout_error": late_err,
    }


# ---------------------------------------------------------------------------
# Main training + sweep
# ---------------------------------------------------------------------------

def run_windowed_experiment(
    cfg: ExperimentConfig,
    train_records: list,
    val_records: list,
    basis: np.ndarray,
    scaler: SourceScaler,
    reduced_train: np.ndarray,
    n_time: int,
    band: tuple[float, float],
) -> dict:
    n_events = len(train_records)
    window = make_forcing_window(n_time, cfg.window_type, cfg.tau)

    windowed_tile = build_windowed_source_tile(
        train_records, scaler, n_time, window
    )  # (p, N*T)

    if cfg.k_switch is None:
        # Fix 1 only: single A, windowed forcing
        A_early, B = infer_windowed_forced(
            reduced_train, windowed_tile,
            n_events, n_time, cfg.regularization,
            t_start=0, t_end=n_time - 1,
        )
        A_late = None
    else:
        # Fix 1 + Fix 2: piece-wise A, windowed forcing
        A_early, B = infer_windowed_forced(
            reduced_train, windowed_tile,
            n_events, n_time, cfg.regularization,
            t_start=0, t_end=cfg.k_switch,
        )
        A_late = infer_autonomous_late(
            reduced_train, n_events, n_time,
            cfg.regularization, t_start=cfg.k_switch,
        )

    stab_early = operator_stability(A_early)
    stab_late  = operator_stability(A_late) if A_late is not None else {"spectral_radius": float("nan")}

    val_metrics = [
        evaluate_one_event(
            record, band=band, basis=basis,
            A_early=A_early, A_late=A_late, B=B,
            scaler=scaler, window=window, k_switch=cfg.k_switch,
        )
        for record in val_records
    ]

    return {
        "window_type":        cfg.window_type,
        "tau":                cfg.tau,
        "k_switch":           cfg.k_switch if cfg.k_switch is not None else -1,
        "rank":               cfg.rank,
        "n_source_params":    B.shape[1],
        "regularization":     cfg.regularization,
        "w_final":            float(window[-1]),
        "rho_A_early":        stab_early["spectral_radius"],
        "rho_A_late":         stab_late["spectral_radius"],
        "B_frobenius":        float(np.linalg.norm(B)),
        "val_one_step":       float(np.mean([m["one_step_error"]        for m in val_metrics])),
        "val_rollout":        float(np.mean([m["rollout_error"]         for m in val_metrics])),
        "val_reduced_rollout":float(np.mean([m["reduced_rollout_error"] for m in val_metrics])),
        "val_late_rollout":   float(np.mean([m["late_rollout_error"]    for m in val_metrics])),
    }


def run_sweep(
    data_root: Path,
    event_class: str,
    band_name: str,
    rank: int,
    regularization: float,
    train_fraction: float,
    seed: int,
    max_events: int | None,
    configs: list[ExperimentConfig],
    output_csv: Path,
) -> Path:
    band = FREQUENCY_BANDS[band_name]
    records = discover_processed_events(data_root, event_class=event_class)
    if max_events is not None:
        records = records[:max_events]
    train_records, val_records = split_records(records, train_fraction, seed)
    print(f"Train: {len(train_records)}, Val: {len(val_records)}")

    train_matrix = assemble_snapshot_matrix(train_records, band=band)
    n_time = train_matrix.shape[1] // len(train_records)
    basis, _ = compute_pod_basis(train_matrix, rank=rank)
    reduced_train = basis.T @ train_matrix

    train_source_raw = np.stack([
        parse_source_params_mw7(rec.path) for rec in train_records
    ])
    scaler = SourceScaler.fit(train_source_raw)

    rows: list[dict] = []
    for cfg in configs:
        print(f"\n--- {cfg.label} ---")
        row = run_windowed_experiment(
            cfg, train_records, val_records,
            basis, scaler, reduced_train, n_time, band,
        )
        rows.append(row)
        print(
            f"  one_step={row['val_one_step']:.4f}  "
            f"rollout={row['val_rollout']:.4f}  "
            f"late={row['val_late_rollout']:.4f}  "
            f"w_final={row['w_final']:.2e}  "
            f"rho_early={row['rho_A_early']:.4f}  "
            f"rho_late={row['rho_A_late']:.4f}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {output_csv}")
    return output_csv


# ---------------------------------------------------------------------------
# Default sweep configuration
# ---------------------------------------------------------------------------

def default_configs(rank: int, regularization: float) -> list[ExperimentConfig]:
    cfgs: list[ExperimentConfig] = []

    # Baseline: static forcing (Fix 0 — existing §3.2b result)
    cfgs.append(ExperimentConfig("static", tau=0.0, k_switch=None,
                                 rank=rank, regularization=regularization))

    # Fix 1 only: windowed forcing, single A
    for tau in [10.0, 20.0, 30.0]:
        cfgs.append(ExperimentConfig("exp", tau=tau, k_switch=None,
                                     rank=rank, regularization=regularization))
    for tau in [20.0, 40.0]:
        cfgs.append(ExperimentConfig("hann", tau=tau, k_switch=None,
                                     rank=rank, regularization=regularization))

    # Fix 1 + Fix 2: windowed forcing + piece-wise A
    for tau, k_switch in [(20.0, 30), (20.0, 50), (30.0, 40), (30.0, 60)]:
        cfgs.append(ExperimentConfig("exp", tau=tau, k_switch=k_switch,
                                     rank=rank, regularization=regularization))

    return cfgs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_PROCESSED / "events")
    parser.add_argument("--event-class", default=DEFAULT_EVENT_CLASS)
    parser.add_argument("--band", choices=sorted(FREQUENCY_BANDS), default=DEFAULT_BAND)
    parser.add_argument("--rank", type=int, default=151)
    parser.add_argument("--regularization", type=float, default=1e-6)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-events", type=int, default=100)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=RESULTS_TABLES / "windowed_forced_opinf_sweep_mw7.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = default_configs(rank=args.rank, regularization=args.regularization)
    run_sweep(
        data_root=args.data_root,
        event_class=args.event_class,
        band_name=args.band,
        rank=args.rank,
        regularization=args.regularization,
        train_fraction=args.train_fraction,
        seed=args.seed,
        max_events=args.max_events,
        configs=configs,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
