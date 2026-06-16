"""Quadratic source-forced OpInf ROM for low-frequency Mw7 wavefields.

Model:
    q_{k+1} = A q_k + H (q_k ⊗_tri q_k) + B u^(e)

where ⊗_tri is the upper-triangular Kronecker product of size r(r+1)//2,
eliminating the redundant cross-terms of the full Kronecker.

Design vector at step k:
    d_k = [q_k; kron_tri(q_k); u^(e)]   shape (r + r(r+1)//2 + p,)

At r=25: A→625, H→325, B→50 parameters; training provides 80×117=9360 pairs
→ overdetermination ratio 9360/1000 = 9.4× per output row.

The script:
  1. Fits quadratic forced model at rank 25 across lambda values.
  2. Fits linear forced model at rank 25 (same data) for direct comparison.
  3. Reports one-step error, rollout error, spectral radius of A.

Usage
-----
    python -m src.quadratic_opinf
    python -m src.quadratic_opinf --rank 25 --lambdas 1e-4 1e-2 1.0 10.0 100.0
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .balanced_pod_analysis import FREQUENCY_BANDS
from .config import DATA_PROCESSED, RESULTS_TABLES
from .train_forced_opinf import (
    SourceScaler,
    build_source_tile,
    parse_source_params_mw7,
)
from .train_lowfreq_opinf import (
    assemble_snapshot_matrix,
    compute_pod_basis,
    load_bandpassed_snapshots,
    operator_stability,
    relative_frobenius_error,
    split_records,
    discover_processed_events,
)

DEFAULT_RANK = 25
DEFAULT_LAMBDAS = [1e-4, 1e-2, 1.0, 10.0, 100.0]
DEFAULT_EVENT_CLASS = "mw7"
DEFAULT_BAND = "low_0_025hz"
MAX_ROLLOUT_NORM = 1e6   # guard against quadratic blow-up


# ---------------------------------------------------------------------------
# Upper-triangular Kronecker product
# ---------------------------------------------------------------------------

def kron_upper_indices(r: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (ii, jj) index arrays for upper-triangular Kronecker of size r."""
    return np.triu_indices(r)


def kron_upper(q: np.ndarray, ii: np.ndarray, jj: np.ndarray) -> np.ndarray:
    """Compute upper-triangular Kronecker product.

    Parameters
    ----------
    q   : (r, T) state matrix
    ii, jj : pre-computed triu index arrays from kron_upper_indices(r)

    Returns
    -------
    (r*(r+1)//2, T) array  q[ii] * q[jj]
    """
    return q[ii] * q[jj]


# ---------------------------------------------------------------------------
# Quadratic operator inference
# ---------------------------------------------------------------------------

def infer_quadratic_operator(
    states: np.ndarray,       # (r, N*T)
    source_tile: np.ndarray,  # (p, N*T)
    regularization: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit q_{k+1} = A q_k + H kron_tri(q_k) + B u via augmented least squares.

    Returns (A_hat, H_hat, B_hat).
    """
    r = states.shape[0]
    ii, jj = kron_upper_indices(r)

    q_cur = states[:, :-1].astype(np.float64)      # (r, M)  M = N*(T-1)
    q_kron = kron_upper(q_cur, ii, jj)              # (r*(r+1)//2, M)
    u_cur = source_tile[:, :-1].astype(np.float64)  # (p, M)
    q_nxt = states[:, 1:].astype(np.float64)        # (r, M)

    design = np.vstack([q_cur, q_kron, u_cur])      # (r + n_tri + p, M)
    n_aug = design.shape[0]

    gram = design @ design.T                         # (n_aug, n_aug)
    rhs = q_nxt @ design.T                          # (r, n_aug)
    reg = regularization * np.eye(n_aug, dtype=np.float64)

    M_T = np.linalg.solve(gram + reg, rhs.T)        # (n_aug, r)
    M = M_T.T                                        # (r, n_aug)

    n_tri = len(ii)
    A_hat = M[:, :r].astype(np.float32)
    H_hat = M[:, r:r + n_tri].astype(np.float32)
    B_hat = M[:, r + n_tri:].astype(np.float32)
    return A_hat, H_hat, B_hat


# ---------------------------------------------------------------------------
# Rollout integrator
# ---------------------------------------------------------------------------

def integrate_quadratic(
    A: np.ndarray,
    H: np.ndarray,
    B: np.ndarray,
    source_u: np.ndarray,   # (p,) normalised, constant
    initial_state: np.ndarray,
    n_steps: int,
) -> tuple[np.ndarray, bool]:
    """Roll out q_{k+1} = A q_k + H kron(q_k) + Bu.

    Returns (trajectory, blew_up). trajectory[:, k] = nan for k beyond blow-up.
    """
    r = A.shape[0]
    ii, jj = kron_upper_indices(r)
    Bu = (B @ source_u).astype(np.float64)
    A64 = A.astype(np.float64)
    H64 = H.astype(np.float64)

    traj = np.empty((r, n_steps), dtype=np.float64)
    traj[:, 0] = initial_state.astype(np.float64)
    blew_up = False
    for k in range(1, n_steps):
        qk = traj[:, k - 1]
        qkk = qk[ii] * qk[jj]
        traj[:, k] = A64 @ qk + H64 @ qkk + Bu
        if np.linalg.norm(traj[:, k]) > MAX_ROLLOUT_NORM:
            traj[:, k:] = np.nan
            blew_up = True
            break
    return traj.astype(np.float32), blew_up


# ---------------------------------------------------------------------------
# Per-event evaluation
# ---------------------------------------------------------------------------

def evaluate_quadratic_event(
    record,
    band: tuple[float, float],
    basis: np.ndarray,
    A_hat: np.ndarray,
    H_hat: np.ndarray,
    B_hat: np.ndarray,
    scaler: SourceScaler,
    ii: np.ndarray,
    jj: np.ndarray,
) -> dict:
    snapshots = load_bandpassed_snapshots(record, band=band)
    reduced = (basis.T @ snapshots).astype(np.float64)  # (r, T)
    source_u = scaler.transform(parse_source_params_mw7(record.path))

    # One-step prediction
    q_cur = reduced[:, :-1]
    q_kron = kron_upper(q_cur, ii, jj)
    Bu = (B_hat.astype(np.float64) @ source_u)[:, None]
    one_step = A_hat.astype(np.float64) @ q_cur + H_hat.astype(np.float64) @ q_kron + Bu
    one_step_err = relative_frobenius_error(reduced[:, 1:], one_step)

    # Rollout from true first state
    traj, blew_up = integrate_quadratic(A_hat, H_hat, B_hat, source_u, reduced[:, 0], reduced.shape[1])

    if blew_up:
        rollout_err = float("nan")
        state_err = float("nan")
    else:
        rollout_err = relative_frobenius_error(reduced, traj.astype(np.float64))
        reconstructed = basis @ traj
        state_err = relative_frobenius_error(snapshots, reconstructed)

    return {
        "one_step_error": one_step_err,
        "rollout_reduced_error": rollout_err,
        "rollout_state_error": state_err,
        "blew_up": blew_up,
    }


# ---------------------------------------------------------------------------
# Linear forced model at the same rank (reference)
# ---------------------------------------------------------------------------

def infer_linear_forced(
    states: np.ndarray,
    source_tile: np.ndarray,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit linear q_{k+1} = A q_k + B u (same interface as forced training)."""
    from .train_forced_opinf import infer_forced_operator
    return infer_forced_operator(states, source_tile, regularization)


def evaluate_linear_event(
    record,
    band: tuple[float, float],
    basis: np.ndarray,
    A_hat: np.ndarray,
    B_hat: np.ndarray,
    scaler: SourceScaler,
) -> dict:
    from .train_forced_opinf import integrate_forced_operator
    snapshots = load_bandpassed_snapshots(record, band=band)
    reduced = basis.T @ snapshots
    source_u = scaler.transform(parse_source_params_mw7(record.path))

    Bu = (B_hat @ source_u).astype(np.float32)
    one_step = A_hat @ reduced[:, :-1].astype(np.float32) + Bu[:, None]
    one_step_err = relative_frobenius_error(reduced[:, 1:], one_step)

    traj = integrate_forced_operator(A_hat, B_hat, source_u, reduced[:, 0], reduced.shape[1])
    reconstructed = basis @ traj
    rollout_err = relative_frobenius_error(reduced, traj)
    state_err = relative_frobenius_error(snapshots, reconstructed)
    return {
        "one_step_error": one_step_err,
        "rollout_reduced_error": rollout_err,
        "rollout_state_error": state_err,
        "blew_up": False,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    data_root: Path,
    output_csv: Path,
    rank: int,
    lambdas: list[float],
    max_events: int,
    train_fraction: float,
) -> None:
    band = FREQUENCY_BANDS[DEFAULT_BAND]
    records = discover_processed_events(data_root, event_class=DEFAULT_EVENT_CLASS)
    if max_events:
        records = records[:max_events]
    train_records, val_records = split_records(records, train_fraction, seed=42)
    print(f"Train: {len(train_records)}  Val: {len(val_records)}")

    # POD basis at target rank
    train_matrix = assemble_snapshot_matrix(train_records, band=band)
    basis, sv = compute_pod_basis(train_matrix, rank=rank)
    n_time = train_matrix.shape[1] // len(train_records)
    energy_captured = float((sv[:rank] ** 2).sum() / (sv ** 2).sum())
    print(f"Rank {rank}: {energy_captured:.4f} energy captured ({len(sv)} total modes)")

    # Source scaler
    train_source_raw = np.stack([parse_source_params_mw7(r.path) for r in train_records])
    scaler = SourceScaler.fit(train_source_raw)

    # Reduced training states and source tile
    reduced_train = basis.T @ train_matrix  # (r, N_train*T)
    source_tile_train = build_source_tile(train_records, scaler, n_time=n_time)

    r = rank
    ii, jj = kron_upper_indices(r)
    n_tri = len(ii)
    p = source_tile_train.shape[0]
    M_quad = r + n_tri + p  # total unknowns per output dimension
    M_lin = r + p
    N_pairs = reduced_train.shape[1] - len(train_records)  # exclude last step of each event
    # Actually: N*(T-1) pairs total
    N_pairs_total = (train_matrix.shape[1] // n_time) * (n_time - 1)
    print(
        f"Quadratic design: {M_quad} unknowns ({r} A + {n_tri} H + {p} B), "
        f"{N_pairs_total} training pairs, "
        f"overdetermination {N_pairs_total / M_quad:.1f}×"
    )

    rows = []

    for lam in lambdas:
        print(f"\n=== lambda={lam:.0e} ===")

        # --- Quadratic model ---
        A_q, H_q, B_q = infer_quadratic_operator(reduced_train, source_tile_train, lam)
        stab_q = operator_stability(A_q)
        H_frob = float(np.linalg.norm(H_q))
        B_frob = float(np.linalg.norm(B_q))

        val_q = [evaluate_quadratic_event(rec, band, basis, A_q, H_q, B_q, scaler, ii, jj)
                 for rec in val_records]
        n_blowup = sum(v["blew_up"] for v in val_q)
        valid_q = [v for v in val_q if not v["blew_up"]]
        mean_one_step_q = float(np.mean([v["one_step_error"] for v in val_q]))
        mean_rollout_q = float(np.mean([v["rollout_state_error"] for v in valid_q])) if valid_q else float("nan")

        # --- Linear model (same rank, same lambda) ---
        A_l, B_l = infer_linear_forced(reduced_train, source_tile_train, lam)
        stab_l = operator_stability(A_l)
        val_l = [evaluate_linear_event(rec, band, basis, A_l, B_l, scaler) for rec in val_records]
        mean_one_step_l = float(np.mean([v["one_step_error"] for v in val_l]))
        mean_rollout_l = float(np.mean([v["rollout_state_error"] for v in val_l]))

        print(
            f"  Quadratic: one_step={mean_one_step_q:.4f}  rollout={mean_rollout_q:.4f}"
            f"  rho(A)={stab_q['spectral_radius']:.4f}  |H|={H_frob:.4f}"
            f"  blowups={n_blowup}/{len(val_records)}"
        )
        print(
            f"  Linear:    one_step={mean_one_step_l:.4f}  rollout={mean_rollout_l:.4f}"
            f"  rho(A)={stab_l['spectral_radius']:.4f}"
        )

        rows.append({
            "model": "quadratic",
            "rank": rank,
            "lambda": lam,
            "n_tri": n_tri,
            "n_design_cols": M_quad,
            "overdetermination": round(N_pairs_total / M_quad, 2),
            "energy_captured": round(energy_captured, 4),
            "val_one_step": mean_one_step_q,
            "val_rollout": mean_rollout_q,
            "A_spectral_radius": stab_q["spectral_radius"],
            "H_frobenius": H_frob,
            "B_frobenius": B_frob,
            "n_blowups": n_blowup,
            "n_val": len(val_records),
        })
        rows.append({
            "model": "linear",
            "rank": rank,
            "lambda": lam,
            "n_tri": 0,
            "n_design_cols": M_lin,
            "overdetermination": round(N_pairs_total / M_lin, 2),
            "energy_captured": round(energy_captured, 4),
            "val_one_step": mean_one_step_l,
            "val_rollout": mean_rollout_l,
            "A_spectral_radius": stab_l["spectral_radius"],
            "H_frobenius": 0.0,
            "B_frobenius": float(np.linalg.norm(B_l)),
            "n_blowups": 0,
            "n_val": len(val_records),
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved → {output_csv}")

    print("\n=== Summary ===")
    print(f"{'model':<12} {'lambda':>8} {'one_step':>10} {'rollout':>10} {'rho':>8} {'blowups':>8}")
    for row in rows:
        blowup_str = f"{int(row['n_blowups'])}/{int(row['n_val'])}" if row["model"] == "quadratic" else "—"
        print(
            f"{row['model']:<12} {row['lambda']:>8.0e}"
            f" {row['val_one_step']:>10.4f} {row['val_rollout']:>10.4f}"
            f" {row['A_spectral_radius']:>8.4f} {blowup_str:>8}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DATA_PROCESSED / "events")
    p.add_argument("--rank", type=int, default=DEFAULT_RANK)
    p.add_argument("--lambdas", type=float, nargs="+", default=DEFAULT_LAMBDAS)
    p.add_argument("--max-events", type=int, default=100)
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument(
        "--output-csv",
        type=Path,
        default=RESULTS_TABLES / "quadratic_opinf_mw7.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(
        data_root=args.data_root,
        output_csv=args.output_csv,
        rank=args.rank,
        lambdas=args.lambdas,
        max_events=args.max_events,
        train_fraction=args.train_fraction,
    )


if __name__ == "__main__":
    main()
