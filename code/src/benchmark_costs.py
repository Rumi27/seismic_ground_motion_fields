"""Computational cost benchmark for POD, Gappy POD, and OpInf at paper scales.

Measures wall-clock time and reports theoretical FLOP counts for:
  1. Snapshot matrix assembly (data loading)
  2. POD basis computation via Gram matrix + eigh
  3. Gappy POD precompute step (fixed per station layout)
  4. Gappy POD per-timestep inference (variable per observation)
  5. Source-forced OpInf fitting

All timings use the actual data and problem dimensions from the paper
(Mw7 low-frequency, rank=151, n_obs=128, n_train=80, n_val=20).

Usage
-----
    python -m src.benchmark_costs
    python -m src.benchmark_costs --n-repeats 5
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .balanced_pod_analysis import FREQUENCY_BANDS
from .config import DATA_PROCESSED, DATA_RAW, N_COMPONENTS, RESULTS_TABLES
from .gappy_pod_reconstruction import (
    find_reference_h5,
    load_station_coordinates,
    select_stations_random,
    solve_gappy_coefficients,
    reconstruct_from_coefficients,
    station_dof_indices,
)
from .train_forced_opinf import (
    SourceScaler,
    build_source_tile,
    infer_forced_operator,
    parse_source_params_mw7,
)
from .train_lowfreq_opinf import (
    assemble_snapshot_matrix,
    compute_pod_basis,
    discover_processed_events,
    split_records,
)

RANK = 151
N_OBS = 128
BAND = "low_0_025hz"
EVENT_CLASS = "mw7"
MAX_EVENTS = 100
TRAIN_FRACTION = 0.8
SEED = 42


def timer(label: str, fn, n_repeats: int = 3):
    """Run fn() n_repeats times, return (mean_seconds, result_of_last_call)."""
    result = None
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    mean_t = float(np.mean(times))
    std_t = float(np.std(times))
    print(f"  {label:<52s}  {mean_t*1000:8.1f} ms  ±{std_t*1000:6.1f} ms")
    return mean_t, result


def flops_gram(m: int, n: int) -> int:
    """FLOPs for A^T @ A, A shape (m, n) → symmetric (n, n) Gram."""
    return m * n * (n + 1)   # ≈ 2 × m × n²/2


def flops_eigh(n: int) -> int:
    """FLOPs for symmetric eigendecomposition (rough O(n³) estimate)."""
    return 4 * n ** 3 // 3


def flops_matmul(m: int, k: int, n: int) -> int:
    """FLOPs for (m×k) @ (k×n) matrix multiply."""
    return 2 * m * k * n


def flops_chol(n: int) -> int:
    """FLOPs for Cholesky factorisation of (n×n) symmetric matrix."""
    return n ** 3 // 3


def flops_tri_solve(n: int, rhs: int) -> int:
    """FLOPs for triangular back-substitution: n×n factor, rhs right-hand sides."""
    return n ** 2 * rhs


def fmt_flops(n: int) -> str:
    if n >= 1e12:
        return f"{n/1e12:.2f} TFLOPs"
    if n >= 1e9:
        return f"{n/1e9:.2f} GFLOPs"
    if n >= 1e6:
        return f"{n/1e6:.2f} MFLOPs"
    return f"{n/1e3:.2f} kFLOPs"


def run(data_root: Path, raw_root: Path, n_repeats: int, output_csv: Path) -> None:
    band = FREQUENCY_BANDS[BAND]
    records = discover_processed_events(data_root, event_class=EVENT_CLASS)[:MAX_EVENTS]
    train_records, val_records = split_records(records, TRAIN_FRACTION, SEED)
    n_train = len(train_records)

    print(f"\n{'='*72}")
    print(f"Computational Cost Benchmark — paper scales")
    print(f"  Events:   {n_train} train / {len(val_records)} val")
    print(f"  Rank:     {RANK}")
    print(f"  Stations observed: {N_OBS}")
    print(f"  Repeats:  {n_repeats}")
    print(f"{'='*72}\n")

    # ------------------------------------------------------------------
    # 0. Snapshot matrix assembly (I/O + bandpass)
    # ------------------------------------------------------------------
    print("[ 0 ] Data loading & bandpass filter")
    t_assemble, train_matrix = timer(
        "Load + bandpass 80 events → snapshot matrix",
        lambda: assemble_snapshot_matrix(train_records, band=band),
        n_repeats=1,   # single run (I/O dominated)
    )
    m_dof, M_snaps = train_matrix.shape
    n_time = M_snaps // n_train
    print(f"        Matrix shape: ({m_dof:,} dof × {M_snaps:,} snapshots)  "
          f"= {train_matrix.nbytes / 1e9:.2f} GB float32\n")

    # ------------------------------------------------------------------
    # 1. POD basis computation
    # ------------------------------------------------------------------
    print("[ 1 ] POD basis computation (training, one-off)")

    def _gram():
        return np.asarray(train_matrix.T @ train_matrix, dtype=np.float64)

    t_gram, gram = timer("  Gram matrix  X^T X  (M×M, M=9440)", _gram, n_repeats)
    f_gram = flops_gram(m_dof, M_snaps)

    def _eigh():
        eigvals, eigvecs = np.linalg.eigh(gram)
        return eigvals, eigvecs

    t_eigh, (eigvals, eigvecs) = timer("  eigh(Gram)  (9440×9440)", _eigh, n_repeats)
    f_eigh = flops_eigh(M_snaps)

    # Sort and truncate to rank
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order][:RANK]
    eigvecs = eigvecs[:, order][:, :RANK]
    sv = np.sqrt(np.maximum(eigvals, 0.0))

    def _basis():
        return (train_matrix @ eigvecs / sv).astype(np.float32)

    t_basis, basis = timer(f"  Basis projection  X @ V_r  (→ {m_dof:,}×{RANK})", _basis, n_repeats)
    f_basis = flops_matmul(m_dof, M_snaps, RANK)

    t_pod_total = t_gram + t_eigh + t_basis
    f_pod_total = f_gram + f_eigh + f_basis
    print(f"\n  POD total:  {t_pod_total*1000:.0f} ms  |  {fmt_flops(f_pod_total)}\n")

    # ------------------------------------------------------------------
    # 2. Gappy POD — precompute (per station layout, amortised)
    # ------------------------------------------------------------------
    print("[ 2 ] Gappy POD — precompute (per layout, amortised over all events & time steps)")
    rng = np.random.default_rng(SEED)
    n_stations = m_dof // N_COMPONENTS
    obs_stations = select_stations_random(n_stations, N_OBS, rng)
    obs_dofs = station_dof_indices(obs_stations, n_stations)
    V_obs = basis[obs_dofs].astype(np.float64)   # (n_obs*N_C, r)
    n_obs_dof = V_obs.shape[0]

    def _precompute():
        gram_obs = V_obs.T @ V_obs   # (r, r)
        lam = 1e-6
        return cho_factor(gram_obs + lam * np.eye(RANK))

    t_pre, L = timer(
        f"  V_obs^T V_obs + Cholesky  ({n_obs_dof}×{RANK} → {RANK}×{RANK})",
        _precompute, n_repeats
    )
    f_pre = flops_gram(n_obs_dof, RANK) + flops_chol(RANK)
    print(f"        ≡ {fmt_flops(f_pre)}  (done once per station layout)\n")

    # ------------------------------------------------------------------
    # 3. Gappy POD — per-timestep inference
    # ------------------------------------------------------------------
    print("[ 3 ] Gappy POD — per-timestep inference (deployment critical path)")

    # Load one validation event's worth of snapshots
    from .train_lowfreq_opinf import load_bandpassed_snapshots
    snaps = load_bandpassed_snapshots(val_records[0], band=band)   # (n_dof, 118)
    x_obs_all = snaps[obs_dofs].astype(np.float64)                 # (n_obs_dof, 118)

    # Precompute the Cholesky factor once
    gram_obs = V_obs.T @ V_obs
    L_pre = cho_factor(gram_obs + 1e-6 * np.eye(RANK))

    def _per_timestep():
        # Single time step: V_obs^T x + Cholesky solve → coefficients → reconstruct
        x_k = x_obs_all[:, 0]
        rhs = V_obs.T @ x_k                    # (r,) — 2*n_obs_dof*r FLOPs
        coeffs = cho_solve(L_pre, rhs)          # (r,) — 2*r² FLOPs
        return basis.astype(np.float64) @ coeffs # (n_dof,) — 2*n_dof*r FLOPs

    t_step, _ = timer(
        f"  V_obs^T x_k + cho_solve + V_r a_k  (1 step, rank={RANK})",
        _per_timestep, n_repeats
    )
    f_step = (2 * n_obs_dof * RANK          # V_obs^T @ x_k
              + flops_tri_solve(RANK, 1) * 2  # forward + back
              + flops_matmul(m_dof, RANK, 1)) # V_r @ coeffs

    # Full event (118 time steps) — with pre-computed Cholesky
    def _full_event():
        rhs_all = V_obs.T @ x_obs_all               # (r, 118)
        coeffs_all = cho_solve(L_pre, rhs_all)       # (r, 118)
        return basis.astype(np.float64) @ coeffs_all # (n_dof, 118)

    t_event, _ = timer(
        f"  Full 118-step event  (V_obs^T X + batch cho_solve + V_r A)",
        _full_event, n_repeats
    )
    f_event = (flops_matmul(RANK, n_obs_dof, n_time)
               + flops_tri_solve(RANK, n_time) * 2
               + flops_matmul(m_dof, RANK, n_time))

    print(f"\n  Per-step throughput:  {1/t_step:.0f} solves/s  "
          f"({t_step*1000:.2f} ms/step)  |  {fmt_flops(f_step)}")
    print(f"  Per-event (118 steps): {t_event*1000:.1f} ms  |  {fmt_flops(f_event)}\n")

    # ------------------------------------------------------------------
    # 4. OpInf fitting (training, one-off)
    # ------------------------------------------------------------------
    print("[ 4 ] OpInf fitting (training, one-off)")

    reduced_train = basis.T @ train_matrix   # (r, N_train*T)
    train_source_raw = np.stack([parse_source_params_mw7(r.path) for r in train_records])
    scaler = SourceScaler.fit(train_source_raw)
    source_tile = build_source_tile(train_records, scaler, n_time=n_time)
    p = source_tile.shape[0]   # 2 source params
    n_aug = RANK + p           # augmented state size

    def _basis_project():
        return (basis.T @ train_matrix).astype(np.float64)

    t_proj, _ = timer(
        f"  POD projection  V_r^T X  ({RANK}×{m_dof:,}) @ ({m_dof:,}×{M_snaps})",
        _basis_project, n_repeats
    )
    f_proj = flops_matmul(RANK, m_dof, M_snaps)

    def _opinf():
        return infer_forced_operator(
            reduced_train.astype(np.float64),
            source_tile,
            regularization=1e-6,
        )

    t_opinf, _ = timer(
        f"  Augmented Gram + solve  ({n_aug}×{n_aug})",
        _opinf, n_repeats
    )
    n_pairs = M_snaps - n_train   # exclude last step of each event
    f_opinf = (flops_gram(n_pairs, n_aug)       # design @ design^T
               + flops_matmul(RANK, n_pairs, n_aug)  # rhs
               + RANK * (2 * n_aug ** 3 // 3))  # n_r independent solves

    t_opinf_total = t_proj + t_opinf
    f_opinf_total = f_proj + f_opinf
    print(f"\n  OpInf total:  {t_opinf_total*1000:.0f} ms  |  {fmt_flops(f_opinf_total)}\n")

    # ------------------------------------------------------------------
    # 5. OpInf rollout (per event, deployment)
    # ------------------------------------------------------------------
    print("[ 5 ] OpInf rollout (per event, deployment)")

    A_hat, B_hat = _opinf()
    from .train_forced_opinf import integrate_forced_operator
    source_u = scaler.transform(parse_source_params_mw7(val_records[0].path))

    reduced_val = (basis.T @ load_bandpassed_snapshots(val_records[0], band=band)).astype(np.float32)

    def _rollout():
        return integrate_forced_operator(A_hat, B_hat, source_u, reduced_val[:, 0], n_steps=n_time)

    t_rollout, traj = timer(
        f"  OpInf rollout  {n_time} steps  (r={RANK})",
        _rollout, n_repeats
    )
    f_rollout = n_time * (flops_matmul(RANK, RANK, 1) + RANK)  # A@q + Bu per step

    def _lift():
        return (basis @ traj).astype(np.float32)

    t_lift, _ = timer(
        f"  Lift to full field  V_r @ traj  ({m_dof:,}×{RANK})",
        _lift, n_repeats
    )
    f_lift = flops_matmul(m_dof, RANK, n_time)

    print(f"\n  Rollout + lift total:  {(t_rollout+t_lift)*1000:.1f} ms/event  "
          f"|  {fmt_flops(f_rollout+f_lift)}\n")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    print(f"{'Operation':<48} {'Time':>10}  {'FLOPs':>14}")
    print(f"{'-'*72}")

    rows_summary = [
        ("TRAINING (one-off)", None, None),
        ("  Data load + bandpass (80 events)", f"{t_assemble*1000:.0f} ms", "I/O"),
        ("  POD Gram matrix  X^T X", f"{t_gram*1000:.0f} ms", fmt_flops(f_gram)),
        ("  POD eigh(Gram)", f"{t_eigh*1000:.0f} ms", fmt_flops(f_eigh)),
        ("  POD basis projection  X @ V_r", f"{t_basis*1000:.0f} ms", fmt_flops(f_basis)),
        ("  POD total (excl. I/O)", f"{t_pod_total*1000:.0f} ms", fmt_flops(f_pod_total)),
        ("  OpInf POD projection  V_r^T X", f"{t_proj*1000:.0f} ms", fmt_flops(f_proj)),
        ("  OpInf Gram + solve", f"{t_opinf*1000:.0f} ms", fmt_flops(f_opinf)),
        ("  OpInf total (excl. POD)", f"{t_opinf_total*1000:.0f} ms", fmt_flops(f_opinf_total)),
        ("PRECOMPUTE (per station layout)", None, None),
        ("  Gappy Gram + Cholesky", f"{t_pre*1000:.2f} ms", fmt_flops(f_pre)),
        ("INFERENCE (per event, deployment)", None, None),
        (f"  Gappy POD — 1 time step", f"{t_step*1000:.2f} ms", fmt_flops(f_step)),
        (f"  Gappy POD — full 118 steps", f"{t_event*1000:.1f} ms", fmt_flops(f_event)),
        (f"  OpInf rollout  {n_time} steps", f"{t_rollout*1000:.2f} ms", fmt_flops(f_rollout)),
        (f"  OpInf lift to full field", f"{t_lift*1000:.1f} ms", fmt_flops(f_lift)),
        (f"  OpInf total (rollout + lift)", f"{(t_rollout+t_lift)*1000:.1f} ms", fmt_flops(f_rollout+f_lift)),
    ]

    for op, t_str, f_str in rows_summary:
        if t_str is None:
            print(f"\n  {op}")
        else:
            print(f"  {op:<46} {t_str:>10}  {f_str:>14}")

    print(f"\n{'='*72}")
    print("PROBLEM DIMENSIONS")
    print(f"{'='*72}")
    print(f"  Spatial DOF:              {m_dof:>10,}  ({n_stations:,} stations × {N_COMPONENTS} components)")
    print(f"  Training snapshots:       {M_snaps:>10,}  ({n_train} events × {n_time} steps)")
    print(f"  POD rank:                 {RANK:>10}")
    print(f"  Observed stations:        {N_OBS:>10}  → {n_obs_dof} observed DOF")
    print(f"  Snapshot matrix size:     {train_matrix.nbytes/1e9:>9.2f} GB  (float32)")
    print(f"  Gram matrix size:         {gram.nbytes/1e6:>9.1f} MB  (float64, {M_snaps}×{M_snaps})")
    print(f"  Basis size (V_r):         {basis.nbytes/1e6:>9.1f} MB  (float32, {m_dof:,}×{RANK})")

    # Save CSV
    import csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_csv = [
        {"stage": "pod_gram",           "time_ms": t_gram*1e3,           "flops": f_gram,      "phase": "train"},
        {"stage": "pod_eigh",           "time_ms": t_eigh*1e3,           "flops": f_eigh,      "phase": "train"},
        {"stage": "pod_basis_project",  "time_ms": t_basis*1e3,          "flops": f_basis,     "phase": "train"},
        {"stage": "pod_total",          "time_ms": t_pod_total*1e3,      "flops": f_pod_total, "phase": "train"},
        {"stage": "opinf_pod_project",  "time_ms": t_proj*1e3,           "flops": f_proj,      "phase": "train"},
        {"stage": "opinf_gram_solve",   "time_ms": t_opinf*1e3,          "flops": f_opinf,     "phase": "train"},
        {"stage": "opinf_total",        "time_ms": t_opinf_total*1e3,    "flops": f_opinf_total, "phase": "train"},
        {"stage": "gappy_precompute",   "time_ms": t_pre*1e3,            "flops": f_pre,       "phase": "precompute"},
        {"stage": "gappy_per_step",     "time_ms": t_step*1e3,           "flops": f_step,      "phase": "inference"},
        {"stage": "gappy_full_event",   "time_ms": t_event*1e3,          "flops": f_event,     "phase": "inference"},
        {"stage": "opinf_rollout",      "time_ms": t_rollout*1e3,        "flops": f_rollout,   "phase": "inference"},
        {"stage": "opinf_lift",         "time_ms": t_lift*1e3,           "flops": f_lift,      "phase": "inference"},
        {"stage": "opinf_rollout_lift", "time_ms": (t_rollout+t_lift)*1e3,"flops": f_rollout+f_lift, "phase": "inference"},
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["stage", "phase", "time_ms", "flops"])
        writer.writeheader()
        writer.writerows(rows_csv)
    print(f"\nSaved → {output_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DATA_PROCESSED / "events")
    p.add_argument("--raw-root", type=Path, default=DATA_RAW)
    p.add_argument("--n-repeats", type=int, default=3)
    p.add_argument("--output-csv", type=Path,
                   default=RESULTS_TABLES / "computational_costs_mw7.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(
        data_root=args.data_root,
        raw_root=args.raw_root,
        n_repeats=args.n_repeats,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
