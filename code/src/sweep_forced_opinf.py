"""Hyperparameter sweep for source-forced OpInf (rank and ridge lambda).

Compares autonomous vs source-forced rollout errors at the same ranks/lambdas.

Example
-------
    python -m src.sweep_forced_opinf
    python -m src.sweep_forced_opinf --ranks 100 151 --lambdas 1e-4 1e-2 1 10
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .config import DATA_PROCESSED, OUTPUTS, RESULTS_TABLES
from .train_forced_opinf import run_forced_training


DEFAULT_RANKS   = [100, 151]
DEFAULT_LAMBDAS = [1e-6, 1e-4, 1e-2, 1.0, 10.0, 1000.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--max-events", type=int, default=100)
    parser.add_argument("--ranks", type=int, nargs="+", default=DEFAULT_RANKS)
    parser.add_argument("--lambdas", type=float, nargs="+", default=DEFAULT_LAMBDAS)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=RESULTS_TABLES / "lowfreq_opinf_forced_sweep_mw7.csv",
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root or (DATA_PROCESSED / "events")
    rows: list[dict] = []

    for rank in args.ranks:
        for lam in args.lambdas:
            tag = f"forced_rank{rank}_lam{lam:.0e}".replace("+", "")
            output_dir = OUTPUTS / "sweep_forced_opinf" / tag
            print(f"\n=== {tag} ===")
            meta = run_forced_training(
                data_root=data_root,
                output_dir=output_dir,
                event_class="mw7",
                band_name="low_0_025hz",
                rank=rank,
                regularization=lam,
                train_fraction=args.train_fraction,
                seed=args.seed,
                max_events=args.max_events,
                save_artifacts=False,
            )
            rows.append({
                "rank":                rank,
                "lambda":              lam,
                "n_source_params":     meta["n_source_params"],
                "train_one_step":      meta["train_one_step_error"],
                "val_one_step":        meta["val_one_step_error"],
                "val_rollout":         meta["val_rollout_error"],
                "val_reduced_rollout": meta["val_reduced_rollout_error"],
                "A_spectral_radius":   meta["A_spectral_radius"],
                "B_frobenius_norm":    meta["B_frobenius_norm"],
                "stable_rollout":      meta["A_spectral_radius"] <= 1.0,
            })

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote forced sweep table to {args.output_csv}")

    # Print comparison summary
    print("\n=== Summary (val_rollout) ===")
    print(f"{'rank':>6} {'lambda':>10} {'one_step':>10} {'rollout':>10} {'rho':>8}")
    for row in rows:
        print(
            f"{row['rank']:>6} {row['lambda']:>10.0e} "
            f"{row['val_one_step']:>10.4f} {row['val_rollout']:>10.4f} "
            f"{row['A_spectral_radius']:>8.4f}"
        )


if __name__ == "__main__":
    main()
