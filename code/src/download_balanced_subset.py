"""Download balanced GMFlow subsets from Hugging Face.

Examples
--------
Download a 30/30/30 pilot subset:

    python -m src.download_balanced_subset --per-class 30

Download a larger 100/100/100 subset:

    python -m src.download_balanced_subset --per-class 100
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .config import DATA_RAW


HF_DATASET = "Yaozhong/GMFlow_raw_simulation"
HF_RESOLVE_BASE = f"https://huggingface.co/datasets/{HF_DATASET}"
HF_API_BASE = f"https://huggingface.co/api/datasets/{HF_DATASET}/tree/main"


@dataclass(frozen=True)
class DatasetClass:
    name: str
    folder: str


DATASET_CLASSES = (
    DatasetClass("point_source", "Batch_1Hz_100_ValidationSet_flat"),
    DatasetClass("mw6", "final_results_M6_Testing"),
    DatasetClass("mw7", "final_results_M7_Testing"),
)


def list_hf_folder(folder: str) -> list[dict]:
    url = f"{HF_API_BASE}/{folder}?recursive=1"
    with urllib.request.urlopen(url, timeout=60) as response:
        data = json.load(response)
    return [
        item
        for item in data
        if item.get("type") == "file" and item.get("path", "").endswith(".h5")
    ]


def download_file(path: str, output_path: Path, expected_size: int | None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and expected_size and output_path.stat().st_size == expected_size:
        print(f"skip existing {output_path}")
        return

    partial = output_path.with_suffix(output_path.suffix + ".part")
    url = f"{HF_RESOLVE_BASE}/resolve/main/{path}"
    print(f"download {path} -> {output_path}")
    urllib.request.urlretrieve(url, partial)
    partial.replace(output_path)

    if expected_size and output_path.stat().st_size != expected_size:
        raise RuntimeError(
            f"Downloaded size mismatch for {output_path}: "
            f"{output_path.stat().st_size} != {expected_size}"
        )


def write_manifest(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class", "hf_path", "local_path", "size_bytes"],
        )
        writer.writeheader()
        writer.writerows(rows)


def run(per_class: int, output_root: Path) -> Path:
    manifest_rows = []
    for dataset_class in DATASET_CLASSES:
        files = sorted(list_hf_folder(dataset_class.folder), key=lambda item: item["path"])
        selected = files[:per_class]
        if len(selected) < per_class:
            raise RuntimeError(
                f"Requested {per_class} files for {dataset_class.name}, "
                f"but only found {len(selected)}"
            )

        for item in selected:
            hf_path = item["path"]
            size = item.get("size")
            local_path = output_root / dataset_class.folder / Path(hf_path).name
            download_file(hf_path, local_path, expected_size=size)
            manifest_rows.append(
                {
                    "class": dataset_class.name,
                    "hf_path": hf_path,
                    "local_path": str(local_path),
                    "size_bytes": size,
                }
            )

    manifest_path = output_root / f"balanced_subset_{per_class}_per_class_manifest.csv"
    write_manifest(manifest_rows, manifest_path)
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=DATA_RAW)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run(per_class=args.per_class, output_root=args.output_root)
    print(f"Wrote manifest to {manifest}")


if __name__ == "__main__":
    main()
