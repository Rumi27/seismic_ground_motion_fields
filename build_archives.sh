#!/usr/bin/env bash
# build_archives.sh
# Creates the five tar.gz archives for Zenodo upload.
# Run from the project root:
#   bash zenodo_deposit/build_archives.sh
#
# Output files (placed in zenodo_deposit/):
#   seismic_rom_data_mw7.tar.gz          (~8.3 GB)
#   seismic_rom_data_mw6.tar.gz          (~8.3 GB)
#   seismic_rom_data_point_source.tar.gz (~8.3 GB)
#   seismic_rom_metadata.tar.gz          (<10 MB)
#   seismic_rom_results.tar.gz           (~5 MB)
#   seismic_rom_code.tar.gz              (<2 MB)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/zenodo_deposit"

echo "Building Zenodo archives from $ROOT"
echo "Output → $OUT"
echo ""

# --- Mw7 processed events ---
echo "[1/6] Packaging Mw7 events (~8.3 GB)..."
tar -czf "$OUT/seismic_rom_data_mw7.tar.gz" \
    -C "$ROOT/data/processed/events" mw7/
echo "  → seismic_rom_data_mw7.tar.gz ($(du -sh "$OUT/seismic_rom_data_mw7.tar.gz" | cut -f1))"

# --- Mw6 processed events ---
echo "[2/6] Packaging Mw6 events (~8.3 GB)..."
tar -czf "$OUT/seismic_rom_data_mw6.tar.gz" \
    -C "$ROOT/data/processed/events" mw6/
echo "  → seismic_rom_data_mw6.tar.gz ($(du -sh "$OUT/seismic_rom_data_mw6.tar.gz" | cut -f1))"

# --- Point-source processed events ---
echo "[3/6] Packaging point-source events (~8.3 GB)..."
tar -czf "$OUT/seismic_rom_data_point_source.tar.gz" \
    -C "$ROOT/data/processed/events" point_source/
echo "  → seismic_rom_data_point_source.tar.gz ($(du -sh "$OUT/seismic_rom_data_point_source.tar.gz" | cut -f1))"

# --- Metadata ---
echo "[4/6] Packaging metadata..."
tar -czf "$OUT/seismic_rom_metadata.tar.gz" \
    -C "$OUT" metadata/
echo "  → seismic_rom_metadata.tar.gz ($(du -sh "$OUT/seismic_rom_metadata.tar.gz" | cut -f1))"

# --- Results (tables + figures) ---
echo "[5/6] Packaging results..."
tar -czf "$OUT/seismic_rom_results.tar.gz" \
    -C "$OUT" results/
echo "  → seismic_rom_results.tar.gz ($(du -sh "$OUT/seismic_rom_results.tar.gz" | cut -f1))"

# --- Code ---
echo "[6/6] Packaging code..."
tar -czf "$OUT/seismic_rom_code.tar.gz" \
    -C "$OUT" code/
echo "  → seismic_rom_code.tar.gz ($(du -sh "$OUT/seismic_rom_code.tar.gz" | cut -f1))"

echo ""
echo "All archives created. Upload to Zenodo:"
ls -lh "$OUT"/*.tar.gz
