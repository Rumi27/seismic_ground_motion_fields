from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_FIGURES = PROJECT_ROOT / "results" / "figures"
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"
OUTPUTS = PROJECT_ROOT / "outputs"

N_COMPONENTS = 3
NX = 256
NY = 128
NT = 96
N_SURFACE_DOF = N_COMPONENTS * NX * NY

DEFAULT_TARGET_ENERGIES = (0.99, 0.999, 0.9999)

