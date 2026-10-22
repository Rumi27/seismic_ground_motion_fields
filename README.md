# Data Repository

**Sparse Observability Limits of Long-Period Regional Seismic Ground-Motion Fields**

Zafar Avzalshoev, Pang-jo Chun  
Department of Civil Engineering, The University of Tokyo  
Contact: zavzalshoev@gmail.com

**Zenodo DOI:** https://doi.org/10.5281/zenodo.20710501  
**arXiv preprint:** [to be added]

---

## Overview

This deposit contains the analysis code associated with the above paper. Processed simulation data and precomputed results are included as separate archives (see below). All wavefields are derived from the GMFlow dataset (Shi et al., 2026, arXiv:2603.17403) of physics-based seismic simulations of the San Francisco Bay Area.

---

## Repository Contents

### Files in this deposit

| Archive | Contents | Size |
|---|---|---|
| `seismic_rom_data_mw7.tar.gz` | 100 processed Mw7 finite-rupture events (`.npy`) | ~8.3 GB |
| `seismic_rom_data_mw6.tar.gz` | 100 processed Mw6 finite-rupture events (`.npy`) | ~8.3 GB |
| `seismic_rom_data_point_source.tar.gz` | 100 processed point-source events (`.npy`) | ~8.3 GB |
| `seismic_rom_metadata.tar.gz` | Rupture summaries, station files, event manifests | < 10 MB |
| `seismic_rom_results.tar.gz` | All result tables (CSV) and figures (PDF) | ~5 MB |
| `seismic_rom_code.tar.gz` | Full Python analysis source code | < 2 MB |

---

## Data Description

### Processed event arrays (`seismic_rom_data_*.tar.gz`)

Each `.npy` file stores one seismic event as a NumPy array of shape `(3, N_stations, N_timesteps)` with `dtype=float32`.

- **Axis 0** — velocity component: `[East (x), North (y), Vertical (z)]`
- **Axis 1** — station (grid point) index: `N_stations = 62,757` for finite-rupture events
- **Axis 2** — time step: `N_timesteps = 118` at `dt ≈ 0.25 s` (sample rate ~4 Hz), covering ~30 s

Units: velocity in m/s (surface velocity waveforms bandpassed 0–1 Hz, decimated from 4 Hz raw output).

**Loading example:**
```python
import numpy as np
data = np.load("DSet_M_7.0_HYP_7.0_0_Vel_0.85_S_98_Loc_0.srf.npy")
# shape: (3, 62757, 118)
east, north, vert = data[0], data[1], data[2]
```

**Filename convention (finite-rupture):**
`DSet_M_{mag}_HYP_{hyp_depth}_{hyp_offset}_Vel_{rup_vel}_S_{seed}_Loc_{loc}.srf.npy`

| Field | Description |
|---|---|
| `mag` | Magnitude (6.0 or 7.0) |
| `hyp_depth` | Hypocenter depth (km) |
| `hyp_offset` | Hypocenter along-fault offset (km) |
| `rup_vel` | Rupture velocity (fraction of shear-wave speed) |
| `seed` | Stochastic slip realisation seed |
| `loc` | Location index |

**Filename convention (point source):**
`x={x}y={y}d={depth}__Dset_h250m_1Hz.npy`  
Hypocenter at grid position `(x, y)` metres with depth `d` metres.

### Metadata (`seismic_rom_metadata.tar.gz`)

| File | Description |
|---|---|
| `M7_rupture_realization_summary.csv` | Rupture parameters for all Mw7 training events |
| `M7_Testing_rupture_realization_summary.csv` | Rupture parameters for Mw7 test events |
| `M6_rupture_realization_summary.csv` | Rupture parameters for all Mw6 training events |
| `M6_Testing_rupture_realization_summary.csv` | Rupture parameters for Mw6 test events |
| `point_coordinates.h5` | (x, y, z) coordinates of all 62,757 grid points in metres (fault-aligned coordinate system) |
| `balanced_subset_100_per_class_manifest.csv` | Event file list for the 100-per-class balanced subset used in all experiments |
| `balanced_subset_30_per_class_manifest.csv` | Event file list for the smaller 30-per-class subset |
| `ncsn_sfba_stations.csv` | NCSN Northern California Seismic Network station list (SFBA region) with lat/lon |
| `ncsn_sfba_fdsn_station_query.txt` | FDSN query used to retrieve station metadata |
| `ncsn_station_mapping.csv` | Mapping from NCSN station names to nearest simulation grid indices |
| `subset_singular_values.npy` | POD singular values for the 100-event Mw7 training set (shape: `(50,)`) |
| `subset_cumulative_energy.npy` | Cumulative fractional energy as a function of POD rank |

**Loading station coordinates:**
```python
import h5py
import numpy as np
with h5py.File("point_coordinates.h5", "r") as f:
    coords = np.array(f["coords"])   # shape (62757, 3), units: metres
    # coords[:, 0] = along-fault (x), coords[:, 1] = across-fault (y), coords[:, 2] = depth
```

### Results (`seismic_rom_results.tar.gz`)

All CSV tables and PDF figures that appear in the paper. File names match the `\label{}` references in the manuscript. The full set of 70 tables documents every experiment sweep (rank, station count, regularisation, noise level, layout type, magnitude).

### Code (`seismic_rom_code.tar.gz`)

Core Python analysis modules in `src/`. Includes:

| Module | Description |
|---|---|
| `balanced_pod_analysis.py` | POD rank analysis and energy decomposition |
| `gappy_pod_reconstruction.py` | Gappy POD coefficient solve and reconstruction |
| `train_lowfreq_opinf.py` | POD basis assembly, data loading, event splits |
| `placement_benchmark.py` | Station-layout benchmark (random, farthest, NCSN, POD-QR) |
| `station_count_sweep.py` | Reconstruction error vs station count |
| `station_count_observability.py` | Observability metrics (σ_min, cond) vs layout |
| `rank_station_scaling.py` | Joint rank–station scaling analysis |
| `ncsn_augmentation.py` | NCSN network augmentation via POD-QR |
| `ncsn_lambda_sweep.py` | Tikhonov λ sweep for NCSN geometry |
| `ncsn_selection_bias.py` | NCSN 128-station selection-bias control |
| `ncsn_offset_sensitivity.py` | NCSN station mapping offset analysis |
| `gappy_pod_distance_bins.py` | Distance-binned reconstruction error |
| `filter_sensitivity.py` | Band-edge and record-length sensitivity |
| `noise_robustness_gappy_pod.py` | Noise and timing-jitter robustness |
| `tiled_pod_reconstruction.py` | Tiled (domain-partitioned) POD |
| `gp_pgv_baseline.py` | Gaussian process kriging (Matérn, UK-GMPE) |
| `gp_pgv_baseline_v2.py` | Loth & Baker two-component kernel kriging |
| `universal_kriging_pgv.py` | Universal kriging with GMPE mean |
| `quadratic_opinf.py` | Operator Inference (autonomous ROM) |
| `sweep_forced_opinf.py` | Forced (observation-constrained) OpInf sweep |
| `train_forced_opinf.py` / `train_windowed_forced_opinf.py` | OpInf training variants |
| `station_count_sweep_components.py` | Component-count (3-comp vs 2-comp) sweep |
| `calibration_bias_robustness.py` | Calibration bias and amplitude robustness |
| `benchmark_costs.py` | Computational cost benchmarking |
| `map_ncsn_stations.py` | NCSN lat/lon → simulation grid mapping |
| `preprocess_events.py` | Raw simulation → processed `.npy` arrays |
| `pod_feasibility.py` | POD rank feasibility checks |
| `config.py` | Path and parameter configuration |

Figure-generation scripts (`plot_*.py`) are not included in this deposit; they depend on precomputed result arrays available in `seismic_rom_results.tar.gz` and are provided in the project's GitHub repository.

---

## Simulation Domain

| Parameter | Value |
|---|---|
| Region | San Francisco Bay Area (SFBA) |
| Domain | ~90 km × 45 km (fault-aligned) |
| Grid spacing | 250 m |
| Grid points | 62,757 surface stations |
| Simulation engine | Described in Shi et al. (2026) |
| Frequency range | 0–1 Hz (raw); analyses use 0–0.25 Hz low-frequency band |
| Duration | ~30 s |

---

## Coordinate System

All spatial coordinates use a fault-aligned Cartesian system:

- **x** — along-fault direction (East-ish), origin at fault trace
- **y** — across-fault direction (North-ish)
- Units: km (in analysis code), m (in `point_coordinates.h5`)

The NCSN station mapping (`ncsn_station_mapping.csv`) provides the conversion from geographic (lat/lon) to grid index.

---

## Reproducing the Paper Results

1. Download and extract all archives into the project directory structure:
   ```
   data/processed/events/mw7/          ← from seismic_rom_data_mw7.tar.gz
   data/processed/events/mw6/          ← from seismic_rom_data_mw6.tar.gz
   data/processed/events/point_source/ ← from seismic_rom_data_point_source.tar.gz
   data/processed/                     ← from seismic_rom_metadata.tar.gz
   results/                            ← from seismic_rom_results.tar.gz
   src/                                ← from seismic_rom_code.tar.gz
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run any analysis module, for example to reproduce the main station-count sweep:
   ```bash
   python -m src.station_count_sweep        # station-count vs error table
   python -m src.placement_benchmark        # layout comparison (Table 4)
   python -m src.station_count_observability # observability metrics (Table 5)
   python -m src.ncsn_augmentation          # NCSN augmentation result
   python -m src.gp_pgv_baseline_v2         # Loth & Baker kriging comparison
   ```

Results are written as CSV tables to `results/tables/`. The pre-computed versions are available in `seismic_rom_results.tar.gz`.

---

## License

Data and code are released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.  
You are free to share and adapt the material for any purpose, provided appropriate credit is given.

Simulation data is derived from the GMFlow dataset (Shi et al., 2026). Please also cite the GMFlow paper if you use the waveform data.

---

## Citation

If you use this dataset or code, please cite:

```
Avzalshoev, Z., & Chun, P. (2026). Sparse Observability Limits of Low-Frequency Regional Seismic Ground-Motion Fields. Zenodo. https://doi.org/10.5281/zenodo.20710501
```

```
Shi, Y., Lavrentiadis, G., Tsalouchidis, K., Ross, Z. E., McCallen, D., Zou, C.,
Azizzadenesheli, K., & Asimaki, D. (2026). Large-Scale 3D Ground-Motion Synthesis
with Physics-Inspired Latent Operator Flow Matching. arXiv:2603.17403.
```

---

## Dependencies

```
numpy
scipy
scikit-learn
h5py
matplotlib
pandas
tqdm
```

Python 3.9+ recommended.
