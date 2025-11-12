# Particle Entry and Matching

Monte Carlo simulation and Bayesian inference pipeline for cosmic dust particle entry into Earth's atmosphere.

## Overview

This pipeline implements:
1. **Monte Carlo Sampling** (`mc_sampler.py`): Generates samples of incoming cosmic dust particles
2. **Subset Selection** (`select_subset.py`): Creates stratified subsets for atmosphere team
3. **Matching and Inference** (`match_and_infer.py`): Bayesian inference from atmosphere outputs and strata observations
4. **Demo Pipeline** (`demo_pipeline.py`): End-to-end demonstration with synthetic data

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Monte Carlo Samples

```bash
python mc_sampler.py
```

This generates:
- `data/mc_results.csv`: Full MC sample dataset
- `outputs/diagnostics_radii.png`: Radius distribution
- `outputs/diagnostics_ventry.png`: Entry velocity distribution
- `outputs/diagnostics_latlon.png`: Impact location map
- `outputs/run_metadata.json`: Simulation metadata

### 2. Create Stratified Subset

```bash
python select_subset.py
```

This creates:
- `data/subset_for_atmosphere.csv`: Stratified subset for atmosphere team (20,000 particles by default)

### 3. Run Matching and Inference

After receiving atmosphere results, run:

```bash
python match_and_infer.py
```

This requires:
- `data/atmosphere_results.csv`: Results from atmosphere team (columns: sim_id, land_lat, land_lon, surv_mass_kg, frag_hist_json)
- `data/strata.csv`: Observed strata (columns: layer_id, lat, lon, obs_hist_json)

Outputs:
- `outputs/posterior_histograms.csv`: Posterior probability distributions
- `outputs/posterior_plot.png`: Visualization of posterior distributions

### 4. Run Demo Pipeline

To test the entire pipeline with synthetic data:

```bash
python demo_pipeline.py
```

This runs all steps with synthetic atmosphere results and strata observations.

## File Structure

```
particle_entry_and_matching/
├── mc_sampler.py          # Monte Carlo sampler
├── select_subset.py       # Stratified subset selection
├── match_and_infer.py     # Bayesian matching and inference
├── demo_pipeline.py       # End-to-end demo
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/                 # Data directory
│   ├── mc_results.csv
│   ├── subset_for_atmosphere.csv
│   ├── atmosphere_results.csv
│   └── strata.csv
└── outputs/              # Output directory
    ├── diagnostics_*.png
    ├── run_metadata.json
    ├── posterior_histograms.csv
    └── posterior_plot.png
```

## Data Formats

### mc_results.csv
Columns: `sim_id`, `source`, `r_m`, `mass_kg`, `rho_kg_m3`, `v_inf_m_s`, `v_entry_m_s`, `vx_entry_m_s`, `vy_entry_m_s`, `vz_entry_m_s`, `lat_entry_deg`, `lon_entry_deg`, `alt_entry_m`, `b_m`, `weight_rel`, `timestamp_utc`, `em_flag`, `notes`

### atmosphere_results.csv
Columns: `sim_id`, `land_lat`, `land_lon`, `surv_mass_kg`, `frag_hist_json`

### strata.csv
Columns: `layer_id`, `lat`, `lon`, `obs_hist_json`

## Parameters

Key parameters can be adjusted in each script:
- `N_SAMPLES`: Number of MC samples (default: 100,000)
- `N_TARGET`: Target subset size (default: 20,000)
- `SIGMA_X_M`: Spatial kernel sigma for matching (default: 5000 m)
- Size bins, velocity bins, angle bins: Configurable in `select_subset.py`

## License

Open source for research and educational purposes.

