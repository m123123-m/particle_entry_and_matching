#!/usr/bin/env python3
"""
demo_pipeline.py

Simple demo that:
- runs mc_sampler (small N),
- synthesizes a fake atmosphere_results.csv by mapping sim -> small perturbation of entry point,
- creates a fake strata.csv from a few landing points,
- runs match_and_infer logic to produce a posterior.

Use to validate the pipeline without real atmosphere team output.
"""

import os
import pandas as pd, numpy as np
from math import radians
import json

# reuse mc_sampler.run_mc by importing if in same folder
import mc_sampler as sampler

DATA_DIR = "data"
OUT_DIR = "outputs"
os.makedirs(DATA_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)

def synthesize_atmosphere_results(mc_df, n_frac=0.2):
    # pick fraction of sims to "survive" and scatter landing coords by small offset
    surv = mc_df.sample(frac=n_frac, random_state=1).copy()
    
    # small dispersal: add 0.01 deg noise (~1 km)
    surv['land_lat'] = surv['lat_entry_deg'] + np.random.normal(scale=0.01, size=len(surv))
    surv['land_lon'] = surv['lon_entry_deg'] + np.random.normal(scale=0.01, size=len(surv))
    surv['surv_mass_kg'] = surv['mass_kg'] * np.random.uniform(0.01, 0.8, size=len(surv))
    
    # build small frag_hist arrays and store as JSON string
    frag_hists = []
    for m in surv['surv_mass_kg'].values:
        # three-bin histogram sim: split mass into random bins
        parts = np.random.dirichlet([1,1,1]) * m
        frag_hists.append(json.dumps(parts.tolist()))
    
    surv['frag_hist_json'] = frag_hists
    
    # write file
    atm_csv = os.path.join(DATA_DIR, "atmosphere_results.csv")
    surv[['sim_id', 'land_lat', 'land_lon', 'surv_mass_kg', 'frag_hist_json']].to_csv(atm_csv, index=False)
    print(f"Wrote synthetic atmosphere results to {atm_csv} with {len(surv)} particles")
    return atm_csv

def synthesize_strata(atm_csv, n_layers=3):
    atm = pd.read_csv(atm_csv)
    picked = atm.sample(min(n_layers, len(atm)), random_state=2)
    rows = []
    
    for i, (idx, r) in enumerate(picked.iterrows()):
        layer_id = f"layer_{i+1}"
        lat = r['land_lat']; lon = r['land_lon']
        
        # obs_hist approximately equal to the sim frag histogram plus noise
        frag_str = r['frag_hist_json']
        if isinstance(frag_str, str):
            try:
                frag = np.array(json.loads(frag_str), dtype=float)
            except (json.JSONDecodeError, ValueError):
                try:
                    frag = np.array(eval(frag_str), dtype=float)
                except:
                    frag = np.array([r['surv_mass_kg']])
        else:
            frag = np.array(frag_str, dtype=float)
        
        obs_hist = frag + np.random.normal(scale=frag*0.2 + 1e-12, size=len(frag))
        obs_hist = np.maximum(obs_hist, 0.0)  # ensure non-negative
        
        rows.append({
            'layer_id': layer_id,
            'lat': lat,
            'lon': lon,
            'obs_hist_json': json.dumps(obs_hist.tolist())
        })
    
    strata_df = pd.DataFrame(rows)
    strata_df.to_csv(os.path.join(DATA_DIR, "strata.csv"), index=False)
    print(f"Wrote synthetic strata to data/strata.csv with {len(strata_df)} layers")
    return strata_df

if __name__ == "__main__":
    print("=" * 60)
    print("Demo Pipeline: Cosmic Dust Entry and Matching")
    print("=" * 60)
    
    # small MC run
    print("\n1. Running MC sampler...")
    mc_df = sampler.run_mc(n_samples=5000)
    mc_df.to_csv(os.path.join(DATA_DIR, "mc_results.csv"), index=False)
    print(f"   Generated {len(mc_df)} MC samples")
    
    # synthesize atmosphere results
    print("\n2. Synthesizing atmosphere results...")
    atm_csv = synthesize_atmosphere_results(mc_df, n_frac=0.2)
    
    # synthesize strata
    print("\n3. Synthesizing strata observations...")
    synthesize_strata(atm_csv, n_layers=3)
    
    # run match and infer
    print("\n4. Running match_and_infer...")
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "match_and_infer.py"], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
        print("\n5. Pipeline complete! Check outputs/ for results.")
    else:
        print("Error running match_and_infer.py:")
        print(result.stderr)
        print(result.stdout)
    print("=" * 60)

