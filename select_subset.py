#!/usr/bin/env python3
"""
select_subset.py

Create a stratified subset from data/mc_results.csv to hand to atmosphere team.

Stratifies by (size bin x v_entry bin x zenith-angle-bin).

Outputs: data/subset_for_atmosphere.csv
"""

import os, math
import numpy as np, pandas as pd

DATA_DIR = "data"
OUT_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

MC_CSV = os.path.join(DATA_DIR, "mc_results.csv")
OUT_CSV = os.path.join(OUT_DIR, "subset_for_atmosphere.csv")

# parameters
N_TARGET = 20000   # number of samples in subset; adjust per compute budget
SIZE_BINS = np.logspace(math.log10(1e-7), math.log10(1e-3), 8)  # 7 bins
V_BINS = np.array([0, 11e3, 20e3, 35e3, 60e3, 200e3])  # m/s
ANGLE_BINS = np.array([0, 15, 30, 45, 60, 75, 90])  # degrees

def compute_entry_angle(row):
    # compute angle between entry vector and local normal. normal ≈ position vector direction
    # use lat/lon to compute normal quickly
    lat = math.radians(row['lat_entry_deg'])
    lon = math.radians(row['lon_entry_deg'])
    nx = math.cos(lat) * math.cos(lon)
    ny = math.cos(lat) * math.sin(lon)
    nz = math.sin(lat)
    
    vx = row['vx_entry_m_s']; vy = row['vy_entry_m_s']; vz = row['vz_entry_m_s']
    vmag = math.sqrt(vx*vx + vy*vy + vz*vz)
    if vmag == 0: return 90.0
    
    dot = -(vx*nx + vy*ny + vz*nz) / vmag
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))

if __name__ == "__main__":
    df = pd.read_csv(MC_CSV)
    
    # compute derived columns if needed
    if 'entry_angle_deg' not in df.columns:
        df['entry_angle_deg'] = df.apply(lambda r: compute_entry_angle(r), axis=1)
    
    # assign bins
    df['size_bin'] = pd.cut(df['r_m'], bins=SIZE_BINS, labels=False, include_lowest=True)
    df['v_bin'] = pd.cut(df['v_entry_m_s'], bins=V_BINS, labels=False, include_lowest=True)
    df['angle_bin'] = pd.cut(df['entry_angle_deg'], bins=ANGLE_BINS, labels=False, include_lowest=True)
    
    # drop rows with NaN bins (outside range)
    df = df.dropna(subset=['size_bin', 'v_bin', 'angle_bin'])
    
    # stratified sample: equal number per occupied cell
    groups = df.groupby(['size_bin', 'v_bin', 'angle_bin'])
    cells = [g for _, g in groups]
    n_cells = len(cells)
    per_cell = max(1, N_TARGET // n_cells)
    
    picked = []
    for g in cells:
        if len(g) <= per_cell:
            picked.append(g)
        else:
            picked.append(g.sample(per_cell, random_state=42))
    
    subset = pd.concat(picked, ignore_index=True)
    
    # if we overshot, downsample
    if len(subset) > N_TARGET:
        subset = subset.sample(N_TARGET, random_state=42)
    
    subset.to_csv(OUT_CSV, index=False)
    print(f"Wrote subset with {len(subset)} rows to {OUT_CSV}")

