#!/usr/bin/env python3
"""
match_and_infer.py

- Reads atmosphere results (data/atmosphere_results.csv) and strata (data/strata.csv)
- Computes spatial + histogram likelihoods to weight MC sims and produce posterior histograms

Outputs:
  - outputs/posterior_histograms.csv
  - outputs/posterior_plot.png

Assumes:
  - atmosphere_results.csv has columns: sim_id, land_lat, land_lon, surv_mass_kg, frag_hist_bins (as JSON string) or frag_hist_* columns
  - strata.csv has columns: layer_id, lat, lon, obs_hist_* (matching binning)
"""

import os, json, math
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "data"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

MC_CSV = os.path.join(DATA_DIR, "mc_results.csv")
ATM_CSV = os.path.join(DATA_DIR, "atmosphere_results.csv")  # provided by atmosphere team
STRATA_CSV = os.path.join(DATA_DIR, "strata.csv")           # observed strata

# parameters
SIGMA_X_M = 5000.0  # spatial kernel sigma (m)
HIST_VARIANCE_EPS = 1e-6

def haversine_m(lat1, lon1, lat2, lon2):
    # returns distance in meters
    R = 6371e3
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def load_frag_hist(row, prefix="frag_hist_"):
    # collect columns frag_hist_0, frag_hist_1, ... OR parse JSON in frag_hist_json
    # Check for frag_hist_json first (higher priority)
    if 'frag_hist_json' in row.index:
        frag_str = row['frag_hist_json']
        if pd.isna(frag_str) or frag_str == '':
            return None
        if isinstance(frag_str, str):
            # Try to parse as JSON
            try:
                return np.array(json.loads(frag_str), dtype=float)
            except (json.JSONDecodeError, ValueError):
                # If JSON fails, try eval as list (for safety, only if it looks like a list)
                if frag_str.strip().startswith('['):
                    try:
                        return np.array(eval(frag_str), dtype=float)
                    except:
                        return None
                return None
        else:
            # Already an array or list
            return np.array(frag_str, dtype=float)
    
    # Otherwise check for frag_hist_* columns (numeric columns only)
    keys = [c for c in row.index if c.startswith(prefix) and c != 'frag_hist_json']
    if keys:
        # Sort keys to ensure consistent order
        keys = sorted(keys)
        try:
            values = row[keys].values
            # Check if values are numeric (not strings)
            if isinstance(values[0], str):
                # If it's a string, it might be a JSON string, skip these columns
                return None
            return values.astype(float)
        except (ValueError, TypeError):
            # If conversion fails, return None
            return None
    return None

def hist_chi2(obs, sim):
    # simple chi2; obs, sim arrays same shape
    var = obs + HIST_VARIANCE_EPS  # Poisson-like variance approx
    chi2 = np.sum((obs - sim)**2 / var)
    return chi2

if __name__ == "__main__":
    # load inputs
    mc = pd.read_csv(MC_CSV)
    atm = pd.read_csv(ATM_CSV)
    strata = pd.read_csv(STRATA_CSV)
    
    # unify: atmosphere results must include sim_id linking to mc
    merged = mc.merge(atm, on='sim_id', how='inner', suffixes=('', '_atm'))
    
    if len(merged) == 0:
        raise RuntimeError("No matching sim_ids between mc_results.csv and atmosphere_results.csv. "
                         f"MC has {len(mc)} rows, atmosphere has {len(atm)} rows. "
                         f"MC sim_ids: {mc['sim_id'].min()}-{mc['sim_id'].max()}, "
                         f"Atm sim_ids: {atm['sim_id'].min()}-{atm['sim_id'].max()}")
    
    # prepare frag hist arrays for sims
    sim_hist_list = []
    for _, row in merged.iterrows():
        h = load_frag_hist(row)
        if h is None:
            # fallback: if only surv_mass_kg available, make coarse single-bin histogram
            h = np.array([row.get('surv_mass_kg', 0.0)])
        sim_hist_list.append(h)
    
    # convert to 2D array (pad to same length)
    maxlen = max(len(h) for h in sim_hist_list) if sim_hist_list else 1
    sim_hist_arr = np.array([np.pad(h, (0, maxlen-len(h))) for h in sim_hist_list])
    
    # for strata, get obs hist arrays
    strata_hist_list = []
    for _, row in strata.iterrows():
        # Check for obs_hist_json first (higher priority)
        if 'obs_hist_json' in row.index:
            obs_str = row['obs_hist_json']
            if pd.isna(obs_str) or obs_str == '':
                raise RuntimeError(f"Empty obs_hist_json for layer {row.get('layer_id', 'unknown')}")
            if isinstance(obs_str, str):
                try:
                    h = np.array(json.loads(obs_str), dtype=float)
                except (json.JSONDecodeError, ValueError):
                    try:
                        h = np.array(eval(obs_str), dtype=float)
                    except:
                        raise RuntimeError(f"Could not parse obs_hist_json for layer {row.get('layer_id', 'unknown')}")
            else:
                h = np.array(obs_str, dtype=float)
        else:
            # Check for obs_hist_* columns (excluding obs_hist_json)
            keys = [c for c in row.index if c.startswith('obs_hist_') and c != 'obs_hist_json']
            if keys:
                keys = sorted(keys)
                try:
                    values = row[keys].values
                    # Check if values are numeric (not strings)
                    if len(values) > 0 and isinstance(values[0], str):
                        raise RuntimeError(f"obs_hist_* columns contain string values for layer {row.get('layer_id', 'unknown')}")
                    h = values.astype(float)
                except (ValueError, TypeError) as e:
                    raise RuntimeError(f"Could not convert obs_hist_* columns to float for layer {row.get('layer_id', 'unknown')}: {e}")
            else:
                raise RuntimeError(f"Strata must include obs_hist_* columns or obs_hist_json for layer {row.get('layer_id', 'unknown')}")
        
        # pad or truncate to maxlen
        if len(h) < maxlen:
            h = np.pad(h, (0, maxlen-len(h)))
        else:
            h = h[:maxlen]
        strata_hist_list.append(h)
    
    strata_hist_arr = np.vstack(strata_hist_list)
    
    # compute spatial distances matrix between sims and strata (meters)
    sim_lats = merged['land_lat'].values
    sim_lons = merged['land_lon'].values
    strata_lats = strata['lat'].values
    strata_lons = strata['lon'].values
    
    # pairwise distances
    D = np.zeros((len(merged), len(strata)))
    for j in range(len(strata)):
        D[:, j] = haversine_m(sim_lats, sim_lons, strata_lats[j], strata_lons[j])
    
    # compute spatial kernel p_space = exp(-0.5*(d/sigma)^2)
    P_space = np.exp(-0.5 * (D / SIGMA_X_M)**2)
    
    # compute hist likelihoods p_hist = exp(-0.5 * chi2)
    P_hist = np.zeros_like(P_space)
    for j in range(len(strata)):
        obs = strata_hist_arr[j]
        # compute chi2 for every sim
        for i in range(len(merged)):
            simh = sim_hist_arr[i]
            chi2 = hist_chi2(obs, simh)
            P_hist[i, j] = math.exp(-0.5 * chi2)
    
    # combine to get per-sim per-strata likelihood
    # prior weight per sim is weight_rel column from mc
    prior_w = merged['weight_rel'].values
    
    L = P_space * P_hist
    post = np.zeros_like(L)
    
    # compute posterior weights for each strata (normalize over sims)
    for j in range(L.shape[1]):
        scores = prior_w * L[:, j]
        ssum = scores.sum()
        if ssum <= 0:
            post[:, j] = 0
        else:
            post[:, j] = scores / ssum
    
    # aggregate posterior by radius bins to estimate incoming distribution per strata
    radius_bins = np.logspace(math.log10(1e-7), math.log10(1e-3), 9)  # 8 bins
    bin_idx = np.digitize(merged['r_m'].values, radius_bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(radius_bins)-2)  # ensure valid bin indices
    
    results = []
    for j in range(L.shape[1]):
        for b in range(len(radius_bins)-1):
            mask = (bin_idx == b)
            est = post[mask, j].sum()
            results.append({'layer_id': strata.iloc[j]['layer_id'], 'bin': b,
                            'r_lo': radius_bins[b], 'r_hi': radius_bins[b+1], 'posterior_prob': est})
    
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUT_DIR, "posterior_histograms.csv"), index=False)
    print("Wrote posterior_histograms.csv")
    
    # quick plotting: sum posterior across strata to get global posterior
    if len(res_df) > 0:
        # Group by bin and sum probabilities, keeping radius bin info
        global_post = res_df.groupby('bin').agg({
            'posterior_prob': 'sum',
            'r_lo': 'first',  # Keep first r_lo for each bin
            'r_hi': 'first'   # Keep first r_hi for each bin
        }).reset_index()
        
        mids = (global_post['r_lo'] + global_post['r_hi']) / 2.0
        plt.figure(figsize=(8,6))
        plt.bar(np.log10(mids), global_post['posterior_prob'], width=0.1)
        plt.xlabel("log10 radius (m)"); plt.ylabel("posterior probability (relative)")
        plt.title("Estimated incoming distribution (aggregated across strata)")
        plt.grid(True, ls=':')
        plt.savefig(os.path.join(OUT_DIR, "posterior_plot.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved posterior plot to outputs/")

