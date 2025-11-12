#!/usr/bin/env python3
"""
mc_sampler.py

Generates Monte Carlo samples of incoming cosmic dust at top-of-atmosphere.

Outputs:
  - data/mc_results.csv
  - outputs/diagnostics_radii.png, diagnostics_ventry.png, diagnostics_latlon.png
  - outputs/run_metadata.json

Schema (CSV columns): sim_id, source, r_m, mass_kg, rho_kg_m3,
  v_inf_m_s, v_entry_m_s, vx_entry_m_s, vy_entry_m_s, vz_entry_m_s,
  lat_entry_deg, lon_entry_deg, alt_entry_m, b_m, weight_rel, timestamp_utc, em_flag, notes
"""

import os, json, math, datetime
import numpy as np, pandas as pd

# ------ Configurable parameters ------
OUT_DIR = "outputs"
DATA_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

N_SAMPLES = 100000   # change to 500k/1M for production if you have resources

R_E = 6371e3
H0 = 100e3
R_TOP = R_E + H0
G = 6.67430e-11
M_E = 5.972e24
V_ESC = math.sqrt(2*G*M_E / R_TOP)

# size distribution
R_MIN = 0.1e-6   # 0.1 micron
R_MAX = 1e-3     # 1 mm
Q = 3.5

# source types and priors
SOURCES = ["asteroid", "comet", "interstellar"]
SRC_FRACS = np.array([0.6, 0.35, 0.05])

# truncated normal v_inf params: (mean_m_s, sigma_m_s, lower_m_s)
V_PARAMS = {
    "asteroid": (11e3, 4e3, 5e3),
    "comet": (30e3, 15e3, 10e3),
    "interstellar": (50e3, 20e3, 20e3)
}

INCL_SIGMA_DEG = {"asteroid": 10.0, "comet": 30.0, "interstellar": 90.0}

# densities and material mixes
DENSITIES = {"silicate": 3000.0, "carbonaceous": 1500.0, "iron": 7800.0}
MIX = {
    "asteroid": (["silicate", "iron"], [0.8, 0.2]),
    "comet": (["carbonaceous", "silicate"], [0.85, 0.15]),
    "interstellar": (["carbonaceous", "silicate"], [0.6, 0.4])
}

EM_THRESHOLD = 0.5e-6  # m
SEED = 42
np.random.seed(SEED)

# ------ Helper samplers ------
def sample_power_law(rmin, rmax, q, size):
    exponent = 1.0 - q
    c1 = rmin ** exponent
    c2 = rmax ** exponent
    u = np.random.rand(size)
    return (c1 + (c2 - c1) * u) ** (1.0 / exponent)

def sample_trunc_normal(mean, sigma, lower, size):
    out = np.random.normal(loc=mean, scale=sigma, size=size)
    mask = out < lower
    while mask.any():
        out[mask] = np.random.normal(loc=mean, scale=sigma, size=mask.sum())
        mask = out < lower
    return out

def direction_from_inclination_azimuth(incl_rad, az_rad):
    # incl_rad is inclination from z-axis (polar angle), az_rad is azimuth
    # For spherical coordinates: x = sin(incl) * cos(az), y = sin(incl) * sin(az), z = cos(incl)
    sini = math.sin(incl_rad); cosi = math.cos(incl_rad)
    x = sini * math.cos(az_rad); y = sini * math.sin(az_rad); z = cosi
    return np.array([x, y, z])

def sample_directions(incl_sigma_deg, nsamples):
    sigma_rad = math.radians(incl_sigma_deg)
    # Sample inclination from z-axis (polar angle), clip to [0, pi]
    incls = np.abs(np.random.normal(loc=math.pi/2.0, scale=sigma_rad, size=nsamples))
    incls = np.clip(incls, 0.0, math.pi)
    azs = np.random.uniform(0.0, 2*math.pi, size=nsamples)
    vecs = np.empty((nsamples, 3))
    for i, (inc, az) in enumerate(zip(incls, azs)):
        vecs[i] = direction_from_inclination_azimuth(inc, az)
    vecs /= np.linalg.norm(vecs, axis=1)[:, None]
    return vecs

def sample_perp_unit_vectors(uvecs):
    n = uvecs.shape[0]
    arb = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    close_to_z = np.abs(uvecs[:, 2]) > 0.999
    arb[close_to_z] = np.array([0.0, 1.0, 0.0])
    perp1 = np.cross(uvecs, arb)
    perp1_norm = np.linalg.norm(perp1, axis=1)
    mask = perp1_norm < 1e-12
    if mask.any():
        arb2 = np.tile(np.array([0.0, 1.0, 0.0]), (n, 1))
        perp1[mask] = np.cross(uvecs[mask], arb2[mask])
        perp1_norm = np.linalg.norm(perp1, axis=1)
    perp1 /= perp1_norm[:, None]
    perp2 = np.cross(uvecs, perp1)
    perp2 /= np.linalg.norm(perp2, axis=1)[:, None]
    thetas = np.random.uniform(0.0, 2*math.pi, size=n)
    cos_t = np.cos(thetas)[:, None]; sin_t = np.sin(thetas)[:, None]
    return perp1 * cos_t + perp2 * sin_t

# ------ Main sampling ------
def run_mc(n_samples=N_SAMPLES):
    # sample source family
    src_choices = np.random.choice(SOURCES, size=n_samples, p=SRC_FRACS)
    
    # radius
    r = sample_power_law(R_MIN, R_MAX, Q, n_samples)
    
    # sample material/density
    materials = np.empty(n_samples, dtype=object); densities = np.empty(n_samples)
    for i, s in enumerate(src_choices):
        opts, probs = MIX[s]
        mat = np.random.choice(opts, p=probs)
        materials[i] = mat; densities[i] = DENSITIES[mat]
    
    mass = 4.0/3.0 * math.pi * r**3 * densities
    
    # v_inf
    v_inf = np.empty(n_samples)
    for s in SOURCES:
        mask = (src_choices == s)
        if mask.any():
            mean, sigma, lower = V_PARAMS[s]
            v_inf[mask] = sample_trunc_normal(mean, sigma, lower, mask.sum())
    
    # directions
    dirs = np.empty((n_samples, 3))
    for s in SOURCES:
        mask = (src_choices == s)
        if mask.any():
            dirs[mask] = sample_directions(INCL_SIGMA_DEG[s], mask.sum())
    
    u = -dirs  # unit vectors pointing toward Earth
    
    # compute bmax per sample
    bmax = R_TOP * np.sqrt(1.0 + (V_ESC**2) / (v_inf**2))
    
    U = np.random.rand(n_samples)
    b_mag = bmax * np.sqrt(U)
    
    perp_units = sample_perp_unit_vectors(u)
    b_vecs = perp_units * b_mag[:, None]
    
    # intersection distance
    s_dist = np.sqrt(np.maximum(0.0, R_TOP**2 - b_mag**2))
    r_points = -u * s_dist[:, None] + b_vecs
    
    x = r_points[:, 0]; y = r_points[:, 1]; z = r_points[:, 2]
    # Clip z/R_TOP to [-1, 1] to avoid numerical issues with arcsin
    z_norm = np.clip(z / R_TOP, -1.0, 1.0)
    lons = np.degrees(np.arctan2(y, x)); lats = np.degrees(np.arcsin(z_norm))
    
    v_entry = np.sqrt(v_inf**2 + V_ESC**2)
    v_entry_vec = -u * v_entry[:, None]
    
    # output dataframe
    sim_ids = np.arange(1, n_samples + 1, dtype=int)
    
    df = pd.DataFrame({
        "sim_id": sim_ids,
        "source": src_choices,
        "r_m": r,
        "mass_kg": mass,
        "rho_kg_m3": densities,
        "v_inf_m_s": v_inf,
        "v_entry_m_s": v_entry,
        "vx_entry_m_s": v_entry_vec[:, 0],
        "vy_entry_m_s": v_entry_vec[:, 1],
        "vz_entry_m_s": v_entry_vec[:, 2],
        "lat_entry_deg": lats,
        "lon_entry_deg": lons,
        "alt_entry_m": np.full(n_samples, H0),
        "b_m": b_mag,
        "weight_rel": np.ones(n_samples) / float(n_samples),
        "timestamp_utc": [datetime.datetime.utcnow().isoformat()]*n_samples,
        "em_flag": r < EM_THRESHOLD,
        "notes": ["" for _ in range(n_samples)]
    })
    
    return df

if __name__ == "__main__":
    print("Running MC sampler...")
    df = run_mc()
    
    out_csv = os.path.join(DATA_DIR, "mc_results.csv")
    df.to_csv(out_csv, index=False)
    
    # diagnostics
    import matplotlib.pyplot as plt
    
    # radii histogram
    plt.figure(figsize=(6,4))
    bins = np.logspace(math.log10(R_MIN), math.log10(R_MAX), 50)
    plt.hist(df['r_m'].values, bins=bins)
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel("radius (m)"); plt.title("Radius distribution (MC samples)")
    plt.grid(True, which='both', ls=':')
    plt.savefig(os.path.join(OUT_DIR, "diagnostics_radii.png"))
    plt.close()
    
    # v_entry histogram
    plt.figure(figsize=(6,4))
    plt.hist(df['v_entry_m_s'].values/1000.0, bins=60)
    plt.xlabel("v_entry (km/s)"); plt.title("Entry speed distribution")
    plt.grid(True, ls=':')
    plt.savefig(os.path.join(OUT_DIR, "diagnostics_ventry.png"))
    plt.close()
    
    # latlon hexbin
    plt.figure(figsize=(8,6))
    plt.hexbin(df['lon_entry_deg'].values, df['lat_entry_deg'].values, gridsize=120)
    plt.xlabel("Longitude (deg)"); plt.ylabel("Latitude (deg)")
    plt.title("Impact location density")
    plt.colorbar(label='counts')
    plt.savefig(os.path.join(OUT_DIR, "diagnostics_latlon.png"))
    plt.close()
    
    # metadata
    meta = {
        "N_samples": int(len(df)),
        "R_min_m": R_MIN, "R_max_m": R_MAX, "q": Q,
        "source_fracs": SRC_FRACS.tolist(), "V_params": V_PARAMS, "h0_m": H0,
        "run_time_utc": datetime.datetime.utcnow().isoformat(), "seed": SEED
    }
    with open(os.path.join(OUT_DIR, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Wrote MC output to {out_csv} and diagnostics to {OUT_DIR}")

