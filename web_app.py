#!/usr/bin/env python3
"""
Web application for Particle Entry and Matching pipeline.
Provides web interface for MC sampling, subset selection, and matching/inference.
"""

from flask import Flask, render_template, jsonify, request, send_file
from flask_cors import CORS
import os
import json
import traceback
import tempfile
import subprocess
import sys

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

DATA_DIR = "data"
OUT_DIR = "outputs"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/api/run_mc', methods=['POST'])
def run_mc():
    """Run MC sampler with custom parameters."""
    try:
        data = request.json
        n_samples = int(data.get('n_samples', 100000))
        seed = data.get('seed', 42)
        
        # Import and run MC sampler
        import mc_sampler as sampler
        import numpy as np
        
        # Set seed
        np.random.seed(seed)
        sampler.SEED = seed
        sampler.np.random.seed(seed)
        
        # Temporarily modify N_SAMPLES
        original_n = sampler.N_SAMPLES
        sampler.N_SAMPLES = n_samples
        
        # Run simulation
        df = sampler.run_mc(n_samples=n_samples)
        
        # Restore original
        sampler.N_SAMPLES = original_n
        
        # Save results
        out_csv = os.path.join(DATA_DIR, "mc_results.csv")
        df.to_csv(out_csv, index=False)
        
        # Get diagnostics
        diagnostics = {
            'total_particles': len(df),
            'source_distribution': df['source'].value_counts().to_dict(),
            'size_stats': {
                'min': float(df['r_m'].min()),
                'max': float(df['r_m'].max()),
                'mean': float(df['r_m'].mean()),
                'median': float(df['r_m'].median())
            },
            'velocity_stats': {
                'min': float(df['v_entry_m_s'].min()),
                'max': float(df['v_entry_m_s'].max()),
                'mean': float(df['v_entry_m_s'].mean()),
                'median': float(df['v_entry_m_s'].median())
            }
        }
        
        return jsonify({
            'success': True,
            'message': f'Generated {len(df)} particles',
            'diagnostics': diagnostics,
            'output_file': out_csv
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/run_subset', methods=['POST'])
def run_subset():
    """Run subset selection."""
    try:
        data = request.json
        n_target = int(data.get('n_target', 20000))
        
        # Check if MC results exist
        mc_csv = os.path.join(DATA_DIR, "mc_results.csv")
        if not os.path.exists(mc_csv):
            return jsonify({
                'success': False,
                'error': 'MC results not found. Please run MC sampler first.'
            }), 400
        
        # Import and run subset selection
        import select_subset
        
        # Temporarily modify N_TARGET
        original_target = select_subset.N_TARGET
        select_subset.N_TARGET = n_target
        
        # Run subset selection
        result = subprocess.run(
            [sys.executable, 'select_subset.py'],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        # Restore original
        select_subset.N_TARGET = original_target
        
        if result.returncode != 0:
            return jsonify({
                'success': False,
                'error': result.stderr
            }), 500
        
        # Check output
        subset_csv = os.path.join(DATA_DIR, "subset_for_atmosphere.csv")
        if os.path.exists(subset_csv):
            import pandas as pd
            df = pd.read_csv(subset_csv)
            return jsonify({
                'success': True,
                'message': f'Created subset with {len(df)} particles',
                'output_file': subset_csv,
                'count': len(df)
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Subset file not created'
            }), 500
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/run_demo', methods=['POST'])
def run_demo():
    """Run demo pipeline."""
    try:
        result = subprocess.run(
            [sys.executable, 'demo_pipeline.py'],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        if result.returncode != 0:
            return jsonify({
                'success': False,
                'error': result.stderr,
                'stdout': result.stdout
            }), 500
        
        # Check for outputs
        outputs = {}
        if os.path.exists(os.path.join(OUT_DIR, 'posterior_histograms.csv')):
            outputs['posterior_histograms'] = True
        if os.path.exists(os.path.join(OUT_DIR, 'posterior_plot.png')):
            outputs['posterior_plot'] = True
        
        return jsonify({
            'success': True,
            'message': 'Demo pipeline completed',
            'outputs': outputs,
            'stdout': result.stdout
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/run_match', methods=['POST'])
def run_match():
    """Run match and infer."""
    try:
        # Check required files
        atm_csv = os.path.join(DATA_DIR, "atmosphere_results.csv")
        strata_csv = os.path.join(DATA_DIR, "strata.csv")
        mc_csv = os.path.join(DATA_DIR, "mc_results.csv")
        
        missing = []
        if not os.path.exists(atm_csv):
            missing.append('atmosphere_results.csv')
        if not os.path.exists(strata_csv):
            missing.append('strata.csv')
        if not os.path.exists(mc_csv):
            missing.append('mc_results.csv')
        
        if missing:
            return jsonify({
                'success': False,
                'error': f'Missing required files: {", ".join(missing)}'
            }), 400
        
        result = subprocess.run(
            [sys.executable, 'match_and_infer.py'],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        if result.returncode != 0:
            return jsonify({
                'success': False,
                'error': result.stderr,
                'stdout': result.stdout
            }), 500
        
        # Check for outputs
        outputs = {}
        if os.path.exists(os.path.join(OUT_DIR, 'posterior_histograms.csv')):
            outputs['posterior_histograms'] = True
        if os.path.exists(os.path.join(OUT_DIR, 'posterior_plot.png')):
            outputs['posterior_plot'] = True
        
        return jsonify({
            'success': True,
            'message': 'Matching and inference completed',
            'outputs': outputs,
            'stdout': result.stdout
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/download/<filename>')
def download_file(filename):
    """Download output files."""
    safe_files = {
        'mc_results': os.path.join(DATA_DIR, 'mc_results.csv'),
        'subset': os.path.join(DATA_DIR, 'subset_for_atmosphere.csv'),
        'posterior': os.path.join(OUT_DIR, 'posterior_histograms.csv'),
        'atmosphere_results': os.path.join(DATA_DIR, 'atmosphere_results.csv'),
        'strata': os.path.join(DATA_DIR, 'strata.csv')
    }
    
    if filename not in safe_files:
        return jsonify({'error': 'Invalid file'}), 400
    
    filepath = safe_files[filename]
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(filepath, as_attachment=True)

@app.route('/api/list_outputs')
def list_outputs():
    """List available output files."""
    outputs = {
        'data_files': [],
        'output_files': []
    }
    
    # Check data directory
    for f in ['mc_results.csv', 'subset_for_atmosphere.csv', 
              'atmosphere_results.csv', 'strata.csv']:
        if os.path.exists(os.path.join(DATA_DIR, f)):
            outputs['data_files'].append(f)
    
    # Check outputs directory
    if os.path.exists(OUT_DIR):
        for f in os.listdir(OUT_DIR):
            if f.endswith(('.csv', '.png', '.json')):
                outputs['output_files'].append(f)
    
    return jsonify(outputs)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8081))
    app.run(debug=True, host='0.0.0.0', port=port)

