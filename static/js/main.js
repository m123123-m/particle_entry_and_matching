// Main JavaScript for Particle Entry and Matching Pipeline

// Tab switching
document.querySelectorAll('.tab-button').forEach(button => {
    button.addEventListener('click', () => {
        const tabName = button.dataset.tab;
        
        document.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));
        button.classList.add('active');
        
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        document.getElementById(`${tabName}Tab`).classList.add('active');
    });
});

// Status management
function showStatus(message, progress = 0) {
    const panel = document.getElementById('statusPanel');
    const text = document.getElementById('statusText');
    const bar = document.getElementById('progressBar');
    
    panel.style.display = 'block';
    text.textContent = message;
    bar.style.width = `${progress}%`;
}

function hideStatus() {
    document.getElementById('statusPanel').style.display = 'none';
}

// Run MC Sampler
document.getElementById('runMC').addEventListener('click', async () => {
    const button = document.getElementById('runMC');
    const n_samples = parseInt(document.getElementById('n_samples').value);
    const seed = parseInt(document.getElementById('seed').value);
    
    button.disabled = true;
    showStatus('Running MC sampler...', 30);
    
    try {
        const response = await fetch('/api/run_mc', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ n_samples, seed })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showStatus(data.message, 100);
            updateResults(data.diagnostics);
            setTimeout(() => {
                hideStatus();
                loadOutputs();
            }, 2000);
        } else {
            showStatus(`Error: ${data.error}`, 0);
            alert(`Error: ${data.error}`);
        }
    } catch (error) {
        showStatus(`Error: ${error.message}`, 0);
        alert(`Error: ${error.message}`);
    } finally {
        button.disabled = false;
    }
});

// Run Subset Selection
document.getElementById('runSubset').addEventListener('click', async () => {
    const button = document.getElementById('runSubset');
    const n_target = parseInt(document.getElementById('n_target').value);
    
    button.disabled = true;
    showStatus('Creating subset...', 50);
    
    try {
        const response = await fetch('/api/run_subset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ n_target })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showStatus(data.message, 100);
            setTimeout(() => {
                hideStatus();
                loadOutputs();
            }, 2000);
        } else {
            showStatus(`Error: ${data.error}`, 0);
            alert(`Error: ${data.error}`);
        }
    } catch (error) {
        showStatus(`Error: ${error.message}`, 0);
        alert(`Error: ${error.message}`);
    } finally {
        button.disabled = false;
    }
});

// Run Demo Pipeline
document.getElementById('runDemo').addEventListener('click', async () => {
    const button = document.getElementById('runDemo');
    
    if (!confirm('This will run the full demo pipeline. Continue?')) {
        return;
    }
    
    button.disabled = true;
    showStatus('Running demo pipeline...', 20);
    
    try {
        const response = await fetch('/api/run_demo', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        
        if (data.success) {
            showStatus('Demo pipeline completed!', 100);
            setTimeout(() => {
                hideStatus();
                loadOutputs();
            }, 2000);
        } else {
            showStatus(`Error: ${data.error}`, 0);
            alert(`Error: ${data.error}\n\n${data.stdout || ''}`);
        }
    } catch (error) {
        showStatus(`Error: ${error.message}`, 0);
        alert(`Error: ${error.message}`);
    } finally {
        button.disabled = false;
    }
});

// Run Match & Infer
document.getElementById('runMatch').addEventListener('click', async () => {
    const button = document.getElementById('runMatch');
    
    if (!confirm('This requires atmosphere_results.csv and strata.csv. Continue?')) {
        return;
    }
    
    button.disabled = true;
    showStatus('Running matching and inference...', 50);
    
    try {
        const response = await fetch('/api/run_match', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        
        if (data.success) {
            showStatus('Matching completed!', 100);
            setTimeout(() => {
                hideStatus();
                loadOutputs();
            }, 2000);
        } else {
            showStatus(`Error: ${data.error}`, 0);
            alert(`Error: ${data.error}\n\n${data.stdout || ''}`);
        }
    } catch (error) {
        showStatus(`Error: ${error.message}`, 0);
        alert(`Error: ${error.message}`);
    } finally {
        button.disabled = false;
    }
});

// Update results display
function updateResults(diagnostics) {
    const container = document.getElementById('resultsContent');
    
    if (!diagnostics) {
        container.innerHTML = '<p class="placeholder">Run a simulation step to see results</p>';
        return;
    }
    
    let html = '<div class="stat-grid">';
    
    html += `
        <div class="stat-card">
            <h4>Total Particles</h4>
            <div class="value">${diagnostics.total_particles.toLocaleString()}</div>
        </div>
    `;
    
    if (diagnostics.size_stats) {
        html += `
            <div class="stat-card">
                <h4>Mean Radius</h4>
                <div class="value">${(diagnostics.size_stats.mean * 1e6).toFixed(2)} μm</div>
            </div>
        `;
    }
    
    if (diagnostics.velocity_stats) {
        html += `
            <div class="stat-card">
                <h4>Mean Entry Velocity</h4>
                <div class="value">${(diagnostics.velocity_stats.mean / 1000).toFixed(2)} km/s</div>
            </div>
        `;
    }
    
    if (diagnostics.source_distribution) {
        html += '</div><h3>Source Distribution</h3><ul>';
        for (const [source, count] of Object.entries(diagnostics.source_distribution)) {
            const percentage = (count / diagnostics.total_particles * 100).toFixed(1);
            html += `<li><strong>${source}:</strong> ${count.toLocaleString()} (${percentage}%)</li>`;
        }
        html += '</ul>';
    }
    
    container.innerHTML = html;
}

// Load outputs
async function loadOutputs() {
    try {
        const response = await fetch('/api/list_outputs');
        const data = await response.json();
        
        const container = document.getElementById('filesList');
        let html = '<div class="file-list">';
        
        if (data.data_files.length > 0) {
            html += '<h3>Data Files</h3>';
            data.data_files.forEach(file => {
                const downloadName = file.replace('.csv', '');
                html += `
                    <div class="file-item">
                        <span>${file}</span>
                        <a href="/api/download/${downloadName}" download>Download</a>
                    </div>
                `;
            });
        }
        
        if (data.output_files.length > 0) {
            html += '<h3>Output Files</h3>';
            data.output_files.forEach(file => {
                html += `
                    <div class="file-item">
                        <span>${file}</span>
                        <span>Available</span>
                    </div>
                `;
            });
        }
        
        if (data.data_files.length === 0 && data.output_files.length === 0) {
            html += '<p class="placeholder">No output files available yet</p>';
        }
        
        html += '</div>';
        container.innerHTML = html;
    } catch (error) {
        console.error('Error loading outputs:', error);
    }
}

// Refresh outputs button
document.getElementById('refreshOutputs').addEventListener('click', loadOutputs);

// Load outputs on page load
loadOutputs();

