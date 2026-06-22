/**
 * Dashboard JavaScript - Main application logic
 */

// Chart.js default options
Chart.defaults.font.family = "system-ui, -apple-system, sans-serif";
Chart.defaults.plugins.legend.position = "top";

// Global state
let dashboardData = {
    metrics: {},
    explanations: {},
    predictions: {},
};

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    loadDashboardData();
    setupTabNavigation();
    setupExplanationFilters();
});

/**
 * Load all dashboard data from API endpoints
 */
async function loadDashboardData() {
    try {
        // Fetch all data in parallel
        const [metricsRes, explanationsRes, predictionsRes] = await Promise.all([
            axios.get('/api/metrics'),
            axios.get('/api/explanations'),
            axios.get('/api/predictions'),
        ]);

        dashboardData = {
            metrics: metricsRes.data,
            explanations: explanationsRes.data,
            predictions: predictionsRes.data,
        };

        // Render dashboard
        renderOverview();
        renderCalibration();
        renderExplanations();
        renderPredictions();
        renderUncertainty();
    } catch (error) {
        console.error('Error loading dashboard data:', error);
        showError('Failed to load dashboard data. Please refresh the page.');
    }
}

/**
 * Setup tab navigation
 */
function setupTabNavigation() {
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', function() {
            const tabName = this.getAttribute('data-tab');
            showTab(tabName);
        });
    });
}

/**
 * Show specific tab
 */
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });

    // Remove active class from all buttons
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(tabName).classList.add('active');

    // Mark button as active
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
}

/**
 * Render Overview Tab
 */
function renderOverview() {
    const container = document.getElementById('overview-metrics');
    const metrics = dashboardData.metrics;

    const cards = [
        {
            title: 'Calibration Method',
            value: metrics.calibration?.calibration_method || 'N/A',
            unit: '',
        },
        {
            title: 'Validation Accuracy',
            value: (metrics.calibration?.validation_accuracy * 100)?.toFixed(2) || 'N/A',
            unit: '%',
        },
        {
            title: 'Test Accuracy',
            value: (metrics.calibration?.test_accuracy * 100)?.toFixed(2) || 'N/A',
            unit: '%',
        },
        {
            title: 'ECE (Uncertainty)',
            value: (metrics.uncertainty?.ece || 0).toFixed(4),
            unit: '',
        },
    ];

    container.innerHTML = cards.map(card => `
        <div class="metric-card">
            <h3>${card.title}</h3>
            <div><span class="value">${card.value}</span><span class="unit">${card.unit}</span></div>
        </div>
    `).join('');
}

/**
 * Render Calibration Tab
 */
function renderCalibration() {
    const metrics = dashboardData.metrics;
    const container = document.getElementById('calibration-metrics');

    const calibration = metrics.calibration || {};
    const cards = [
        {
            title: 'Best Threshold',
            value: (calibration.best_threshold || 0.5).toFixed(3),
            unit: '',
        },
        {
            title: 'Validation Accuracy',
            value: (calibration.validation_accuracy * 100)?.toFixed(2) || 'N/A',
            unit: '%',
        },
        {
            title: 'Test Accuracy',
            value: (calibration.test_accuracy * 100)?.toFixed(2) || 'N/A',
            unit: '%',
        },
        {
            title: 'Sensitivity (Internal)',
            value: (calibration.internal_test_sensitivity * 100)?.toFixed(2) || 'N/A',
            unit: '%',
        },
    ];

    container.innerHTML = cards.map(card => `
        <div class="metric-card">
            <h3>${card.title}</h3>
            <div><span class="value">${card.value}</span><span class="unit">${card.unit}</span></div>
        </div>
    `).join('');
}

/**
 * Render Explanations Tab
 */
function renderExplanations() {
    const explanations = dashboardData.explanations;

    // Render global feature importance
    renderGlobalImportance(explanations.global || []);

    // Setup local explanations
    renderLocalExplanations(explanations.local || []);
}

/**
 * Render global feature importance
 */
function renderGlobalImportance(data) {
    const tableBody = document.querySelector('#global-importance-table tbody');
    
    if (!data || data.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="3">No data available</td></tr>';
        return;
    }

    tableBody.innerHTML = data.slice(0, 20).map(row => `
        <tr>
            <td>${row.feature}</td>
            <td>${(row.importance).toFixed(4)}</td>
            <td>${(row.std).toFixed(4)}</td>
        </tr>
    `).join('');

    // Create chart
    createGlobalImportanceChart(data.slice(0, 10));
}

/**
 * Create global importance chart
 */
function createGlobalImportanceChart(data) {
    const ctx = document.getElementById('global-importance-chart');
    
    if (ctx.chart instanceof Chart) {
        ctx.chart.destroy();
    }

    ctx.chart = new Chart(ctx, {
        type: 'barHorizontal',
        data: {
            labels: data.map(d => d.feature),
            datasets: [{
                label: 'Global Importance',
                data: data.map(d => d.importance),
                backgroundColor: '#3498db',
                borderColor: '#2c3e50',
                borderWidth: 1,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
            },
            scales: {
                x: { beginAtZero: true },
            },
        },
    });
}

/**
 * Render local explanations
 */
function renderLocalExplanations(data) {
    const selector = document.getElementById('sample-id-select');
    const uniqueSamples = [...new Set(data.map(d => d.sample_id))];

    selector.innerHTML = uniqueSamples.map(id => 
        `<option value="${id}">Sample ${id}</option>`
    ).join('');

    if (uniqueSamples.length > 0) {
        selector.addEventListener('change', function() {
            const sampleId = parseInt(this.value);
            displaySampleExplanation(sampleId, data);
        });

        // Display first sample
        displaySampleExplanation(uniqueSamples[0], data);
    }
}

/**
 * Display explanation for selected sample
 */
function displaySampleExplanation(sampleId, data) {
    const sampleData = data.filter(d => d.sample_id === sampleId);
    const container = document.getElementById('sample-explanation');

    if (sampleData.length === 0) {
        container.innerHTML = '<p>No explanation data for this sample</p>';
        return;
    }

    const sample = sampleData[0];
    const html = `
        <div style="margin-bottom: 20px;">
            <p><strong>Sample ID:</strong> ${sample.sample_id}</p>
            <p><strong>Prediction:</strong> ${sample.prediction === 1 ? 'Anomaly' : 'Normal'}</p>
            <p><strong>Anomaly Probability:</strong> ${(sample.anomaly_probability).toFixed(4)}</p>
            <p><strong>Label:</strong> ${sample.label}</p>
        </div>
        <table class="table">
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Value</th>
                    <th>Z-Score</th>
                    <th>Global Importance</th>
                    <th>Weighted Score</th>
                </tr>
            </thead>
            <tbody>
                ${sampleData.map(row => `
                    <tr>
                        <td>${row.feature}</td>
                        <td>${(row.value).toFixed(2)}</td>
                        <td>${(row.zscore).toFixed(2)}</td>
                        <td>${(row.global_importance).toFixed(4)}</td>
                        <td>${(row.weighted_score).toFixed(4)}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;

    container.innerHTML = html;
}

/**
 * Render Predictions Tab
 */
function renderPredictions() {
    const predictions = dashboardData.predictions;
    const container = document.getElementById('predictions-metrics');

    const cards = [];
    
    if (predictions.internal_test) {
        cards.push({
            title: 'Internal Test - Anomalies',
            value: predictions.internal_test.anomalies,
            unit: 'samples',
        });
        cards.push({
            title: 'Internal Test - Normals',
            value: predictions.internal_test.normals,
            unit: 'samples',
        });
    }

    if (predictions.validation) {
        cards.push({
            title: 'Validation - Anomalies',
            value: predictions.validation.anomalies,
            unit: 'samples',
        });
        cards.push({
            title: 'Validation - Normals',
            value: predictions.validation.normals,
            unit: 'samples',
        });
    }

    container.innerHTML = cards.map(card => `
        <div class="metric-card">
            <h3>${card.title}</h3>
            <div><span class="value">${card.value}</span><span class="unit">${card.unit}</span></div>
        </div>
    `).join('');
}

/**
 * Render Uncertainty Tab
 */
function renderUncertainty() {
    const uncertainty = dashboardData.metrics.uncertainty || {};
    const container = document.getElementById('uncertainty-metrics');

    const cards = [
        {
            title: 'ECE (Expected Calibration Error)',
            value: (uncertainty.ece || 0).toFixed(4),
            unit: '',
        },
        {
            title: 'Brier Score',
            value: (uncertainty.brier_score || 0).toFixed(4),
            unit: '',
        },
        {
            title: 'Log Loss',
            value: (uncertainty.log_loss || 0).toFixed(4),
            unit: '',
        },
        {
            title: 'Average Confidence',
            value: (uncertainty.avg_confidence * 100 || 0).toFixed(2),
            unit: '%',
        },
    ];

    container.innerHTML = cards.map(card => `
        <div class="metric-card">
            <h3>${card.title}</h3>
            <div><span class="value">${card.value}</span><span class="unit">${card.unit}</span></div>
        </div>
    `).join('');

    // Render uncertainty details
    const listContainer = document.getElementById('uncertainty-list');
    const items = [
        `Confidence Cutoff: ${(uncertainty.confidence_cutoff || 0.8).toFixed(2)}`,
        `Coverage at Confidence Cutoff: ${(uncertainty.coverage_at_confidence_cutoff * 100 || 0).toFixed(2)}%`,
        `Selective Accuracy: ${(uncertainty.selective_accuracy_at_confidence_cutoff * 100 || 0).toFixed(2)}%`,
        `Selective F1 Score: ${(uncertainty.selective_f1_at_confidence_cutoff || 0).toFixed(4)}`,
    ];

    listContainer.innerHTML = items.map(item => `<li>${item}</li>`).join('');
}

/**
 * Setup explanation filters
 */
function setupExplanationFilters() {
    // Placeholder for additional filter setup
}

/**
 * Show error message
 */
function showError(message) {
    const container = document.querySelector('.container');
    const alert = document.createElement('div');
    alert.style.cssText = `
        background-color: #e74c3c;
        color: white;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 20px;
    `;
    alert.textContent = message;
    container.insertBefore(alert, container.firstChild);
}
