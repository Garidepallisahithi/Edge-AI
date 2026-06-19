# Phase 5 Dashboard

A Flask-based web dashboard for visualizing Phase 5 outputs: calibration metrics, feature importance, local explanations, and uncertainty quantification.

## Directory Structure

```
app/dashboard/
├── app.py                    # Flask application and API endpoints
├── utils.py                  # Utility classes for data loading and analysis
├── templates/
│   └── index.html           # Main dashboard HTML template
├── static/
│   ├── css/
│   │   └── style.css        # Dashboard styling
│   └── js/
│       └── dashboard.js     # Dashboard JavaScript/interactivity
└── README.md                # This file
```

## Features

- **Overview Tab**: Key metrics and summary statistics
- **Calibration Tab**: Calibration metrics and reliability curves
- **Explanations Tab**: Global feature importance and per-sample local explanations
- **Predictions Tab**: Prediction distribution and statistics
- **Uncertainty Tab**: Uncertainty quantification metrics (ECE, Brier, etc.)

## Setup

### Install Dependencies

```bash
pip install flask flask-cors pandas numpy
```

### Run the Dashboard

```bash
cd app/dashboard
python app.py
```

The dashboard will be available at `http://localhost:5000`

## Data Sources

The dashboard automatically loads data from:
- `reports/phase5/calibration.json`
- `reports/phase5/metrics.json`
- `reports/phase5/uncertainty.json`
- `reports/phase5/explainability_summary.json`
- `reports/phase5/explanations/global_feature_importance.csv`
- `reports/phase5/explanations/local_explanations.csv`
- `reports/phase5/*_predictions_calibrated.csv`

## API Endpoints

- `GET /` - Main dashboard page
- `GET /api/metrics` - All metrics (calibration, classification, uncertainty)
- `GET /api/explanations` - Feature importance and local explanations
- `GET /api/predictions` - Prediction statistics

## Customization

- **Styling**: Edit `static/css/style.css`
- **Layout**: Edit `templates/index.html`
- **Charts**: Modify `static/js/dashboard.js`
- **Data Loading**: Update `utils.py` or `app.py` API endpoints

## Future Enhancements

- [ ] Interactive sample selection with drill-down
- [ ] Real-time model retraining monitoring
- [ ] Export reports as PDF
- [ ] Custom metric filters and ranges
- [ ] Dark mode toggle
- [ ] Model comparison views
