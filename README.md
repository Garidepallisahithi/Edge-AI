# Edge AI EV Battery Thermal Anomaly Early-Warning and Diagnostic System

## Overview

Electric vehicle battery failures can lead to thermal runaway, unexpected downtime, reduced battery lifespan, and significant safety risks.

This project presents an end-to-end Edge AI system for early battery anomaly detection using vehicle telemetry, uncertainty-aware prediction, and explainable diagnostics.

The system is designed to detect abnormal battery behavior before critical failure occurs while remaining lightweight enough for edge deployment.

---

## Problem Statement

Current battery management systems primarily provide protection and monitoring after abnormal conditions emerge.

Challenges include:

* Delayed fault detection
* High maintenance costs
* Fleet downtime
* Limited explainability
* False alarms caused by noisy sensor readings

An intelligent early-warning system is needed to identify battery anomalies before safety-critical events occur.

---

## Proposed Solution

The system combines:

* Telemetry-based feature engineering
* Supervised anomaly detection
* Probability calibration
* Uncertainty estimation
* Explainable AI
* Interactive monitoring dashboard

to provide interpretable and trustworthy battery-health monitoring.

---

## System Architecture

Battery Telemetry

↓

Feature Engineering

↓

XGBoost Anomaly Detection

↓

Probability Calibration

↓

Uncertainty Estimation

↓

Explainability Engine

↓

Dashboard & Alert Generation

---

## Key Capabilities

### Early Anomaly Detection

Detects abnormal battery behavior using engineered battery-health indicators.

### Uncertainty-Aware Decisions

Provides confidence estimates for every prediction.

### Explainable AI

Identifies which battery characteristics contribute most to anomaly predictions.

### Edge-AI Readiness

Designed for deployment on resource-constrained systems.

### Interactive Dashboard

Visualizes:

* Risk scores
* Model confidence
* Calibration quality
* Feature contributions
* High-risk battery samples

---

## Dataset

Battery telemetry data containing:

* Voltage measurements
* Current measurements
* Temperature measurements
* State-of-charge indicators
* Mileage information

Processed into engineered battery-health features for anomaly detection.

---

## Machine Learning Pipeline

### Baseline Models

* Random Forest
* Isolation Forest

### Final Model

* XGBoost Classifier

Selected based on superior anomaly detection performance.

---

## Evaluation Results

### Internal Test Results

| Metric   | Score  |
| -------- | ------ |
| Accuracy | 0.9672 |
| F1 Score | 0.8986 |
| ROC-AUC  | 0.9917 |
| PR-AUC   | 0.9689 |

### Calibration Results

| Metric                     | Score  |
| -------------------------- | ------ |
| Brier Score                | 0.0231 |
| Expected Calibration Error | 0.0048 |

### Selective Prediction

| Metric             | Score  |
| ------------------ | ------ |
| Selective Accuracy | 0.9889 |
| Coverage           | 93.52% |

---

## Explainability

The system provides local explanations for anomaly predictions.

Example outputs include:

* Feature contribution ranking
* Risk factor identification
* Confidence estimation
* Anomaly reasoning

This enables engineers to understand why a battery is flagged as abnormal.

---

## Dashboard

The Streamlit dashboard includes:

* Performance monitoring
* Uncertainty analysis
* Calibration diagnostics
* Explainability explorer
* Prediction analysis

---

## Repository Structure

app/dashboard/

data/

docs/

models/

reports/

scripts/

tests/

---

## How to Run

### Install Dependencies

pip install -r requirements.txt

### Launch Dashboard

streamlit run app/dashboard/app.py

---

## Project Status

Prototype Complete

* Feature Engineering
* Model Training
* Cross Validation
* Explainability
* Calibration
* Dashboard Integration
* Deployment Preparation

---

## Potential Industry Impact

* Early thermal anomaly detection
* Improved EV safety
* Reduced fleet maintenance costs
* Lower downtime
* Better battery lifespan management
* Explainable decision support for operators

---

## Future Work

* Real-time edge deployment
* Fleet-wide federated learning
* Digital twin integration
* Online model adaptation
* Multi-vehicle anomaly generalization

