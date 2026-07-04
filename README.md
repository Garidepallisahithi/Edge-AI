# Edge AI EV Battery Thermal Anomaly Early Warning System

A competition-grade EV battery monitoring prototype that combines calibrated anomaly detection, uncertainty estimation, and explainable AI for early thermal risk detection.

## Executive Summary

Electric vehicle battery failures are safety-critical, expensive, and disruptive. This project provides an early-warning and diagnostic layer for EV battery telemetry using engineered battery-health features, calibrated machine learning predictions, uncertainty estimation, and explainable AI. The system is deployed as a public Streamlit dashboard and designed as a competition-grade prototype for safe, interpretable battery monitoring.

## Live Demo

https://edge-ai-ffubw72wesajtbzf6aarjp.streamlit.app/ 

## Problem Statement

Current battery monitoring systems are often reactive. They detect issues after abnormal behavior is already visible, which increases safety risk, downtime, and maintenance cost. A useful monitoring system must detect risk early, explain why a sample is flagged, and quantify confidence.

## Solution Overview

This project implements an end-to-end battery anomaly detection pipeline:

- Telemetry ingestion
- Battery-health feature engineering
- XGBoost anomaly detection
- Probability calibration
- Uncertainty estimation
- Explainability
- Interactive dashboard deployment

## System Architecture

https://drive.google.com/file/d/1f1IPbU6ssZN3KsdxxoPeEcKU6hnQWcfz/view?usp=drivesdk

The system ingests battery telemetry, engineers battery-health features, trains an XGBoost anomaly detector, calibrates probabilities, estimates uncertainty, and explains alerts through a Streamlit dashboard.

## Dataset

The prototype uses battery telemetry with engineered features derived from:

- Voltage
- Current
- Temperature
- State of charge
- Mileage

The labeled modeling set contains 500 samples with class imbalance between normal and fault cases.

## Modeling Approach

### Baselines
- Isolation Forest
- Random Forest

### Final Model
- XGBoost classifier with class imbalance handling

## Evaluation Results

### Cross-Validation
- Accuracy: 0.9580
- Precision: 0.8450
- Recall: 0.8495
- F1 Score: 0.8438

### Internal Test
- F1 Score: 0.8986
- ROC-AUC: 0.9917
- PR-AUC: 0.9689

### Calibration and Uncertainty
- Brier Score: 0.0231
- Expected Calibration Error (ECE): 0.0048
- Selective Accuracy: 0.9889
- Coverage: 93.52%

## Explainability

The system exposes the most influential factors behind each prediction. The strongest observed drivers include:
- `soc_std`
- `mileage`
- `current_min`
- `temp_mean`
- `volt_range`

## Dashboard Features

- Model performance summary
- Uncertainty and calibration analysis
- Local explanation explorer
- Prediction explorer
- Public deployment

## Repository Structure

- `app/dashboard/` — Streamlit application
- `scripts/` — feature engineering, model training, evaluation, and prediction scripts
- `models/` — trained model artifact
- `reports/` — metrics, analyses, and prediction outputs
- `docs/` — competitive analysis and supporting documents
  
 ## Authors
 - `Garedepalli Sahithi`
 -  `Gannavaram Lakshmi Satwika`

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app/dashboard/app.py

