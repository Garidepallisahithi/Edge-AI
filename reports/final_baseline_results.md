# Battery Fault Detection Baseline Results

## Dataset

* Normal (00): 23,735 samples
* Fault (10): 4,654 samples

## Random Forest

Accuracy: 0.93

Class 10:

* Precision: 0.71
* Recall: 0.77
* F1: 0.74

Feature Importance:

* mileage: 0.3560
* temp_min: 0.1329
* temp_max: 0.1285
* volt_std: 0.1253
* current_mean: 0.0999
* current_std: 0.0582
* soc_mean: 0.0497
* volt_mean: 0.0496

## XGBoost

Accuracy: 0.95

Class 10:

* Precision: 0.79
* Recall: 0.85
* F1: 0.81

Feature Importance:

* mileage: 0.2327
* volt_std: 0.1626
* temp_max: 0.1514
* temp_min: 0.1269
* current_std: 0.0915
* current_mean: 0.0899
* volt_mean: 0.0802
* soc_mean: 0.0647

## Conclusions

* Mileage is the strongest predictive feature.
* XGBoost currently performs best.
* Unsupervised anomaly detection models underperform supervised models.
* Test dataset is unlabeled and intended for inference only.
