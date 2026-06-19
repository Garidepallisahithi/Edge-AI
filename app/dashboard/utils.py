"""
Dashboard utilities for data processing and visualization helpers.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import numpy as np


class MetricsLoader:
    """Load and parse phase 5 metrics and outputs."""
    
    def __init__(self, reports_dir: Path):
        """Initialize with reports directory path."""
        self.reports_dir = Path(reports_dir)
        self._cache = {}
    
    def get_calibration_metrics(self) -> Dict[str, Any]:
        """Load calibration metrics."""
        cache_key = "calibration"
        if cache_key not in self._cache:
            path = self.reports_dir / "calibration.json"
            if path.exists():
                with open(path) as f:
                    self._cache[cache_key] = json.load(f)
            else:
                self._cache[cache_key] = {}
        return self._cache[cache_key]
    
    def get_uncertainty_metrics(self) -> Dict[str, Any]:
        """Load uncertainty quantification metrics."""
        cache_key = "uncertainty"
        if cache_key not in self._cache:
            path = self.reports_dir / "uncertainty.json"
            if path.exists():
                with open(path) as f:
                    self._cache[cache_key] = json.load(f)
            else:
                self._cache[cache_key] = {}
        return self._cache[cache_key]
    
    def get_classification_metrics(self) -> Dict[str, Any]:
        """Load classification metrics."""
        cache_key = "classification"
        if cache_key not in self._cache:
            path = self.reports_dir / "metrics.json"
            if path.exists():
                with open(path) as f:
                    self._cache[cache_key] = json.load(f)
            else:
                self._cache[cache_key] = {}
        return self._cache[cache_key]
    
    def get_global_importance(self) -> pd.DataFrame:
        """Load global feature importance rankings."""
        cache_key = "global_importance"
        if cache_key not in self._cache:
            path = self.reports_dir / "explanations" / "global_feature_importance.csv"
            if path.exists():
                self._cache[cache_key] = pd.read_csv(path)
            else:
                self._cache[cache_key] = pd.DataFrame()
        return self._cache[cache_key].copy()
    
    def get_local_explanations(self) -> pd.DataFrame:
        """Load local explanations (per-sample feature contributions)."""
        cache_key = "local_explanations"
        if cache_key not in self._cache:
            path = self.reports_dir / "explanations" / "local_explanations.csv"
            if path.exists():
                self._cache[cache_key] = pd.read_csv(path)
            else:
                self._cache[cache_key] = pd.DataFrame()
        return self._cache[cache_key].copy()
    
    def get_predictions(self, split: str = "internal_test") -> pd.DataFrame:
        """
        Load predictions for a given split (internal_test, validation, official_test).
        """
        cache_key = f"predictions_{split}"
        if cache_key not in self._cache:
            filename = f"{split}_predictions_calibrated.csv"
            path = self.reports_dir / filename
            if path.exists():
                self._cache[cache_key] = pd.read_csv(path)
            else:
                self._cache[cache_key] = pd.DataFrame()
        return self._cache[cache_key].copy()


class ExplanationAnalyzer:
    """Analyze local and global explanations."""
    
    @staticmethod
    def get_top_features(
        df_local: pd.DataFrame,
        sample_id: int,
        n_features: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get top N features for a given sample."""
        sample_data = df_local[df_local["sample_id"] == sample_id]
        if sample_data.empty:
            return []
        
        top = sample_data.nlargest(n_features, "weighted_score")
        return top[
            ["feature", "value", "zscore", "global_importance", "weighted_score"]
        ].to_dict(orient="records")
    
    @staticmethod
    def get_anomalous_samples(
        df_local: pd.DataFrame,
        n_samples: int = 5,
    ) -> List[Dict[str, Any]]:
        """Get most anomalous samples (highest anomaly_probability)."""
        unique_samples = df_local.drop_duplicates("sample_id").sort_values(
            "anomaly_probability", ascending=False
        )
        return unique_samples.head(n_samples)[
            ["sample_id", "label", "anomaly_probability", "prediction"]
        ].to_dict(orient="records")
    
    @staticmethod
    def get_uncertain_samples(
        df_predictions: pd.DataFrame,
        threshold: float = 0.05,
        n_samples: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get samples with lowest confidence (closest to decision boundary)."""
        df_predictions["confidence"] = df_predictions["anomaly_probability"].apply(
            lambda x: min(x, 1 - x)
        )
        uncertain = df_predictions[df_predictions["confidence"] < threshold]
        return uncertain.nlargest(n_samples, "confidence")[
            ["sample_id", "anomaly_probability", "confidence", "prediction", "label"]
        ].to_dict(orient="records")


class MetricsFormatter:
    """Format metrics for dashboard display."""
    
    @staticmethod
    def format_metric(value: float, precision: int = 4) -> str:
        """Format metric value to specified precision."""
        if not isinstance(value, (int, float)):
            return str(value)
        return f"{value:.{precision}f}"
    
    @staticmethod
    def format_percentage(value: float, precision: int = 2) -> str:
        """Format value as percentage."""
        return f"{value * 100:.{precision}f}%"
    
    @staticmethod
    def summary_card(title: str, value: float, unit: str = "", formatting: str = "metric") -> Dict[str, str]:
        """Create a summary card for dashboard."""
        if formatting == "metric":
            formatted = MetricsFormatter.format_metric(value)
        elif formatting == "percentage":
            formatted = MetricsFormatter.format_percentage(value)
        else:
            formatted = str(value)
        
        return {
            "title": title,
            "value": formatted,
            "unit": unit,
        }
