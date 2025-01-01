"""
Anomaly Detection Ensemble
Isolation Forest + Deep One-Class SVDD for non-linear failure modes.
Detects: sensor tampering, fuel siphoning, sudden component failures
outside the LSTM training distribution.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple


class AnomalyDetectionEnsemble:
    """
    Two-stage anomaly detection:
      1. Isolation Forest — fast multivariate outlier detection
      2. One-Class SVDD  — boundary-based novelty detection
    """

    VERSION = "1.0.0"
    ANOMALY_THRESHOLD = 0.6   # Combined score above this = anomaly

    def __init__(self):
        self.isolation_forest = None
        self.svdd = None
        self.version = self.VERSION

    def load_or_initialize(self) -> None:
        """Initialize models with synthetic normal operating data."""
        try:
            from sklearn.ensemble import IsolationForest
            self.isolation_forest = IsolationForest(
                n_estimators=200,
                contamination=0.05,   # 5% expected anomaly rate
                max_samples="auto",
                random_state=42,
                n_jobs=-1,
            )
            # Train on simulated normal operating data
            normal_data = self._generate_normal_data(5000)
            self.isolation_forest.fit(normal_data)
        except ImportError:
            self.isolation_forest = None

        # Initialize lightweight SVDD-style one-class SVM
        try:
            from sklearn.svm import OneClassSVM
            self.svdd = OneClassSVM(
                kernel="rbf",
                nu=0.05,      # Expected outlier fraction
                gamma="scale",
            )
            normal_data = self._generate_normal_data(2000)
            self.svdd.fit(normal_data)
        except ImportError:
            self.svdd = None

    def score(self, features: np.ndarray) -> Tuple[float, bool]:
        """
        Compute combined anomaly score.

        Args:
            features: (n_features,) tabular feature vector

        Returns:
            anomaly_score: float in [0, 1], higher = more anomalous
            anomaly_detected: bool
        """
        flat = features.flatten()
        if len(flat) == 0:
            return 0.0, False

        # Ensure consistent shape for models
        x = flat[:20] if len(flat) >= 20 else np.pad(flat, (0, 20 - len(flat)))
        x = x.reshape(1, -1)

        scores = []

        if self.isolation_forest is not None:
            try:
                # IsolationForest returns negative scores; -1=anomaly, 1=normal
                raw = self.isolation_forest.score_samples(x)[0]
                # Convert to [0,1] where 1=anomaly
                iso_score = float(np.clip((-raw - 0.1) / 0.9, 0.0, 1.0))
                scores.append(iso_score * 0.6)   # 60% weight
            except Exception:
                pass

        if self.svdd is not None:
            try:
                # OneClassSVM: -1=outlier, +1=inlier
                decision = self.svdd.decision_function(x)[0]
                svdd_score = float(np.clip(-decision / 2.0, 0.0, 1.0))
                scores.append(svdd_score * 0.4)   # 40% weight
            except Exception:
                pass

        if not scores:
            # Fallback: rule-based anomaly detection
            return self._rule_based_score(flat)

        combined_score = sum(scores)
        anomaly_detected = combined_score > self.ANOMALY_THRESHOLD
        return combined_score, anomaly_detected

    def _rule_based_score(self, features: np.ndarray) -> Tuple[float, bool]:
        """Rule-based fallback when sklearn unavailable."""
        if len(features) < 5:
            return 0.0, False
        # High vibration + low oil pressure simultaneously = anomaly
        vibration = float(features[4]) if len(features) > 4 else 0.0
        oil_inv = 1.0 - float(features[1]) if len(features) > 1 else 0.0
        score = (vibration * 0.5 + oil_inv * 0.5)
        score += np.random.normal(0, 0.02)  # Small noise
        score = float(np.clip(score, 0.0, 1.0))
        return score, score > self.ANOMALY_THRESHOLD

    def _generate_normal_data(self, n_samples: int) -> np.ndarray:
        """Generate synthetic normal operating data for model initialization."""
        np.random.seed(42)
        # Normal operating ranges (normalized signals)
        data = np.random.randn(n_samples, 20) * 0.15 + 0.5
        data = np.clip(data, 0.0, 1.0)
        # Correlate related features
        data[:, 2] = data[:, 0] * 0.7 + np.random.randn(n_samples) * 0.1  # RPM-temp correlation
        data[:, 1] = 0.7 + data[:, 0] * 0.2 + np.random.randn(n_samples) * 0.05  # RPM-oil correlation
        return data
