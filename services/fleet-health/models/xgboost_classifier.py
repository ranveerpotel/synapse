"""
XGBoost Near-Term Failure Classifier
Rapid failure probability classification within a 72-hour window.
Aggregates decisions from thousands of weak learners (decision trees)
on high-dimensional telematics + environmental + behavioral features.
"""
from __future__ import annotations
import os
import numpy as np
from typing import Optional


class XGBoostFailureClassifier:
    """
    Gradient boosting failure classifier.
    Output: P(failure within 72 hours) ∈ [0, 1].
    """

    VERSION = "1.0.0"
    FAILURE_THRESHOLD = 0.5
    HIGH_RISK_THRESHOLD = 0.7
    CRITICAL_THRESHOLD = 0.9

    def __init__(self):
        self.model = None
        self.feature_names: list[str] = []
        self.version = self.VERSION

    def load_or_initialize(self, model_path: str = None) -> None:
        """Load saved XGBoost model or train a lightweight surrogate."""
        try:
            import xgboost as xgb

            self.model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="auc",
                random_state=42,
                n_jobs=-1,
            )

            if model_path and os.path.exists(model_path):
                self.model.load_model(model_path)
            else:
                # Initialize with synthetic training data for dev
                self._train_synthetic()

        except ImportError:
            self.model = None

    def _train_synthetic(self) -> None:
        """Train on synthetic data matching Weibull failure distributions."""
        try:
            import xgboost as xgb
            np.random.seed(42)
            n_samples = 10_000
            n_features = 20

            # Simulate correlated features
            X = np.random.randn(n_samples, n_features)
            # Higher thermal stress + vibration → higher failure prob
            failure_score = (
                0.3 * X[:, 2]   # thermal stress
                + 0.25 * X[:, 4]  # vibration
                + 0.2 * (-X[:, 1])  # inv oil pressure
                + 0.15 * X[:, 0]  # engine load
                + 0.1 * np.random.randn(n_samples)
            )
            y = (failure_score > 0.5).astype(int)

            self.model.fit(X, y, eval_set=[(X, y)], verbose=False)
        except Exception:
            self.model = None

    def predict_proba(self, features: np.ndarray) -> float:
        """
        Predict probability of component failure within 72 hours.

        Args:
            features: (n_features,) or (1, n_features) tabular feature vector

        Returns:
            failure_probability: float in [0, 1]
        """
        if self.model is not None:
            return self._xgb_predict(features)
        return self._heuristic_predict(features)

    def _xgb_predict(self, features: np.ndarray) -> float:
        try:
            if features.ndim == 1:
                features = features.reshape(1, -1)
            # Ensure shape matches model expectations
            if features.shape[1] != 20:
                features = self._pad_features(features, 20)
            proba = self.model.predict_proba(features)[0, 1]
            return float(np.clip(proba, 0.0, 1.0))
        except Exception:
            return self._heuristic_predict(features)

    def _heuristic_predict(self, features: np.ndarray) -> float:
        """Heuristic failure probability when XGBoost unavailable."""
        if len(features) == 0:
            return 0.1
        flat = features.flatten()
        # Use first available features
        thermal = float(flat[2]) if len(flat) > 2 else 0.3
        vibration = float(flat[4]) if len(flat) > 4 else 0.2
        oil_risk = 1.0 - float(flat[1]) if len(flat) > 1 else 0.2
        base_prob = (thermal * 0.4 + vibration * 0.35 + oil_risk * 0.25)
        return float(np.clip(base_prob + np.random.normal(0, 0.02), 0.0, 1.0))

    def _pad_features(self, features: np.ndarray, target_dim: int) -> np.ndarray:
        current_dim = features.shape[1]
        if current_dim >= target_dim:
            return features[:, :target_dim]
        padding = np.zeros((features.shape[0], target_dim - current_dim))
        return np.hstack([features, padding])

    def get_feature_importance(self) -> dict:
        """Return top feature importances for SHAP explainability."""
        if self.model is None:
            return {}
        try:
            importances = self.model.feature_importances_
            names = [f"feature_{i}" for i in range(len(importances))]
            return dict(sorted(zip(names, importances), key=lambda x: x[1], reverse=True)[:10])
        except Exception:
            return {}
