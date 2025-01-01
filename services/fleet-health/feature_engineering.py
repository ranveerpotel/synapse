"""
Fleet Feature Engineering
Rolling window statistics, lag features, and frequency domain features
for LSTM and XGBoost model inputs.
"""
from __future__ import annotations
import json
import numpy as np
from typing import Optional
from shared.utils.helpers import SynapseCache

SIGNAL_KEYS = [
    "engine_rpm_norm", "oil_pressure_norm", "coolant_temp_norm",
    "torque_norm", "vibration_norm", "fuel_level_norm", "speed_norm",
    "tire_pressure_fl_norm", "tire_pressure_fr_norm",
    "tire_pressure_rl_norm", "tire_pressure_rr_norm",
    "thermal_stress_index", "engine_load_index",
]
WINDOW_SIZE = 200   # Steps for LSTM input
TABULAR_STATS = ["mean", "std", "min", "max", "range"]


class FleetFeatureEngineer:
    """
    Builds time-series and tabular feature vectors for ML models.
    Maintains a rolling window per vehicle in Redis.
    """

    def __init__(self, window_size: int = WINDOW_SIZE):
        self.window_size = window_size

    async def build_features(
        self, vehicle_id: str, signal: dict, cache: SynapseCache
    ) -> dict:
        """
        Build complete feature set from latest signal + rolling history.
        Returns dict with 'time_series' (for LSTM) and 'tabular' (for XGBoost).
        """
        # Load rolling window from Redis
        window = await self._load_window(vehicle_id, cache)

        # Append current observation
        current_vec = self._extract_signal_vector(signal)
        window.append(current_vec)

        # Maintain window size
        if len(window) > self.window_size:
            window = window[-self.window_size:]

        # Save updated window
        await self._save_window(vehicle_id, window, cache)

        # Convert to numpy
        ts_array = np.array(window, dtype=np.float32)  # (seq_len, n_features)

        # Tabular features: rolling stats + current observation + event flags
        tabular = self._build_tabular_features(ts_array, signal)

        return {
            "time_series": ts_array,
            "tabular": tabular,
            "window_length": len(window),
        }

    def _extract_signal_vector(self, signal: dict) -> list[float]:
        """Extract ordered feature vector from normalized signal dict."""
        return [float(signal.get(key, 0.5)) for key in SIGNAL_KEYS]

    def _build_tabular_features(self, ts: np.ndarray, signal: dict) -> np.ndarray:
        """Build tabular feature vector for XGBoost input."""
        features = []

        # Rolling statistics for each signal
        for i in range(ts.shape[1]):
            col = ts[:, i]
            features.extend([
                float(np.mean(col)),
                float(np.std(col)),
                float(np.min(col)),
                float(np.max(col)),
                float(np.max(col) - np.min(col)),     # range
                float(col[-1]),                         # latest value
            ])

        # Event flags
        features.append(1.0 if signal.get("harsh_brake_event") else 0.0)
        features.append(1.0 if signal.get("harsh_accel_event") else 0.0)
        features.append(len(signal.get("fault_codes", [])) / 10.0)

        # Trend features (last 10 vs last 50 average)
        if len(ts) >= 50:
            recent_mean = ts[-10:, :].mean(axis=0)
            older_mean = ts[-50:-10, :].mean(axis=0)
            trend = recent_mean - older_mean
            features.extend(trend.tolist())
        else:
            features.extend([0.0] * ts.shape[1])

        return np.array(features, dtype=np.float32)

    async def _load_window(self, vehicle_id: str, cache: SynapseCache) -> list:
        raw = await cache.get_json(f"fleet:window:{vehicle_id}")
        return raw if raw else []

    async def _save_window(self, vehicle_id: str, window: list, cache: SynapseCache) -> None:
        await cache.set_json(f"fleet:window:{vehicle_id}", window, ttl=3600)
