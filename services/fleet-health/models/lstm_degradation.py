"""
LSTM / TCN Degradation Model
Long-term wear trend detection via recurrent networks.
Processes rolling 200-step time-series windows of CAN bus signals.
Target: 72hr failure lead time, ROC-AUC 0.89.
"""
from __future__ import annotations
import os
import numpy as np
from typing import Optional


class LSTMDegradationModel:
    """
    LSTM-based degradation model for fleet components.
    In production: uses PyTorch LSTM/TCN trained on 6yr Weibull-calibrated history.
    Development: uses a physics-informed heuristic for demonstration.
    """

    VERSION = "1.0.0"
    N_COMPONENTS = 10
    SEQUENCE_LENGTH = 200
    INPUT_DIM = 12  # Normalized CAN bus features

    def __init__(self):
        self.model = None
        self.version = self.VERSION
        self._history: dict[str, list] = {}  # vehicle_id -> rolling window

    def load_or_initialize(self, model_path: str = None) -> None:
        """Load saved model or initialize with random weights for dev."""
        try:
            import torch
            import torch.nn as nn

            class LSTMDegradation(nn.Module):
                def __init__(self, input_dim: int, hidden_dim: int, n_components: int, n_layers: int = 2):
                    super().__init__()
                    self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.2)
                    self.fc_degradation = nn.Linear(hidden_dim, n_components)
                    self.sigmoid = nn.Sigmoid()

                def forward(self, x):
                    # x: (batch, seq_len, input_dim)
                    lstm_out, _ = self.lstm(x)
                    # Use last hidden state
                    last_hidden = lstm_out[:, -1, :]
                    degradation = self.sigmoid(self.fc_degradation(last_hidden))
                    return degradation  # (batch, n_components) in [0,1]

            self.model = LSTMDegradation(
                input_dim=self.INPUT_DIM,
                hidden_dim=128,
                n_components=self.N_COMPONENTS,
            )

            if model_path and os.path.exists(model_path):
                import torch
                state = torch.load(model_path, map_location="cpu")
                self.model.load_state_dict(state)
                self.model.eval()
            else:
                # Dev mode: random weights (replace with trained weights in prod)
                self.model.eval()

        except ImportError:
            # PyTorch not available — use heuristic fallback
            self.model = None

    def predict(self, time_series: np.ndarray) -> np.ndarray:
        """
        Predict degradation scores for all components.

        Args:
            time_series: (seq_len, input_dim) array of normalized signals

        Returns:
            degradation_scores: (n_components,) array in [0, 1]
                                0 = new/healthy, 1 = failed
        """
        if self.model is not None:
            return self._torch_predict(time_series)
        return self._heuristic_predict(time_series)

    def _torch_predict(self, time_series: np.ndarray) -> np.ndarray:
        try:
            import torch
            # Pad or trim to SEQUENCE_LENGTH
            seq = self._pad_sequence(time_series)
            tensor = torch.FloatTensor(seq).unsqueeze(0)  # (1, seq_len, input_dim)
            with torch.no_grad():
                output = self.model(tensor)
            return output.squeeze(0).numpy()
        except Exception:
            return self._heuristic_predict(time_series)

    def _heuristic_predict(self, time_series: np.ndarray) -> np.ndarray:
        """Physics-informed heuristic degradation estimate."""
        if len(time_series) == 0:
            return np.zeros(self.N_COMPONENTS) + 0.1

        # Use last observation
        last = time_series[-1] if len(time_series.shape) > 1 else time_series

        # Map normalized signals to component degradation estimates
        # Higher thermal stress → engine/cooling degradation
        # Higher vibration → tire/brake degradation
        # Low oil pressure → engine/transmission risk
        thermal_stress = float(last[2]) if len(last) > 2 else 0.5   # coolant_temp_norm
        vibration = float(last[4]) if len(last) > 4 else 0.3        # vibration_norm
        oil_pressure = 1.0 - float(last[1]) if len(last) > 1 else 0.3  # inv oil_pressure_norm

        scores = np.array([
            (thermal_stress * 0.6 + oil_pressure * 0.4) * 0.8,  # engine
            oil_pressure * 0.7,                                    # transmission
            vibration * 0.6,                                       # brakes
            vibration * 0.5,                                       # tire_fl
            vibration * 0.5,                                       # tire_fr
            vibration * 0.5,                                       # tire_rl
            vibration * 0.5,                                       # tire_rr
            oil_pressure * 0.8,                                    # oil_system
            thermal_stress * 0.7,                                  # cooling_system
            0.1,                                                   # electrical (low baseline)
        ])
        # Add realistic noise
        noise = np.random.normal(0, 0.02, self.N_COMPONENTS)
        return np.clip(scores + noise, 0.0, 1.0)

    def _pad_sequence(self, seq: np.ndarray) -> np.ndarray:
        """Pad or trim sequence to SEQUENCE_LENGTH."""
        if len(seq) >= self.SEQUENCE_LENGTH:
            return seq[-self.SEQUENCE_LENGTH:]
        pad_len = self.SEQUENCE_LENGTH - len(seq)
        if len(seq.shape) == 1:
            return np.pad(seq, (pad_len, 0), mode="edge")
        return np.pad(seq, ((pad_len, 0), (0, 0)), mode="edge")
