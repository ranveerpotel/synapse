"""
Telemetry Normalizer
Min-max normalization, unit conversion, and temporal alignment
for CAN bus signals before publishing to Kafka.
"""
from __future__ import annotations
from datetime import datetime
from shared.schemas.models import CANBusSignal


# Signal bounds for min-max normalization
BOUNDS = {
    "engine_rpm":         (0.0, 8000.0),
    "torque_nm":          (0.0, 3000.0),
    "oil_pressure_kpa":   (0.0, 700.0),
    "coolant_temp_c":     (-40.0, 150.0),
    "tire_pressure_kpa":  (0.0, 1000.0),
    "vibration_rms_g":    (0.0, 50.0),
    "fuel_level_pct":     (0.0, 100.0),
    "speed_kmh":          (0.0, 200.0),
}


def _norm(value: float, key: str) -> float:
    lo, hi = BOUNDS.get(key, (0.0, 1.0))
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


class TelemetryNormalizer:
    """
    Normalizes raw CAN bus signals to [0, 1] range and adds
    derived features used by downstream ML models.
    """

    def normalize_can(self, signal: CANBusSignal) -> dict:
        raw = signal.model_dump()

        normalized = {
            # Identity fields (not normalized)
            "vehicle_id": signal.vehicle_id,
            "timestamp": signal.timestamp.isoformat(),
            "latitude": signal.latitude,
            "longitude": signal.longitude,
            "odometer_km": signal.odometer_km,
            "fault_codes": signal.fault_codes,
            "harsh_brake_event": signal.harsh_brake_event,
            "harsh_accel_event": signal.harsh_accel_event,

            # Normalized continuous signals
            "engine_rpm_norm": _norm(signal.engine_rpm, "engine_rpm"),
            "torque_norm": _norm(signal.torque_nm, "torque_nm"),
            "oil_pressure_norm": _norm(signal.oil_pressure_kpa, "oil_pressure_kpa"),
            "coolant_temp_norm": _norm(signal.coolant_temp_c, "coolant_temp_c"),
            "tire_pressure_fl_norm": _norm(signal.tire_pressure_fl_kpa, "tire_pressure_kpa"),
            "tire_pressure_fr_norm": _norm(signal.tire_pressure_fr_kpa, "tire_pressure_kpa"),
            "tire_pressure_rl_norm": _norm(signal.tire_pressure_rl_kpa, "tire_pressure_kpa"),
            "tire_pressure_rr_norm": _norm(signal.tire_pressure_rr_kpa, "tire_pressure_kpa"),
            "vibration_norm": _norm(signal.vibration_rms_g, "vibration_rms_g"),
            "fuel_level_norm": _norm(signal.fuel_level_pct, "fuel_level_pct"),
            "speed_norm": _norm(signal.speed_kmh, "speed_kmh"),

            # Derived features
            "tire_pressure_variance": self._tire_variance(signal),
            "thermal_stress_index": self._thermal_stress(signal),
            "engine_load_index": self._engine_load(signal),
            "fuel_consumption_rate_lph": self._fuel_rate(signal),

            # Raw values (retained for interpretability)
            "engine_rpm_raw": signal.engine_rpm,
            "coolant_temp_c_raw": signal.coolant_temp_c,
            "oil_pressure_kpa_raw": signal.oil_pressure_kpa,
            "speed_kmh_raw": signal.speed_kmh,
        }
        return normalized

    def _tire_variance(self, s: CANBusSignal) -> float:
        """Variance in tire pressure across four tires (indicator of imbalance)."""
        pressures = [
            s.tire_pressure_fl_kpa, s.tire_pressure_fr_kpa,
            s.tire_pressure_rl_kpa, s.tire_pressure_rr_kpa,
        ]
        mean = sum(pressures) / 4
        variance = sum((p - mean) ** 2 for p in pressures) / 4
        return min(1.0, variance / 10000.0)

    def _thermal_stress(self, s: CANBusSignal) -> float:
        """Combined thermal stress index from coolant temp and oil pressure."""
        temp_norm = _norm(s.coolant_temp_c, "coolant_temp_c")
        # Low oil pressure at high temp = high stress
        oil_inv = 1.0 - _norm(s.oil_pressure_kpa, "oil_pressure_kpa")
        return (temp_norm * 0.6 + oil_inv * 0.4)

    def _engine_load(self, s: CANBusSignal) -> float:
        """Engine load index combining RPM and torque."""
        return (_norm(s.engine_rpm, "engine_rpm") * 0.5 +
                _norm(s.torque_nm, "torque_nm") * 0.5)

    def _fuel_rate(self, s: CANBusSignal) -> float:
        """Estimated fuel consumption rate (L/h) from engine RPM and torque."""
        # Simplified BSFC model: fuel_rate ≈ (RPM * torque) / (efficiency * constant)
        power_kw = (s.engine_rpm * s.torque_nm) / (9549.0)
        bsfc_g_per_kwh = 210.0  # typical diesel BSFC
        return (power_kw * bsfc_g_per_kwh) / (840.0 * 1000.0) * 1000  # L/h
