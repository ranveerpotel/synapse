"""
SYNAPSE Quick Start Test
Verifies all services can be imported and core functionality works.
Run: python test_quickstart.py
"""
from __future__ import annotations
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

RESULTS = {}


def pytest_approx(expected, rel=1e-6):
    """Lightweight approximate comparison."""
    class Approx:
        def __eq__(self, actual):
            return abs(actual - expected) <= rel * abs(expected)
    return Approx()


def test(name: str):
    def decorator(fn):
        try:
            fn()
            RESULTS[name] = "✅ PASS"
        except Exception as e:
            RESULTS[name] = f"❌ FAIL: {e}"
        return fn
    return decorator


@test("Shared schemas")
def test_schemas():
    from shared.schemas.models import (
        CANBusSignal, VehicleHealthState, DriverStateScore,
        FusedStateVector, ModeDDRDecision, ComplianceEvent
    )
    from datetime import datetime
    signal = CANBusSignal(
        vehicle_id="VH0001", timestamp=datetime.utcnow(),
        engine_rpm=1800, torque_nm=500, oil_pressure_kpa=350,
        coolant_temp_c=90, tire_pressure_fl_kpa=800, tire_pressure_fr_kpa=800,
        tire_pressure_rl_kpa=800, tire_pressure_rr_kpa=800,
        vibration_rms_g=0.5, fuel_level_pct=75, odometer_km=120000,
        speed_kmh=80, latitude=40.71, longitude=-74.01,
    )
    assert signal.vehicle_id == "VH0001"


@test("Shared config")
def test_config():
    from shared.config.settings import get_settings
    settings = get_settings()
    assert settings.kafka_bootstrap == "localhost:9092"


@test("Shared utils")
def test_utils():
    from shared.utils.helpers import compute_event_hash, create_access_token, verify_token
    h = compute_event_hash({"key": "value"}, "GENESIS")
    assert len(h) == 64
    token = create_access_token({"sub": "test"})
    claims = verify_token(token)
    assert claims["sub"] == "test"


@test("Telemetry normalizer")
def test_normalizer():
    from services.telemetry_ingestion.normalizer import TelemetryNormalizer
    from shared.schemas.models import CANBusSignal
    from datetime import datetime
    norm = TelemetryNormalizer()
    signal = CANBusSignal(
        vehicle_id="VH_TEST", timestamp=datetime.utcnow(),
        engine_rpm=1500, torque_nm=400, oil_pressure_kpa=300,
        coolant_temp_c=85, tire_pressure_fl_kpa=780, tire_pressure_fr_kpa=785,
        tire_pressure_rl_kpa=775, tire_pressure_rr_kpa=780,
        vibration_rms_g=0.3, fuel_level_pct=60, odometer_km=80000,
        speed_kmh=90, latitude=40.0, longitude=-74.0,
    )
    result = norm.normalize_can(signal)
    assert 0.0 <= result["engine_rpm_norm"] <= 1.0
    assert "thermal_stress_index" in result


@test("Fleet health LSTM")
def test_lstm():
    import numpy as np
    from services.fleet_health.models.lstm_degradation import LSTMDegradationModel
    model = LSTMDegradationModel()
    model.load_or_initialize()
    ts = np.random.rand(50, 12).astype(np.float32)
    scores = model.predict(ts)
    assert len(scores) == 10
    assert all(0.0 <= s <= 1.0 for s in scores)


@test("Fleet health XGBoost")
def test_xgb():
    import numpy as np
    from services.fleet_health.models.xgboost_classifier import XGBoostFailureClassifier
    clf = XGBoostFailureClassifier()
    clf.load_or_initialize()
    features = np.random.rand(20).astype(np.float32)
    prob = clf.predict_proba(features)
    assert 0.0 <= prob <= 1.0


@test("Fleet health anomaly detection")
def test_anomaly():
    import numpy as np
    from services.fleet_health.models.anomaly_detector import AnomalyDetectionEnsemble
    det = AnomalyDetectionEnsemble()
    det.load_or_initialize()
    features = np.random.rand(20).astype(np.float32)
    score, detected = det.score(features)
    assert 0.0 <= score <= 1.0
    assert isinstance(detected, bool)


@test("Driver monitoring CNN + Bayesian HRV")
def test_driver():
    from services.driver_monitoring.main import CNNFatigueDetector, BayesianHRVFilter
    from shared.schemas.models import DriverVisionSignal, DriverPhysiologicalSignal
    from datetime import datetime

    cnn = CNNFatigueDetector()
    vision = DriverVisionSignal(
        driver_id="DR0001", vehicle_id="VH0001", timestamp=datetime.utcnow(),
        eye_closure_rate_pct=15.0, gaze_deviation_deg=5.0,
        head_pose_yaw_deg=3.0, head_pose_pitch_deg=1.0,
    )
    scores = cnn.predict(vision)
    assert 0.0 <= scores["fatigue_probability"] <= 1.0

    bf = BayesianHRVFilter()
    physio = DriverPhysiologicalSignal(
        driver_id="DR0001", timestamp=datetime.utcnow(),
        heart_rate_bpm=75, hrv_ms=55, skin_conductance_us=3.0, stress_index=0.3,
    )
    result = bf.update(physio)
    assert 0.0 <= result["fatigue_probability"] <= 1.0


@test("Routing GNN optimizer")
def test_routing():
    from services.routing_engine.main import GNNPathOptimiser, LogisticsGraph
    graph = LogisticsGraph()
    optimizer = GNNPathOptimiser(graph)
    result = optimizer.find_optimal_route("HUB_DALLAS", "HUB_HOUSTON")
    assert len(result["route"]) >= 2
    assert result["route"][0] == "HUB_DALLAS"
    assert result["route"][-1] == "HUB_HOUSTON"


@test("Supply chain Transformer")
def test_supply_chain():
    from services.supply_chain.main import MultivariateTransformerForecaster
    forecaster = MultivariateTransformerForecaster()
    result = forecaster.predict_delay(
        supplier_id="SUP001", tier=1, lane_type="ocean",
        current_lead_time_days=10.0, weather_index=0.3,
        port_congestion_index=0.4, vessel_schedule_risk=0.2,
    )
    assert 0.0 <= result["delay_probability"] <= 1.0


@test("Carbon calculator")
def test_carbon():
    from services.carbon_reporting.main import CarbonCalculator
    calc = CarbonCalculator()
    report = calc.calculate_shipment(
        shipment_id="SH_TEST", vehicle_id="VH_TEST",
        fuel_consumed_liters=200.0, distance_km=600.0,
    )
    assert report.direct_co2e_kg == pytest_approx(200 * 2.68, rel=0.01)
    assert report.iso_14083_compliant is True


@test("RL environment")
def test_rl_env():
    from services.mode_ddr.rl.environment import SynapseLogisticsEnv
    env = SynapseLogisticsEnv(n_vehicles=5, n_drivers=5, n_shipments=10, n_suppliers=5)
    obs, info = env.reset(seed=42)
    assert len(obs) == env.state_dim
    assert all(0.0 <= v <= 1.0 for v in obs)
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info2 = env.step(action)
    assert len(obs2) == env.state_dim
    assert isinstance(reward, float)


@test("Digital twin simulation (mini)")
def test_simulation():
    from ml.simulation.run_simulation import SynapseDigitalTwin
    sim = SynapseDigitalTwin(n_trucks=5, n_drivers=5, n_shipments=10, seed=42)
    results = sim.run(duration_hours=2.0, verbose=False)
    assert "on_time_performance" in results
    assert 0.0 <= results["on_time_performance"] <= 1.0


@test("Blockchain hash chaining")
def test_audit():
    from shared.utils.helpers import compute_event_hash
    h0 = compute_event_hash({"event": "GENESIS"}, "")
    h1 = compute_event_hash({"event": "LOAD_ASSIGNMENT"}, h0)
    h2 = compute_event_hash({"event": "HOS_LOG"}, h1)
    h1_tampered = compute_event_hash({"event": "LOAD_ASSIGNMENT_TAMPERED"}, h0)
    h2_valid = compute_event_hash({"event": "HOS_LOG"}, h1)
    h2_broken = compute_event_hash({"event": "HOS_LOG"}, h1_tampered)
    assert h2_valid != h2_broken  # Tamper detected!


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SYNAPSE Quick Start Test Suite")
    print("="*60)

    for name, result in RESULTS.items():
        print(f"  {result:<6} {name}")

    passed = sum(1 for r in RESULTS.values() if r.startswith("✅"))
    total = len(RESULTS)
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed")

    if passed < total:
        print("\nFailed tests (install dependencies first):")
        print("  pip install -r requirements.txt")
    else:
        print("\nAll tests passed! SYNAPSE is ready to run.")
    print("="*60)
