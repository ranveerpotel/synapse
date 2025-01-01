"""
SYNAPSE Telemetry Ingestion Service
Ingests CAN bus + IoT sensor signals at 1-10Hz from 500 vehicles.
Validates, normalises, and publishes to Kafka topics partitioned by vehicle_id.
Target: <50ms ingestion latency, 5000 msgs/sec peak.
"""
from __future__ import annotations
import asyncio, json, time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional
import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sys, os
sys.path.insert(0, "/app")
from shared.schemas.models import CANBusSignal, FreightIoTSignal
from shared.utils.helpers import KafkaProducerClient, MESSAGES_PRODUCED
from shared.config.settings import get_settings

log = structlog.get_logger()
settings = get_settings()

# Kafka producer (global)
producer: Optional[KafkaProducerClient] = None

TOPIC_VEHICLE_TELEMETRY = "vehicle.telemetry.raw"
TOPIC_FREIGHT_IOT       = "freight.iot.raw"
TOPIC_ANOMALY_ALERTS    = "synapse.alerts.anomaly"

# ── Signal Normaliser ────────────────────────────────────────────────────────

class TelemetryNormaliser:
    """Min-max normalise CAN signals; flag out-of-range values."""

    BOUNDS = {
        "engine_rpm":        (0, 8000),
        "torque_nm":         (0, 3000),
        "oil_pressure_kpa":  (0, 700),
        "coolant_temp_c":    (-40, 150),
        "tire_pressure_fl_kpa": (0, 1000),
        "vibration_rms_g":   (0, 50),
        "fuel_level_pct":    (0, 100),
        "speed_kmh":         (0, 200),
    }

    @classmethod
    def normalise(cls, signal: CANBusSignal) -> dict:
        raw = signal.model_dump()
        normalised = {}
        anomalies = []
        for field, (lo, hi) in cls.BOUNDS.items():
            val = raw.get(field, 0.0)
            norm = (val - lo) / (hi - lo)
            normalised[field] = round(max(0.0, min(1.0, norm)), 6)
            if val < lo or val > hi:
                anomalies.append({"field": field, "value": val, "bounds": [lo, hi]})
        raw["_normalised"] = normalised
        raw["_anomalies"]  = anomalies
        raw["_ingested_at"] = datetime.utcnow().isoformat()
        return raw

# ── Fuel Anomaly Detector ────────────────────────────────────────────────────

class FuelAnomalyDetector:
    """
    Simple threshold-based fuel siphoning / theft detector.
    Flags when fuel drops > 5% without engine running.
    Full implementation uses Isolation Forest in fleet-health-svc.
    """
    _prev: dict[str, float] = {}

    @classmethod
    def check(cls, signal: CANBusSignal) -> Optional[dict]:
        prev_level = cls._prev.get(signal.vehicle_id)
        cls._prev[signal.vehicle_id] = signal.fuel_level_pct
        if prev_level is None:
            return None
        drop = prev_level - signal.fuel_level_pct
        if drop > 5.0 and signal.engine_rpm < 200:
            return {
                "vehicle_id": signal.vehicle_id,
                "alert_type": "FUEL_ANOMALY",
                "severity": "HIGH",
                "drop_pct": round(drop, 2),
                "timestamp": signal.timestamp.isoformat(),
            }
        return None

# ── FastAPI App ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global producer
    producer = KafkaProducerClient(settings.kafka_bootstrap, "telemetry-ingestion")
    await producer.start()
    log.info("telemetry_ingestion_started")
    yield
    await producer.stop()

app = FastAPI(title="SYNAPSE Telemetry Ingestion", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

normaliser = TelemetryNormaliser()
fuel_detector = FuelAnomalyDetector()

@app.get("/health")
async def health(): return {"status": "ok", "service": "telemetry-ingestion"}

@app.post("/ingest/canbus", status_code=202)
async def ingest_canbus(signal: CANBusSignal, bg: BackgroundTasks):
    """Ingest a CAN bus signal. Returns immediately; processing is async."""
    bg.add_task(_process_canbus, signal)
    return {"accepted": True, "vehicle_id": signal.vehicle_id}

@app.post("/ingest/canbus/batch", status_code=202)
async def ingest_canbus_batch(signals: list[CANBusSignal]):
    """Batch ingest for high-frequency telemetry (1-10Hz per vehicle)."""
    tasks = [_process_canbus(s) for s in signals]
    await asyncio.gather(*tasks)
    return {"accepted": len(signals)}

@app.post("/ingest/freight-iot", status_code=202)
async def ingest_freight_iot(signal: FreightIoTSignal, bg: BackgroundTasks):
    bg.add_task(_process_freight_iot, signal)
    return {"accepted": True, "shipment_id": signal.shipment_id}

async def _process_canbus(signal: CANBusSignal):
    t0 = time.monotonic()
    enriched = normaliser.normalise(signal)
    alert = fuel_detector.check(signal)
    await producer.send(TOPIC_VEHICLE_TELEMETRY, enriched, key=signal.vehicle_id)
    if alert:
        await producer.send(TOPIC_ANOMALY_ALERTS, alert, key=signal.vehicle_id)
    latency_ms = (time.monotonic() - t0) * 1000
    if latency_ms > 50:
        log.warning("ingestion_latency_high", vehicle_id=signal.vehicle_id, ms=round(latency_ms, 1))

async def _process_freight_iot(signal: FreightIoTSignal):
    payload = signal.model_dump()
    payload["_ingested_at"] = datetime.utcnow().isoformat()
    await producer.send(TOPIC_FREIGHT_IOT, payload, key=signal.shipment_id)

@app.get("/metrics/summary")
async def metrics_summary():
    return {
        "service": "telemetry-ingestion",
        "target_latency_ms": 50,
        "target_throughput_msgs_per_sec": 5000,
        "topics": [TOPIC_VEHICLE_TELEMETRY, TOPIC_FREIGHT_IOT, TOPIC_ANOMALY_ALERTS],
    }
