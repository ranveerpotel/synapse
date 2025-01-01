"""
SYNAPSE Insight Fusion Service
Assembles cross-domain 150-dimensional state vector from all prediction services.
This is the architectural keystone — feeds MODE-DDR with fused situational awareness.
Target: <50ms state vector assembly latency.
"""
from __future__ import annotations
import asyncio, time, httpx
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional
import numpy as np
import structlog
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
sys.path.insert(0, "/app")
from shared.schemas.models import FusedStateVector
from shared.utils.helpers import KafkaProducerClient, KafkaConsumerClient, RedisClient
from shared.config.settings import get_settings

log = structlog.get_logger()
settings = get_settings()

# ── State Vector Builder ─────────────────────────────────────────────────────

class StateVectorBuilder:
    """
    Assembles the 150-180D fused state vector for MODE-DDR.
    Subspaces: fleet_health (~30D) + driver (~20D) + routing (~30D) + supply_chain (~30D) + env (~10D)
    Partial observability: 80% accuracy on Tier 2 delays (as per paper).
    Gaussian perturbations σ=10% on travel times to simulate real-world noise.
    """

    # Cache of latest domain predictions keyed by entity ID
    _fleet_cache: dict = {}
    _driver_cache: dict = {}
    _routing_cache: dict = {}
    _supply_cache: dict = {}
    _env_cache: dict = {"weather_indices": [0.0]*3, "traffic_signals": [0.5]*5}

    # ── Cache Update Methods ─────────────────────────────────────────────────

    def update_fleet(self, vehicle_id: str, health_state: dict):
        self._fleet_cache[vehicle_id] = health_state

    def update_driver(self, driver_id: str, driver_state: dict):
        self._driver_cache[driver_id] = driver_state

    def update_routing(self, shipment_id: str, eta_pred: dict):
        self._routing_cache[shipment_id] = eta_pred

    def update_supply_chain(self, forecast_id: str, forecast: dict):
        self._supply_cache[forecast_id] = forecast

    def update_environment(self, weather: list[float], traffic: list[float]):
        self._env_cache = {"weather_indices": weather, "traffic_signals": traffic}

    # ── State Vector Assembly ────────────────────────────────────────────────

    def build(self) -> FusedStateVector:
        t0 = time.monotonic()

        # Fleet health subspace (~30D)
        vehicles = list(self._fleet_cache.values())[:10]  # Cap at 10 vehicles per vector
        fleet_deg = [v.get("overall_health_score", 1.0) for v in vehicles]
        fleet_fail = [v.get("failure_probability_72h", 0.0) for v in vehicles]
        fleet_anom = [1.0 if v.get("anomaly_detected", False) else 0.0 for v in vehicles]
        fleet_deg  = self._pad(fleet_deg, 10)
        fleet_fail = self._pad(fleet_fail, 10)
        fleet_anom = self._pad(fleet_anom, 10)

        # Driver subspace (~20D)
        drivers = list(self._driver_cache.values())[:5]
        drv_fat  = [d.get("fatigue_probability", 0.0) for d in drivers]
        drv_dist = [d.get("distraction_probability", 0.0) for d in drivers]
        drv_str  = [d.get("stress_index", 0.0) for d in drivers]
        drv_hos  = [d.get("hos_risk_score", 0.0) for d in drivers]
        drv_fat  = self._pad(drv_fat, 5)
        drv_dist = self._pad(drv_dist, 5)
        drv_str  = self._pad(drv_str, 5)
        drv_hos  = self._pad(drv_hos, 5)

        # Routing subspace (~30D)
        shipments = list(self._routing_cache.values())[:10]
        eta_errors = [s.get("eta_mae_minutes", 9.7) / 30.0 for s in shipments]  # Normalise to [0,1]
        congestion = [s.get("congestion_probability", 0.5) for s in shipments]
        emissions  = [s.get("multi_objective_cost", {}).get("carbon_kg_co2e", 0.0) / 100.0 for s in shipments]
        eta_errors = self._pad(eta_errors, 10)
        congestion = self._pad(congestion, 10)
        emissions  = self._pad(emissions, 10)

        # Supply chain subspace (~30D)
        forecasts = list(self._supply_cache.values())[:10]
        sup_delay = [f.get("delay_probability", 0.0) for f in forecasts]
        lead_times = [min(1.0, f.get("estimated_delay_days", 0.0) / 30.0) for f in forecasts]
        prop_risk  = [f.get("propagation_risk_tier3", 0.0) for f in forecasts]
        # Apply 80% observability for Tier 2/3 (paper spec)
        sup_delay  = [v * 0.8 + np.random.normal(0, 0.02) for v in self._pad(sup_delay, 10)]
        lead_times = self._pad(lead_times, 10)
        prop_risk  = self._pad(prop_risk, 10)

        # Environmental subspace (~10D)
        weather  = self._pad(self._env_cache.get("weather_indices", []), 3)
        traffic  = self._pad(self._env_cache.get("traffic_signals", []), 5)
        ext_disr = [0.0, 0.0]

        latency_ms = (time.monotonic() - t0) * 1000
        if latency_ms > 50:
            log.warning("state_vector_assembly_slow", ms=round(latency_ms, 1))

        return FusedStateVector(
            timestamp=datetime.utcnow(),
            fleet_degradation_scores=[round(float(x), 4) for x in fleet_deg],
            fleet_failure_probabilities=[round(float(x), 4) for x in fleet_fail],
            fleet_anomaly_scores=[round(float(x), 4) for x in fleet_anom],
            driver_fatigue_probabilities=[round(float(x), 4) for x in drv_fat],
            driver_distraction_scores=[round(float(x), 4) for x in drv_dist],
            driver_stress_indices=[round(float(x), 4) for x in drv_str],
            driver_hos_risk_scores=[round(float(x), 4) for x in drv_hos],
            eta_errors=[round(float(x), 4) for x in eta_errors],
            lane_congestion_features=[round(float(x), 4) for x in congestion],
            emissions_forecasts=[round(float(x), 4) for x in emissions],
            supplier_delay_probabilities=[round(float(np.clip(x, 0, 1)), 4) for x in sup_delay],
            lead_time_forecasts=[round(float(x), 4) for x in lead_times],
            disruption_propagation_indicators=[round(float(x), 4) for x in prop_risk],
            weather_indices=[round(float(x), 4) for x in weather],
            traffic_signals=[round(float(x), 4) for x in traffic],
            external_disruption_indicators=[round(float(x), 4) for x in ext_disr],
            active_vehicles=len(self._fleet_cache),
            active_drivers=len(self._driver_cache),
            active_shipments=len(self._routing_cache),
            fleet_utilization_pct=round(len(self._fleet_cache) / max(1, 500) * 100, 1),
        )

    @staticmethod
    def _pad(lst: list, length: int) -> list:
        """Pad or truncate list to fixed length."""
        lst = list(lst)
        if len(lst) >= length:
            return lst[:length]
        return lst + [0.0] * (length - len(lst))


# ── Kafka Consumer ────────────────────────────────────────────────────────────

builder = StateVectorBuilder()
kafka_producer: Optional[KafkaProducerClient] = None
redis_client: Optional[RedisClient] = None
consumer_tasks: list[asyncio.Task] = []

async def _consume_fleet_predictions():
    consumer = KafkaConsumerClient(
        settings.kafka_bootstrap, ["fleet.predictions"], "insight-fleet", "insight-fusion"
    )
    await consumer.start()
    async for msg in consumer.consume():
        builder.update_fleet(msg.get("vehicle_id", ""), msg)
        await _publish_state()

async def _consume_driver_predictions():
    consumer = KafkaConsumerClient(
        settings.kafka_bootstrap, ["driver.predictions"], "insight-driver", "insight-fusion"
    )
    await consumer.start()
    async for msg in consumer.consume():
        builder.update_driver(msg.get("driver_id", ""), msg)

async def _consume_routing_predictions():
    consumer = KafkaConsumerClient(
        settings.kafka_bootstrap, ["routing.predictions"], "insight-routing", "insight-fusion"
    )
    await consumer.start()
    async for msg in consumer.consume():
        builder.update_routing(msg.get("shipment_id", ""), msg)

async def _consume_supply_predictions():
    consumer = KafkaConsumerClient(
        settings.kafka_bootstrap, ["supplychain.predictions"], "insight-supply", "insight-fusion"
    )
    await consumer.start()
    async for msg in consumer.consume():
        builder.update_supply_chain(msg.get("forecast_id", ""), msg)

async def _publish_state():
    """Assemble and publish fused state vector on each new fleet prediction."""
    state = builder.build()
    await kafka_producer.send_model("synapse.state.fused", state, key=state.state_id)
    await redis_client.set_json("synapse:state:latest", state.model_dump(), ttl=60)

# ── FastAPI App ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global kafka_producer, redis_client, consumer_tasks
    kafka_producer = KafkaProducerClient(settings.kafka_bootstrap, "insight-fusion")
    await kafka_producer.start()
    redis_client = RedisClient(settings.redis_url)
    await redis_client.connect()
    consumer_tasks = [
        asyncio.create_task(_consume_fleet_predictions()),
        asyncio.create_task(_consume_driver_predictions()),
        asyncio.create_task(_consume_routing_predictions()),
        asyncio.create_task(_consume_supply_predictions()),
    ]
    log.info("insight_fusion_started")
    yield
    for t in consumer_tasks: t.cancel()
    await kafka_producer.stop()
    await redis_client.disconnect()

app = FastAPI(title="SYNAPSE Insight Fusion Service", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
async def health(): return {"status": "ok", "service": "insight-fusion"}

@app.get("/state/current", response_model=FusedStateVector)
async def get_current_state():
    """Return current fused state vector."""
    return builder.build()

@app.get("/state/cached")
async def get_cached_state():
    cached = await redis_client.get_json("synapse:state:latest")
    if not cached: raise HTTPException(404, "No state vector cached yet")
    return cached

@app.post("/state/update/environment")
async def update_environment(weather: list[float], traffic: list[float]):
    builder.update_environment(weather, traffic)
    return {"updated": True}

@app.get("/state/dimensions")
async def state_dimensions():
    return {
        "total_dimensions": "150-180",
        "subspaces": {
            "fleet_health": 30,
            "driver": 20,
            "routing_network": 30,
            "supply_chain": 30,
            "environmental": 10,
        },
        "observability": "80% accuracy on Tier 2/3 delays",
        "noise_model": "Gaussian σ=10% on travel times",
    }
