"""
SYNAPSE Supply Chain Service
Multivariate Transformer for Tier 1-3 delay forecasting.
MAPE targets: Ocean 17%, Port 14%, Inland TL 9%.
"""
from __future__ import annotations
import asyncio, time, uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import structlog
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
sys.path.insert(0, "/app")
from shared.schemas.models import SupplyChainDisruptionForecast, DisruptionType
from shared.utils.helpers import KafkaProducerClient, RedisClient, PREDICTION_LATENCY
from shared.config.settings import get_settings

log = structlog.get_logger()
settings = get_settings()

# ── Multivariate Transformer Forecaster (Simulated) ───────────────────────────

class MultivariateTransformerForecaster:
    """
    Transformer model for supply chain delay prediction.
    Attention mechanism captures long-range dependencies across Tier 1-3 events.
    In production: trained on vessel positions, customs status, supplier milestones.
    MAPE: Ocean 17%, Port 14%, Inland TL 9% (vs 22-29% baseline).
    """

    # Per lane-type baseline MAPE (paper Table in Appendix B.2)
    MAPE_BY_LANE = {"ocean": 0.17, "port": 0.14, "inland_tl": 0.09}

    _supplier_history: dict[str, list] = {}

    def predict_delay(
        self,
        supplier_id: str,
        tier: int,
        lane_type: str,
        current_lead_time_days: float,
        weather_index: float = 0.0,
        port_congestion_index: float = 0.0,
        vessel_schedule_risk: float = 0.0,
    ) -> dict:
        """Predict probability and magnitude of supply chain delay."""
        # Tier-based risk amplification (Tier 2/3 have less visibility)
        tier_visibility = {1: 1.0, 2: 0.8, 3: 0.6}.get(tier, 0.6)

        # Base delay probability from risk factors
        base_risk = (
            0.3 * weather_index +
            0.4 * port_congestion_index +
            0.3 * vessel_schedule_risk
        )

        # Tier 2/3 propagation amplification
        propagation_factor = 1.0 + (tier - 1) * 0.2
        delay_probability = float(np.clip(base_risk * propagation_factor, 0, 1))

        # Expected delay magnitude (days)
        if delay_probability > 0.5:
            estimated_delay = current_lead_time_days * 0.15 * delay_probability
        else:
            estimated_delay = 0.0

        mape = self.MAPE_BY_LANE.get(lane_type, 0.17)
        # Simulate Transformer attention confidence
        confidence = float(np.clip(tier_visibility * (1 - mape), 0, 1))

        # Tier 2/3 propagation risk estimates
        propagation_t2 = delay_probability * 0.7 if tier == 1 else delay_probability
        propagation_t3 = delay_probability * 0.5 if tier <= 2 else delay_probability

        return {
            "supplier_id": supplier_id,
            "tier": tier,
            "delay_probability": round(delay_probability, 4),
            "estimated_delay_days": round(estimated_delay, 2),
            "confidence": round(confidence, 4),
            "propagation_risk_tier2": round(propagation_t2, 4),
            "propagation_risk_tier3": round(propagation_t3, 4),
            "mape": mape,
        }

    def detect_disruption_type(
        self, weather: float, port: float, vessel: float, labor: float
    ) -> DisruptionType:
        scores = {
            DisruptionType.WEATHER: weather,
            DisruptionType.PORT_CONGESTION: port,
            DisruptionType.SUPPLIER_DELAY: vessel,
            DisruptionType.LABOR_SHORTAGE: labor,
        }
        return max(scores, key=scores.get)


# ── Multi-Tier Correlator ─────────────────────────────────────────────────────

class MultiTierCorrelator:
    """
    Correlates disruptions across Tier 1-3 nodes.
    Detects upstream risk propagation before it impacts Tier 1 / fleet execution.
    """
    _tier3_risk_buffer: list[dict] = []

    def propagate_risk(self, forecast: SupplyChainDisruptionForecast) -> list[dict]:
        """Generate downstream propagation alerts if Tier 3 risk is high."""
        alerts = []
        if forecast.propagation_risk_tier3 > 0.6:
            alerts.append({
                "alert_type": "UPSTREAM_RISK_TIER3",
                "source_suppliers": forecast.affected_supplier_ids,
                "propagation_probability": forecast.propagation_risk_tier3,
                "recommended_buffer_days": round(forecast.estimated_delay_days * 1.5, 1),
                "timestamp": datetime.utcnow().isoformat(),
            })
        if forecast.propagation_risk_tier2 > 0.7:
            alerts.append({
                "alert_type": "UPSTREAM_RISK_TIER2",
                "source_suppliers": forecast.affected_supplier_ids,
                "propagation_probability": forecast.propagation_risk_tier2,
                "recommended_buffer_days": round(forecast.estimated_delay_days * 1.2, 1),
                "timestamp": datetime.utcnow().isoformat(),
            })
        return alerts


# ── Service State ─────────────────────────────────────────────────────────────

forecaster = MultivariateTransformerForecaster()
correlator = MultiTierCorrelator()
kafka_producer: Optional[KafkaProducerClient] = None
redis_client: Optional[RedisClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global kafka_producer, redis_client
    kafka_producer = KafkaProducerClient(settings.kafka_bootstrap, "supply-chain")
    await kafka_producer.start()
    redis_client = RedisClient(settings.redis_url)
    await redis_client.connect()
    log.info("supply_chain_service_started")
    yield
    await kafka_producer.stop()
    await redis_client.disconnect()

app = FastAPI(title="SYNAPSE Supply Chain Service", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class DelayForecastRequest(Exception): pass

from pydantic import BaseModel

class ForecastRequest(BaseModel):
    supplier_ids: list[str]
    tier: int = 1
    lane_type: str = "inland_tl"
    current_lead_time_days: float = 5.0
    weather_index: float = 0.0
    port_congestion_index: float = 0.0
    vessel_schedule_risk: float = 0.0
    labor_shortage_index: float = 0.0

@app.get("/health")
async def health(): return {"status": "ok", "service": "supply-chain"}

@app.post("/forecast/disruption", response_model=SupplyChainDisruptionForecast)
async def forecast_disruption(req: ForecastRequest):
    """Forecast supply chain disruption probability and magnitude."""
    t0 = time.monotonic()

    raw = forecaster.predict_delay(
        supplier_id=req.supplier_ids[0] if req.supplier_ids else "UNKNOWN",
        tier=req.tier, lane_type=req.lane_type,
        current_lead_time_days=req.current_lead_time_days,
        weather_index=req.weather_index,
        port_congestion_index=req.port_congestion_index,
        vessel_schedule_risk=req.vessel_schedule_risk,
    )
    disruption_type = forecaster.detect_disruption_type(
        req.weather_index, req.port_congestion_index,
        req.vessel_schedule_risk, req.labor_shortage_index
    )

    forecast = SupplyChainDisruptionForecast(
        timestamp=datetime.utcnow(),
        affected_supplier_ids=req.supplier_ids,
        disruption_type=disruption_type,
        delay_probability=raw["delay_probability"],
        estimated_delay_days=raw["estimated_delay_days"],
        confidence=raw["confidence"],
        propagation_risk_tier2=raw["propagation_risk_tier2"],
        propagation_risk_tier3=raw["propagation_risk_tier3"],
        mape=raw["mape"],
    )

    latency = (time.monotonic() - t0) * 1000
    PREDICTION_LATENCY.labels(model="multivariate_transformer", service="supply-chain").observe(latency/1000)

    await kafka_producer.send_model("supplychain.predictions", forecast, key=forecast.forecast_id)

    propagation_alerts = correlator.propagate_risk(forecast)
    for alert in propagation_alerts:
        await kafka_producer.send("synapse.alerts.supplychain", alert)

    return forecast

@app.post("/forecast/batch")
async def forecast_batch(requests: list[ForecastRequest]):
    results = []
    for req in requests:
        raw = forecaster.predict_delay(
            req.supplier_ids[0] if req.supplier_ids else "UNKNOWN",
            req.tier, req.lane_type, req.current_lead_time_days,
            req.weather_index, req.port_congestion_index, req.vessel_schedule_risk,
        )
        results.append(raw)
    return {"forecasts": results, "count": len(results)}

@app.get("/model/performance")
async def model_performance():
    return {
        "model": "Multivariate Transformer",
        "mape_ocean": 0.17, "baseline_ocean": 0.17,
        "mape_port": 0.14, "baseline_port": 0.29,
        "mape_inland_tl": 0.09, "baseline_inland_tl": 0.22,
        "tier_coverage": [1, 2, 3],
    }
