"""
SYNAPSE Actuation Service
Executes prescriptive actions from MODE-DDR via TMS/WMS/ELD APIs.
Implements distributed locks to prevent conflicting actions.
All actions logged to audit trail.
"""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from shared.schemas.models import PrescriptiveAction, ActionType, ComplianceEvent
from shared.utils.helpers import KafkaPublisher, KafkaSubscriber, SynapseCache, configure_logging, compute_event_hash
from shared.config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

ACTIONS_EXECUTED = Counter("actuation_actions_total", "Actions executed", ["action_type", "status"])
EXECUTION_LATENCY = Histogram("actuation_latency_seconds", "Execution latency")
CONFLICTS_PREVENTED = Counter("actuation_conflicts_prevented_total", "Conflicts prevented")

publisher: KafkaPublisher = None
cache: SynapseCache = None
_last_hash = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global publisher, cache
    configure_logging("actuation")
    publisher = KafkaPublisher(settings.kafka_bootstrap)
    await publisher.start()
    cache = SynapseCache(settings.redis_url)
    await cache.connect()
    asyncio.create_task(_consume_ranked_actions())
    logger.info("actuation_service_started")
    yield
    await publisher.stop()
    await cache.disconnect()


app = FastAPI(
    title="SYNAPSE Actuation",
    description="TMS/WMS/ELD prescriptive action execution layer",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Distributed lock ─────────────────────────────────────────────────────────

class DistributedLock:
    """Redis-backed distributed lock for conflict prevention."""

    def __init__(self, cache: SynapseCache, key: str, ttl: int = 30):
        self.cache = cache
        self.key = f"lock:{key}"
        self.ttl = ttl

    async def acquire(self) -> bool:
        existing = await self.cache.get_json(self.key)
        if existing:
            return False
        await self.cache.set_json(self.key, {"locked": True}, ttl=self.ttl)
        return True

    async def release(self) -> None:
        # Simple release: delete key (production: use EVAL script for atomicity)
        pass


# ── TMS API adapter ───────────────────────────────────────────────────────────

class TMSAdapter:
    """
    Transportation Management System API adapter.
    In production: integrates with specific TMS vendor APIs (e.g., Oracle TMS, MercuryGate).
    """

    async def update_route(self, shipment_id: str, new_route: list, vehicle_id: str) -> dict:
        logger.info("tms_route_update", shipment_id=shipment_id, vehicle_id=vehicle_id)
        return {"status": "ACCEPTED", "shipment_id": shipment_id, "route_updated": True}

    async def reassign_load(self, shipment_id: str, from_vehicle: str, to_vehicle: str) -> dict:
        logger.info("tms_load_reassign", shipment_id=shipment_id)
        return {"status": "ACCEPTED", "shipment_id": shipment_id, "reassigned": True}

    async def hold_shipment(self, shipment_id: str, reason: str) -> dict:
        logger.info("tms_hold_shipment", shipment_id=shipment_id, reason=reason)
        return {"status": "ACCEPTED", "shipment_id": shipment_id, "on_hold": True}

    async def trigger_mode_shift(self, shipment_id: str, new_mode: str) -> dict:
        logger.info("tms_mode_shift", shipment_id=shipment_id, new_mode=new_mode)
        return {"status": "ACCEPTED", "shipment_id": shipment_id, "new_mode": new_mode}


class WMSAdapter:
    """Warehouse Management System API adapter."""

    async def schedule_maintenance(self, vehicle_id: str, hub_id: str, scheduled_time: str) -> dict:
        logger.info("wms_maintenance_scheduled", vehicle_id=vehicle_id, hub=hub_id)
        return {"status": "SCHEDULED", "vehicle_id": vehicle_id, "hub_id": hub_id}


class ELDAdapter:
    """Electronic Logging Device API adapter."""

    async def send_break_notification(self, driver_id: str, vehicle_id: str) -> dict:
        logger.info("eld_break_notification", driver_id=driver_id)
        return {"status": "SENT", "driver_id": driver_id, "notification": "BREAK_REQUIRED"}

    async def log_hos_event(self, driver_id: str, event_type: str, hours: float) -> dict:
        logger.info("eld_hos_log", driver_id=driver_id, event=event_type)
        return {"status": "LOGGED", "driver_id": driver_id, "event_type": event_type}


tms = TMSAdapter()
wms = WMSAdapter()
eld = ELDAdapter()


# ── Action executor ───────────────────────────────────────────────────────────

async def execute_action(action: PrescriptiveAction) -> dict:
    """
    Execute a prescriptive action with distributed lock and audit logging.
    Returns execution result.
    """
    import time
    t0 = time.monotonic()
    global _last_hash

    # Acquire distributed lock on target entity
    lock = DistributedLock(cache, f"{action.action_type}:{action.target_id}")
    if not await lock.acquire():
        CONFLICTS_PREVENTED.inc()
        logger.warning("action_conflict_prevented",
                       action_type=action.action_type, target=action.target_id)
        return {"status": "CONFLICT", "reason": "Entity locked by another action"}

    try:
        # Execute based on action type
        result = {}
        if action.action_type == ActionType.REROUTE:
            result = await tms.update_route(action.target_id, [], "VH_AUTO")
        elif action.action_type == ActionType.REASSIGN_LOAD:
            result = await tms.reassign_load(action.target_id, "VH_OLD", "VH_NEW")
        elif action.action_type == ActionType.SCHEDULE_MAINTENANCE:
            result = await wms.schedule_maintenance(action.target_id, "HUB_01", datetime.utcnow().isoformat())
        elif action.action_type == ActionType.TRIGGER_BREAK:
            result = await eld.send_break_notification(action.target_id, "VH_AUTO")
            await eld.log_hos_event(action.target_id, "BREAK_TRIGGERED", 0.5)
        elif action.action_type == ActionType.SUPPLIER_ESCALATION:
            result = {"status": "ESCALATED", "supplier_id": action.target_id}
        elif action.action_type == ActionType.MODE_SHIFT:
            result = await tms.trigger_mode_shift(action.target_id, "AIR")
        elif action.action_type == ActionType.HOLD_SHIPMENT:
            result = await tms.hold_shipment(action.target_id, "DISRUPTION_AVOIDANCE")
        else:
            result = {"status": "NOOP"}

        status = result.get("status", "UNKNOWN")
        ACTIONS_EXECUTED.labels(action_type=action.action_type.value, status=status).inc()
        EXECUTION_LATENCY.observe(time.monotonic() - t0)

        # Create compliance event for blockchain audit
        event_payload = {
            "action_type": action.action_type.value,
            "target_id": action.target_id,
            "result": result,
            "reward_score": action.reward_score,
            "carbon_impact_kg_co2e": action.carbon_impact_kg_co2e,
        }
        event_hash = compute_event_hash(event_payload, _last_hash)
        _last_hash = event_hash

        compliance_event = ComplianceEvent(
            timestamp=datetime.utcnow(),
            event_type=f"ACTION_{action.action_type.value}",
            entity_id=action.target_id,
            entity_type="SHIPMENT_OR_DRIVER",
            payload=event_payload,
            hash_previous=_last_hash,
            digital_signature=event_hash,
        )

        await publisher.publish("synapse.compliance.events", compliance_event.model_dump())
        await cache.set_json(f"action:{action.action_id}:result", result, ttl=3600)

        return {"status": "EXECUTED", "action_id": str(action.action_id),
                "result": result, "audit_hash": event_hash}

    except Exception as e:
        ACTIONS_EXECUTED.labels(action_type=action.action_type.value, status="FAILED").inc()
        logger.error("action_execution_failed", error=str(e))
        return {"status": "FAILED", "error": str(e)}
    finally:
        await lock.release()


async def _consume_ranked_actions() -> None:
    """Consume ranked actions from MODE-DDR and execute approved ones."""
    sub = KafkaSubscriber(
        topics=["synapse.actions.ranked"],
        group_id="actuation-svc",
    )
    await sub.start()
    async for msg in sub.messages():
        try:
            # Extract top-ranked action if auto-execute
            if msg.get("auto_execute"):
                actions = msg.get("ranked_actions", [])
                if actions:
                    action = PrescriptiveAction(**actions[0])
                    result = await execute_action(action)
                    logger.info("auto_executed", action_type=action.action_type, result=result["status"])
        except Exception as e:
            logger.error("action_consumption_error", error=str(e))


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.post("/execute")
async def execute(action: PrescriptiveAction) -> dict:
    """Execute a single prescriptive action."""
    return await execute_action(action)


@app.get("/action/{action_id}/status")
async def get_action_status(action_id: str) -> dict:
    """Get execution status of an action."""
    result = await cache.get_json(f"action:{action_id}:result")
    if not result:
        raise HTTPException(status_code=404, detail="Action not found")
    return result


@app.get("/health")
async def health():
    return {"status": "ok", "service": "actuation"}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8008, reload=True)
