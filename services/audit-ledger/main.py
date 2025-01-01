"""
SYNAPSE Audit Ledger Service
Immutable blockchain-style compliance event logging.
Hash-chained records for FMCSA HOS, ISO 14083 carbon, and load assignment events.
7-year retention with tamper-evident SHA-256 chaining.
"""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from shared.schemas.models import ComplianceEvent
from shared.utils.helpers import KafkaSubscriber, SynapseCache, configure_logging, compute_event_hash
from shared.config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

EVENTS_LOGGED = Counter("audit_events_total", "Audit events logged", ["event_type"])
CHAIN_LENGTH = Counter("audit_chain_length_total", "Total events in chain")

cache: SynapseCache = None
_chain_head_hash: str = "GENESIS"
_event_sequence: List[ComplianceEvent] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global cache
    configure_logging("audit-ledger")
    cache = SynapseCache(settings.redis_url)
    await cache.connect()

    # Load chain head from cache
    head = await cache.get_json("audit:chain:head")
    if head:
        global _chain_head_hash
        _chain_head_hash = head.get("hash", "GENESIS")

    asyncio.create_task(_consume_compliance_events())
    logger.info("audit_ledger_started", chain_head=_chain_head_hash[:16])
    yield
    await cache.disconnect()


app = FastAPI(
    title="SYNAPSE Audit Ledger",
    description="Hash-chained immutable compliance event log",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


async def _log_event(event: ComplianceEvent) -> ComplianceEvent:
    """Append event to blockchain-style hash chain."""
    global _chain_head_hash

    # Compute hash linking to previous event
    event_hash = compute_event_hash(event.model_dump(), _chain_head_hash)
    event.hash_previous = _chain_head_hash
    event.digital_signature = event_hash
    event.verified = True

    # Update chain head
    _chain_head_hash = event_hash
    await cache.set_json("audit:chain:head", {
        "hash": _chain_head_hash,
        "event_id": str(event.event_id),
        "timestamp": event.timestamp.isoformat(),
    }, ttl=86400 * 365 * 7)  # 7-year retention

    # Store event
    await cache.set_json(
        f"audit:event:{event.event_id}",
        event.model_dump(),
        ttl=86400 * 365 * 7,
    )

    # Append to in-memory chain (production: database with index)
    _event_sequence.append(event)
    if len(_event_sequence) > 10000:
        _event_sequence.pop(0)  # Bound memory usage

    EVENTS_LOGGED.labels(event_type=event.event_type).inc()
    CHAIN_LENGTH.inc()

    logger.info("audit_event_logged",
                event_id=str(event.event_id),
                event_type=event.event_type,
                hash=event_hash[:16])
    return event


async def _consume_compliance_events() -> None:
    """Consume compliance events from Kafka and append to chain."""
    sub = KafkaSubscriber(
        topics=["synapse.compliance.events"],
        group_id="audit-ledger-svc",
    )
    await sub.start()
    async for msg in sub.messages():
        try:
            event = ComplianceEvent(**msg)
            await _log_event(event)
        except Exception as e:
            logger.error("audit_log_error", error=str(e))


@app.post("/events/log")
async def log_event(event: ComplianceEvent) -> ComplianceEvent:
    """Manually log a compliance event to the audit chain."""
    return await _log_event(event)


@app.get("/events/{event_id}")
async def get_event(event_id: str) -> ComplianceEvent:
    """Retrieve a specific compliance event by ID."""
    cached = await cache.get_json(f"audit:event:{event_id}")
    if not cached:
        raise HTTPException(status_code=404, detail="Event not found")
    return ComplianceEvent(**cached)


@app.get("/events")
async def list_recent_events(limit: int = 50) -> List[ComplianceEvent]:
    """List recent compliance events."""
    return _event_sequence[-limit:]


@app.get("/chain/verify")
async def verify_chain_integrity() -> dict:
    """Verify integrity of the hash chain (tamper detection)."""
    if len(_event_sequence) < 2:
        return {"verified": True, "chain_length": len(_event_sequence),
                "message": "Insufficient events to verify chain"}

    broken_at = None
    for i in range(1, len(_event_sequence)):
        current = _event_sequence[i]
        previous = _event_sequence[i - 1]

        # Recompute expected hash
        expected_hash = compute_event_hash(
            previous.model_dump(), previous.hash_previous or "GENESIS"
        )
        if current.hash_previous != expected_hash:
            broken_at = i
            break

    return {
        "verified": broken_at is None,
        "chain_length": len(_event_sequence),
        "chain_head_hash": _chain_head_hash[:32],
        "broken_at_index": broken_at,
        "message": "Chain integrity verified" if broken_at is None else f"Chain broken at index {broken_at}",
    }


@app.get("/chain/head")
async def get_chain_head() -> dict:
    """Get current chain head hash."""
    head = await cache.get_json("audit:chain:head")
    return {
        "chain_head_hash": _chain_head_hash,
        "chain_length": len(_event_sequence),
        "last_event": head,
    }


@app.get("/compliance/summary")
async def compliance_summary() -> dict:
    """Aggregate compliance status summary."""
    hos_events = [e for e in _event_sequence if "HOS" in e.event_type]
    carbon_events = [e for e in _event_sequence if "CARBON" in e.event_type]
    action_events = [e for e in _event_sequence if "ACTION" in e.event_type]

    return {
        "total_events": len(_event_sequence),
        "hos_events": len(hos_events),
        "carbon_events": len(carbon_events),
        "action_events": len(action_events),
        "chain_head_hash": _chain_head_hash[:32],
        "retention_years": 7,
        "standards": ["FMCSA HOS", "ISO 14083", "GLEC 3.0"],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "service": "audit-ledger",
            "chain_length": len(_event_sequence), "chain_head": _chain_head_hash[:16]}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8011, reload=True)
