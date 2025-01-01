"""SYNAPSE Shared Utilities - Kafka, Redis, Prometheus, HOS compliance, Auth."""
from __future__ import annotations
import asyncio, hashlib, json, time
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Optional
import redis.asyncio as aioredis
import structlog
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from prometheus_client import Counter, Histogram, Gauge

log = structlog.get_logger()

MESSAGES_PRODUCED = Counter("synapse_kafka_messages_produced_total","Total Kafka messages produced",["topic","service"])
MESSAGES_CONSUMED = Counter("synapse_kafka_messages_consumed_total","Total Kafka messages consumed",["topic","service"])
PREDICTION_LATENCY = Histogram("synapse_prediction_latency_seconds","Model inference latency",["model","service"],buckets=[0.01,0.05,0.1,0.25,0.5,1.0,2.0,5.0])
ACTIVE_VEHICLES = Gauge("synapse_active_vehicles_total","Active vehicles in network")
ACTIVE_DRIVERS = Gauge("synapse_active_drivers_total","Active drivers on duty")

class KafkaProducerClient:
    def __init__(self, bootstrap_servers: str, service_name: str):
        self.bootstrap_servers = bootstrap_servers
        self.service_name = service_name
        self._producer: Optional[AIOKafkaProducer] = None

    async def start(self):
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode(),
            compression_type="gzip", acks="all", enable_idempotence=True)
        await self._producer.start()

    async def stop(self):
        if self._producer: await self._producer.stop()

    async def send(self, topic: str, value: dict, key: Optional[str] = None):
        if not self._producer: raise RuntimeError("Producer not started")
        await self._producer.send_and_wait(topic, value=value, key=key.encode() if key else None)
        MESSAGES_PRODUCED.labels(topic=topic, service=self.service_name).inc()

    async def send_model(self, topic: str, model_obj: Any, key: Optional[str] = None):
        await self.send(topic, model_obj.model_dump(), key=key)

class KafkaConsumerClient:
    def __init__(self, bootstrap_servers: str, topics: list, group_id: str, service_name: str):
        self.bootstrap_servers = bootstrap_servers
        self.topics = topics
        self.group_id = group_id
        self.service_name = service_name
        self._consumer: Optional[AIOKafkaConsumer] = None

    async def start(self):
        self._consumer = AIOKafkaConsumer(
            *self.topics, bootstrap_servers=self.bootstrap_servers, group_id=self.group_id,
            value_deserializer=lambda v: json.loads(v.decode()),
            auto_offset_reset="latest", enable_auto_commit=True)
        await self._consumer.start()

    async def stop(self):
        if self._consumer: await self._consumer.stop()

    async def consume(self) -> AsyncGenerator[dict, None]:
        if not self._consumer: raise RuntimeError("Consumer not started")
        async for msg in self._consumer:
            MESSAGES_CONSUMED.labels(topic=msg.topic, service=self.service_name).inc()
            yield msg.value

class RedisClient:
    def __init__(self, url: str, default_ttl: int = 300):
        self.url = url
        self.default_ttl = default_ttl
        self._client: Optional[aioredis.Redis] = None

    async def connect(self):
        self._client = await aioredis.from_url(self.url, encoding="utf-8", decode_responses=True)

    async def disconnect(self):
        if self._client: await self._client.aclose()

    async def set_json(self, key: str, value: Any, ttl: Optional[int] = None):
        await self._client.setex(key, ttl or self.default_ttl, json.dumps(value, default=str))

    async def get_json(self, key: str) -> Optional[dict]:
        raw = await self._client.get(key)
        return json.loads(raw) if raw else None

    async def acquire_lock(self, resource: str, ttl: int = 30) -> bool:
        return bool(await self._client.set(f"synapse:lock:{resource}", "1", nx=True, ex=ttl))

    async def release_lock(self, resource: str):
        await self._client.delete(f"synapse:lock:{resource}")

def compute_event_hash(payload: dict, previous_hash: Optional[str] = None) -> str:
    content = json.dumps(payload, sort_keys=True, default=str)
    if previous_hash: content = previous_hash + content
    return hashlib.sha256(content.encode()).hexdigest()

def check_hos_compliance(driving_h: float, on_duty_h: float, weekly_h: float) -> dict:
    return {
        "compliant": driving_h < 11.0 and on_duty_h < 14.0 and weekly_h < 70.0,
        "remaining_drive_hours": max(0.0, 11.0 - driving_h),
        "remaining_duty_hours": max(0.0, 14.0 - on_duty_h),
        "remaining_weekly_hours": max(0.0, 70.0 - weekly_h),
    }


# ── Alternate Kafka interfaces (used by newer services) ───────────────────────

class KafkaPublisher:
    """Async Kafka producer with simple publish() interface."""

    def __init__(self, bootstrap_servers: str):
        self._client = KafkaProducerClient(bootstrap_servers, "synapse")

    async def start(self):
        await self._client.start()

    async def stop(self):
        await self._client.stop()

    async def publish(self, topic: str, value: dict, key: Optional[str] = None):
        await self._client.send(topic, value, key=key)


class KafkaSubscriber:
    """Async Kafka consumer with async messages() generator interface."""

    def __init__(self, topics: list[str], group_id: str):
        from shared.config.settings import get_settings
        settings = get_settings()
        self._client = KafkaConsumerClient(
            settings.kafka_bootstrap, topics, group_id, group_id
        )

    async def start(self):
        await self._client.start()

    async def stop(self):
        await self._client.stop()

    async def messages(self) -> AsyncGenerator[dict, None]:
        async for msg in self._client.consume():
            yield msg


# ── Redis cache alias ─────────────────────────────────────────────────────────

class SynapseCache(RedisClient):
    """Redis-backed cache used by newer services (alias for RedisClient)."""
    pass


# ── Logging ───────────────────────────────────────────────────────────────────

def configure_logging(service_name: str) -> None:
    """Configure structlog for the given service."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer() if True else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


# ── JWT Auth ──────────────────────────────────────────────────────────────────

def create_access_token(data: dict) -> str:
    """Create a signed JWT access token."""
    from shared.config.settings import get_settings
    settings = get_settings()
    try:
        from jose import jwt
        payload = dict(data)
        payload["exp"] = datetime.utcnow() + timedelta(minutes=settings.jwt_expire_minutes)
        return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    except ImportError:
        import base64
        encoded = base64.urlsafe_b64encode(json.dumps(data).encode()).decode()
        sig = hashlib.sha256((encoded + settings.jwt_secret).encode()).hexdigest()[:16]
        return f"{encoded}.{sig}"


def verify_token(token: str) -> dict:
    """Verify a JWT token and return the payload claims."""
    from shared.config.settings import get_settings
    settings = get_settings()
    try:
        from jose import jwt, JWTError
        claims = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return dict(claims)
    except ImportError:
        import base64
        parts = token.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError("Invalid token format")
        encoded, sig = parts
        expected = hashlib.sha256((encoded + settings.jwt_secret).encode()).hexdigest()[:16]
        if sig != expected:
            raise ValueError("Invalid token signature")
        return json.loads(base64.urlsafe_b64decode(encoded + "==").decode())
    except Exception as exc:
        raise ValueError(f"Token verification failed: {exc}") from exc
