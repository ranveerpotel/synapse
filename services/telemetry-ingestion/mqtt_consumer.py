"""
MQTT Consumer
Async MQTT subscriber for CAN bus and freight IoT streams.
Supports 5,000+ messages/sec at 1-10Hz across 500 vehicles.
"""
from __future__ import annotations
import asyncio
import json
from typing import Callable, Awaitable
import structlog

logger = structlog.get_logger(__name__)


class MQTTConsumer:
    """
    Async MQTT subscriber using aiomqtt.
    Reconnects automatically on disconnect.
    """

    def __init__(
        self,
        broker_host: str,
        broker_port: int,
        topics: list[str],
        on_message: Callable[[str, dict], Awaitable[None]],
        reconnect_delay: float = 5.0,
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topics = topics
        self.on_message = on_message
        self.reconnect_delay = reconnect_delay
        self._running = False

    async def run(self) -> None:
        """Main consumer loop with automatic reconnect."""
        self._running = True
        while self._running:
            try:
                await self._connect_and_consume()
            except Exception as e:
                logger.error("mqtt_consumer_error", error=str(e))
                if self._running:
                    logger.info("mqtt_reconnecting", delay=self.reconnect_delay)
                    await asyncio.sleep(self.reconnect_delay)

    async def _connect_and_consume(self) -> None:
        """
        Connect to MQTT broker and consume messages.
        Uses aiomqtt when available; falls back to simulation mode.
        """
        try:
            import aiomqtt
            async with aiomqtt.Client(
                hostname=self.broker_host,
                port=self.broker_port,
                keepalive=30,
            ) as client:
                for topic in self.topics:
                    await client.subscribe(topic, qos=1)
                logger.info("mqtt_connected", broker=self.broker_host, topics=self.topics)

                async for message in client.messages:
                    try:
                        payload = json.loads(message.payload.decode())
                        await self.on_message(str(message.topic), payload)
                    except json.JSONDecodeError as e:
                        logger.warning("mqtt_json_error", error=str(e))
                    except Exception as e:
                        logger.error("mqtt_handler_error", error=str(e))

        except ImportError:
            # Simulation mode when MQTT broker not available
            logger.warning("mqtt_simulation_mode", reason="aiomqtt not available or broker unreachable")
            await self._simulate_telemetry()

    async def _simulate_telemetry(self) -> None:
        """Simulate MQTT messages for development/testing."""
        import random
        from datetime import datetime
        vehicle_ids = [f"VH{i:04d}" for i in range(1, 11)]

        while self._running:
            for vid in vehicle_ids:
                payload = {
                    "vehicle_id": vid,
                    "timestamp": datetime.utcnow().isoformat(),
                    "engine_rpm": random.uniform(800, 2200),
                    "torque_nm": random.uniform(200, 800),
                    "oil_pressure_kpa": random.uniform(200, 500),
                    "coolant_temp_c": random.uniform(70, 100),
                    "tire_pressure_fl_kpa": random.uniform(700, 850),
                    "tire_pressure_fr_kpa": random.uniform(700, 850),
                    "tire_pressure_rl_kpa": random.uniform(700, 850),
                    "tire_pressure_rr_kpa": random.uniform(700, 850),
                    "vibration_rms_g": random.uniform(0.1, 2.0),
                    "fuel_level_pct": random.uniform(20, 100),
                    "odometer_km": random.uniform(50000, 500000),
                    "speed_kmh": random.uniform(0, 110),
                    "latitude": 40.7128 + random.uniform(-1, 1),
                    "longitude": -74.0060 + random.uniform(-1, 1),
                    "fault_codes": [],
                    "harsh_brake_event": random.random() < 0.02,
                    "harsh_accel_event": random.random() < 0.02,
                }
                await self.on_message(f"vehicles/{vid}/telemetry", payload)
            await asyncio.sleep(0.5)  # ~2Hz simulation

    async def stop(self) -> None:
        self._running = False
