"""
SYNAPSE Digital Twin Service
Per-asset Gymnasium + SimPy simulation service.
Provides on-demand scenario simulation and what-if analysis.
"""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from shared.utils.helpers import SynapseCache, configure_logging
from shared.config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

SIMULATIONS_RUN = Counter("digital_twin_simulations_total", "Simulations executed")
SIM_DURATION = Histogram("digital_twin_sim_duration_seconds", "Simulation wall clock time")

cache: SynapseCache = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global cache
    configure_logging("digital-twin")
    cache = SynapseCache(settings.redis_url)
    await cache.connect()
    logger.info("digital_twin_started")
    yield
    await cache.disconnect()


app = FastAPI(
    title="SYNAPSE Digital Twin",
    description="Per-asset Gymnasium + SimPy scenario simulation",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


async def _run_simulation_async(sim_id: str, params: dict) -> None:
    """Run simulation in executor thread to avoid blocking event loop."""
    import time
    import sys
    sys.path.insert(0, ".")

    t0 = time.monotonic()
    try:
        from ml.simulation.run_simulation import SynapseDigitalTwin
        sim = SynapseDigitalTwin(
            n_trucks=params.get("n_trucks", 20),
            n_drivers=params.get("n_drivers", 20),
            n_shipments=params.get("n_shipments", 50),
            seed=params.get("seed", 42),
        )
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: sim.run(duration_hours=params.get("duration_hours", 24.0), verbose=False)
        )
        duration = time.monotonic() - t0
        SIM_DURATION.observe(duration)
        SIMULATIONS_RUN.inc()

        await cache.set_json(f"digital_twin:sim:{sim_id}", {
            "sim_id": sim_id,
            "status": "COMPLETE",
            "duration_wall_seconds": round(duration, 2),
            "results": results,
            "completed_at": datetime.utcnow().isoformat(),
        }, ttl=7200)

    except Exception as e:
        logger.error("simulation_error", sim_id=sim_id, error=str(e))
        await cache.set_json(f"digital_twin:sim:{sim_id}", {
            "sim_id": sim_id, "status": "FAILED", "error": str(e),
        }, ttl=3600)


@app.post("/simulate")
async def run_simulation(params: dict, bg: BackgroundTasks) -> dict:
    """Start an async simulation run."""
    import uuid
    sim_id = str(uuid.uuid4())
    await cache.set_json(f"digital_twin:sim:{sim_id}", {"sim_id": sim_id, "status": "RUNNING"}, ttl=7200)
    bg.add_task(_run_simulation_async, sim_id, params)
    return {"sim_id": sim_id, "status": "RUNNING", "message": "Simulation started"}


@app.get("/simulate/{sim_id}")
async def get_simulation_result(sim_id: str) -> dict:
    """Get simulation results by ID."""
    result = await cache.get_json(f"digital_twin:sim:{sim_id}")
    if not result:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return result


@app.post("/whatif")
async def what_if_analysis(scenario: dict) -> dict:
    """
    Synchronous what-if analysis for small scenarios.
    Returns performance comparison between baseline and proposed intervention.
    """
    import sys
    sys.path.insert(0, ".")
    try:
        from ml.simulation.run_simulation import SynapseDigitalTwin

        # Baseline
        baseline_sim = SynapseDigitalTwin(n_trucks=10, n_drivers=10, n_shipments=20, seed=42)
        baseline = baseline_sim.run(duration_hours=8.0, verbose=False)

        # Intervention (modify parameters based on proposed action)
        action_type = scenario.get("action_type", "NONE")
        modified_params = {"n_trucks": 10, "n_drivers": 10, "n_shipments": 20}

        if action_type == "ADD_CAPACITY":
            modified_params["n_trucks"] = 12
        elif action_type == "REROUTE":
            modified_params["n_shipments"] = 18  # Fewer shipments on congested lane

        intervention_sim = SynapseDigitalTwin(**modified_params, seed=42)
        intervention = intervention_sim.run(duration_hours=8.0, verbose=False)

        return {
            "scenario": scenario,
            "baseline": {k: v for k, v in baseline.items() if isinstance(v, (int, float, str))},
            "intervention": {k: v for k, v in intervention.items() if isinstance(v, (int, float, str))},
            "delta": {
                "otp_delta": round(intervention["on_time_performance"] - baseline["on_time_performance"], 4),
                "breakdown_delta": intervention["breakdowns"] - baseline["breakdowns"],
                "co2e_delta_kg": round(intervention["total_co2e_kg"] - baseline["total_co2e_kg"], 1),
            }
        }
    except Exception as e:
        return {"error": str(e), "message": "Simulation failed — check service dependencies"}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "digital-twin"}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8009, reload=True)
