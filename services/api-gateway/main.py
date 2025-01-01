"""
SYNAPSE API Gateway
Kong-style unified API gateway with JWT auth, rate limiting,
service discovery, and request routing to all microservices.
"""
from __future__ import annotations
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import structlog
import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response, JSONResponse

from shared.utils.helpers import configure_logging, create_access_token, verify_token
from shared.config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

REQUEST_TOTAL = Counter("gateway_requests_total", "Total gateway requests", ["service", "method", "status"])
REQUEST_LATENCY = Histogram("gateway_latency_seconds", "Gateway request latency", ["service"])
RATE_LIMIT_HITS = Counter("gateway_rate_limit_hits_total", "Rate limit violations")

# Service registry
SERVICE_REGISTRY = {
    "telemetry":   f"{settings.kafka_bootstrap.split(':')[0].replace('kafka', 'telemetry-ingestion')}:8001",
    "driver":      "driver-monitoring:8002",
    "fleet":       "fleet-health:8003",
    "routing":     "routing-engine:8004",
    "supply-chain":"supply-chain:8005",
    "insight":     "insight-fusion:8006",
    "mode-ddr":    "mode-ddr:8007",
    "actuation":   "actuation:8008",
    "digital-twin":"digital-twin:8009",
    "carbon":      "carbon-reporting:8010",
    "audit":       "audit-ledger:8011",
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)
http_client: httpx.AsyncClient = None

# Rate limiting (in-memory: production uses Redis)
_request_counts: dict[str, list] = {}
RATE_LIMIT_PER_MINUTE = 300


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    configure_logging("api-gateway")
    http_client = httpx.AsyncClient(timeout=30.0)
    logger.info("api_gateway_started", services=list(SERVICE_REGISTRY.keys()))
    yield
    await http_client.aclose()


app = FastAPI(
    title="SYNAPSE API Gateway",
    description="Unified entry point for all SYNAPSE microservices",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth endpoints ────────────────────────────────────────────────────────────

@app.post("/auth/token")
async def login(form: OAuth2PasswordRequestForm = Depends()):
    """Issue JWT token for service authentication."""
    # Production: validate against Keycloak/LDAP
    if form.password != "synapse_demo":
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": form.username, "role": "operator"})
    return {"access_token": token, "token_type": "bearer"}


def get_current_user(token: str = Depends(oauth2_scheme)) -> Optional[dict]:
    """Validate JWT token and return user claims."""
    if not token:
        return None  # Public endpoints allowed without auth in dev mode
    try:
        return verify_token(token)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# ── Rate limiter ──────────────────────────────────────────────────────────────

def check_rate_limit(client_ip: str) -> bool:
    now = time.monotonic()
    if client_ip not in _request_counts:
        _request_counts[client_ip] = []
    # Sliding window: keep last 60 seconds
    _request_counts[client_ip] = [t for t in _request_counts[client_ip] if now - t < 60]
    if len(_request_counts[client_ip]) >= RATE_LIMIT_PER_MINUTE:
        return False
    _request_counts[client_ip].append(now)
    return True


# ── Proxy middleware ──────────────────────────────────────────────────────────

@app.api_route("/{service}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(
    service: str,
    path: str,
    request: Request,
    user: Optional[dict] = Depends(get_current_user),
):
    """Transparent reverse proxy to microservices with auth and rate limiting."""
    client_ip = request.client.host if request.client else "unknown"

    # Rate limiting
    if not check_rate_limit(client_ip):
        RATE_LIMIT_HITS.inc()
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Service lookup
    if service not in SERVICE_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Service '{service}' not found")

    target_host = SERVICE_REGISTRY[service]
    target_url = f"http://{target_host}/{path}"
    if request.url.query:
        target_url += f"?{request.url.query}"

    t0 = time.monotonic()
    try:
        body = await request.body()
        headers = dict(request.headers)
        headers.pop("host", None)

        response = await http_client.request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=body,
        )

        latency = time.monotonic() - t0
        REQUEST_TOTAL.labels(service=service, method=request.method, status=response.status_code).inc()
        REQUEST_LATENCY.labels(service=service).observe(latency)

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
        )

    except httpx.ConnectError:
        REQUEST_TOTAL.labels(service=service, method=request.method, status=503).inc()
        raise HTTPException(status_code=503, detail=f"Service '{service}' unavailable")
    except Exception as e:
        logger.error("proxy_error", service=service, path=path, error=str(e))
        raise HTTPException(status_code=502, detail="Gateway error")


# ── Info endpoints ────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """API Gateway root — service registry and status."""
    return {
        "service": "SYNAPSE API Gateway",
        "version": "1.0.0",
        "registered_services": list(SERVICE_REGISTRY.keys()),
        "auth_required": False,  # Set True in production
        "rate_limit_per_minute": RATE_LIMIT_PER_MINUTE,
    }


@app.get("/health")
async def health():
    """Gateway health check."""
    return {"status": "ok", "service": "api-gateway"}


@app.get("/services/health")
async def services_health():
    """Check health of all registered services."""
    results = {}
    for name, host in SERVICE_REGISTRY.items():
        try:
            r = await http_client.get(f"http://{host}/health", timeout=2.0)
            results[name] = {"status": "up", "code": r.status_code}
        except Exception:
            results[name] = {"status": "down"}
    return results


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
