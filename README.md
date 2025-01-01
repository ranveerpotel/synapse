# SYNAPSE Orchestration Framework

**Synergized Networks, Analytics, Platforms, Security, and Edge**

A unified predictive–prescriptive orchestration framework for fleet, driver, and supply chain optimization achieving first- and last-mile excellence.

Based on: *Potel, R. (2025). Fleet, Driver & Supply Chain Optimization Achieving First- and Last-Mile Excellence through SYNAPSE Orchestration. IJAIBDCMS, 6(4), 46–74.*

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 5: Security & Governance (Zero-Trust / Hyperledger)      │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4: AI & Analytics Stack (LSTM / XGBoost / GNN / PPO)     │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Cloud-Native Platform (FastAPI / Kafka / Kubernetes)   │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Edge Intelligence (ONNX / TensorRT / Digital Twins)    │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Connectivity Fabric (5G / MQTT / AMQP)                │
└─────────────────────────────────────────────────────────────────┘
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| `api-gateway` | 8000 | Kong-style unified API gateway |
| `telemetry-ingestion` | 8001 | CAN bus + IoT ingestion (1-10Hz) |
| `driver-monitoring` | 8002 | CV fatigue + HRV physiological scoring |
| `fleet-health` | 8003 | LSTM + XGBoost + Isolation Forest ensemble |
| `routing-engine` | 8004 | GNN path optimization + Transformer ETA |
| `supply-chain` | 8005 | Multivariate Transformer delay forecasting |
| `insight-fusion` | 8006 | Cross-domain 150D state vector assembly |
| `mode-ddr` | 8007 | PPO/DDPG RL prescriptive decision engine |
| `actuation` | 8008 | TMS/WMS/ELD API execution layer |
| `digital-twin` | 8009 | Gymnasium + SimPy per-asset simulation |
| `carbon-reporting` | 8010 | ISO 14083/GLEC CO2e calculation |
| `audit-ledger` | 8011 | Blockchain compliance event logging |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start infrastructure (Kafka, Redis, TimescaleDB)
docker-compose up -d kafka redis timescaledb

# 3. Run all services
docker-compose up

# 4. Or run individual service
cd services/fleet-health && uvicorn main:app --port 8003 --reload

# 5. Train MODE-DDR RL agent
python ml/training/train_mode_ddr.py

# 6. Run simulation
python ml/simulation/run_simulation.py
```

## Performance Targets

| Metric | Baseline | SYNAPSE Target |
|--------|---------|----------------|
| Exception resolution latency | 38 min | 6 min (6× faster) |
| On-time performance | 91.2% | 97.8% |
| Unplanned downtime | 22% | 5% (-72%) |
| Operational cost | baseline | -17% |
| Carbon emissions | baseline | -28% |
| Safety risk exposure | baseline | -52% |
| ETA MAE | 25.8 min | 9.7 min |
| PdM ROC-AUC | 0.61 | 0.89 |

## Stack

- **Framework**: FastAPI 0.111+ (async, uvicorn + uvloop)
- **ML/AI**: PyTorch 2.3, PyTorch Geometric, XGBoost 2.0, scikit-learn
- **RL Engine**: stable-baselines3 2.3, Gymnasium 0.29.1, SimPy 4.1.1
- **Streaming**: Apache Kafka + aiokafka, MQTT (aiomqtt)
- **Database**: TimescaleDB (telemetry), PostgreSQL (ops), Redis (cache)
- **Feature Store**: Feast 0.38
- **Edge**: ONNX Runtime + TensorRT
- **Security**: Keycloak (OIDC), OPA (ABAC), Hyperledger Fabric
- **Orchestration**: Kubernetes + Helm, Docker

## License

Research implementation based on academic work. See LICENSE.
