"""
SYNAPSE Shared Configuration
Environment-driven config (12-factor) used across all services.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "SYNAPSE"
    environment: str = "development"
    log_level: str = "INFO"
    debug: bool = False

    kafka_bootstrap: str = "localhost:9092"
    kafka_group_prefix: str = "synapse"

    db_url: str = "postgresql+asyncpg://synapse:synapse_secret@localhost/synapse"
    db_pool_size: int = 10

    redis_url: str = "redis://localhost:6379"
    redis_ttl_seconds: int = 300

    fleet_health_url: str = "http://localhost:8003"
    driver_url: str = "http://localhost:8002"
    routing_url: str = "http://localhost:8004"
    supply_chain_url: str = "http://localhost:8005"
    insight_fusion_url: str = "http://localhost:8006"
    actuation_url: str = "http://localhost:8008"

    model_dir: str = "/app/models"
    lstm_model_path: str = "/app/models/lstm_degradation.onnx"
    xgboost_model_path: str = "/app/models/xgboost_failure.json"
    gnn_model_path: str = "/app/models/gnn_routing.pt"
    driver_cnn_model_path: str = "/app/models/driver_cnn.onnx"
    mode_ddr_model_path: str = "/app/models/mode_ddr_policy.zip"

    hos_max_driving_hours: float = 11.0
    hos_max_on_duty_hours: float = 14.0
    hos_max_weekly_hours: float = 70.0
    hos_min_rest_hours: float = 10.0

    diesel_emission_factor_kg_per_liter: float = 2.68
    default_spend_multiplier_kg_per_usd: float = 1.0

    reward_weight_cost: float = 0.4
    reward_weight_service: float = 0.4
    reward_weight_emissions: float = 0.2
    hos_violation_penalty: float = -10.0

    jwt_secret: str = "synapse-dev-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
