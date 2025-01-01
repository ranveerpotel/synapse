-- SYNAPSE TimescaleDB Schema Initialization
-- Creates hypertables for time-series telemetry, predictions, and compliance events

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ── Telemetry Hypertable ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS vehicle_telemetry (
    time            TIMESTAMPTZ     NOT NULL,
    vehicle_id      VARCHAR(20)     NOT NULL,
    engine_rpm      FLOAT           NOT NULL,
    torque_nm       FLOAT,
    oil_pressure    FLOAT,
    coolant_temp    FLOAT,
    tire_variance   FLOAT,
    vibration_rms   FLOAT,
    fuel_level_pct  FLOAT,
    speed_kmh       FLOAT,
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    fault_codes     TEXT[],
    harsh_brake     BOOLEAN DEFAULT FALSE,
    harsh_accel     BOOLEAN DEFAULT FALSE,
    -- Derived / normalized features
    engine_rpm_norm FLOAT,
    thermal_stress  FLOAT,
    engine_load     FLOAT,
    fuel_rate_lph   FLOAT
);

SELECT create_hypertable('vehicle_telemetry', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_telemetry_vehicle_time
    ON vehicle_telemetry (vehicle_id, time DESC);

-- ── Fleet Health Predictions ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fleet_health_predictions (
    time                        TIMESTAMPTZ     NOT NULL,
    vehicle_id                  VARCHAR(20)     NOT NULL,
    overall_health_score        FLOAT,
    failure_probability_72h     FLOAT,
    anomaly_detected            BOOLEAN,
    anomaly_type                VARCHAR(50),
    maintenance_required        BOOLEAN,
    estimated_failure_hours     FLOAT,
    roc_auc_confidence          FLOAT,
    model_version               VARCHAR(20)
);

SELECT create_hypertable('fleet_health_predictions', 'time',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

-- ── Driver State Predictions ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS driver_state_predictions (
    time                    TIMESTAMPTZ     NOT NULL,
    driver_id               VARCHAR(20)     NOT NULL,
    vehicle_id              VARCHAR(20),
    fatigue_probability     FLOAT,
    distraction_probability FLOAT,
    stress_index            FLOAT,
    hos_risk_score          FLOAT,
    cumulative_drive_hours  FLOAT,
    remaining_drive_hours   FLOAT,
    risk_level              VARCHAR(10),
    recommended_action      VARCHAR(100)
);

SELECT create_hypertable('driver_state_predictions', 'time',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

-- ── ETA Predictions ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS eta_predictions (
    time                        TIMESTAMPTZ     NOT NULL,
    shipment_id                 VARCHAR(30)     NOT NULL,
    vehicle_id                  VARCHAR(20),
    origin_node                 VARCHAR(30),
    destination_node            VARCHAR(30),
    predicted_eta               TIMESTAMPTZ,
    eta_mae_minutes             FLOAT,
    eta_ci_minutes              FLOAT,
    congestion_probability      FLOAT,
    fuel_cost                   FLOAT,
    carbon_kg_co2e              FLOAT,
    distance_km                 FLOAT
);

SELECT create_hypertable('eta_predictions', 'time',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

-- ── Supply Chain Disruptions ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS supply_chain_disruptions (
    time                        TIMESTAMPTZ     NOT NULL,
    forecast_id                 UUID            NOT NULL DEFAULT gen_random_uuid(),
    disruption_type             VARCHAR(50),
    delay_probability           FLOAT,
    estimated_delay_days        FLOAT,
    confidence                  FLOAT,
    propagation_risk_tier2      FLOAT,
    propagation_risk_tier3      FLOAT,
    mape                        FLOAT
);

SELECT create_hypertable('supply_chain_disruptions', 'time',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- ── MODE-DDR Decisions ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS mode_ddr_decisions (
    time                    TIMESTAMPTZ     NOT NULL,
    decision_id             UUID            NOT NULL DEFAULT gen_random_uuid(),
    state_id                UUID,
    disruption_type         VARCHAR(50),
    top_action_type         VARCHAR(50),
    top_action_reward       FLOAT,
    resolution_latency_ms   FLOAT,
    auto_executed           BOOLEAN,
    human_approved          BOOLEAN,
    pareto_quality          FLOAT
);

SELECT create_hypertable('mode_ddr_decisions', 'time',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

-- ── Carbon Reports ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS carbon_reports (
    time                        TIMESTAMPTZ     NOT NULL,
    shipment_id                 VARCHAR(30)     NOT NULL,
    vehicle_id                  VARCHAR(20),
    fuel_consumed_liters        FLOAT,
    direct_co2e_kg              FLOAT,
    upstream_co2e_kg            FLOAT,
    total_co2e_kg               FLOAT,
    co2e_per_km                 FLOAT,
    baseline_co2e_kg            FLOAT,
    reduction_pct               FLOAT,
    iso_compliant               BOOLEAN,
    glec_version                VARCHAR(10)
);

SELECT create_hypertable('carbon_reports', 'time',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- ── Compliance Events (Audit Ledger) ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS compliance_events (
    time                TIMESTAMPTZ     NOT NULL,
    event_id            UUID            NOT NULL DEFAULT gen_random_uuid(),
    event_type          VARCHAR(50)     NOT NULL,
    entity_id           VARCHAR(50)     NOT NULL,
    entity_type         VARCHAR(30),
    payload             JSONB,
    hash_previous       VARCHAR(64),
    digital_signature   VARCHAR(64),
    verified            BOOLEAN
);

SELECT create_hypertable('compliance_events', 'time',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_compliance_entity
    ON compliance_events (entity_id, time DESC);

-- ── Continuous aggregates (performance) ──────────────────────────────────────

-- Hourly fleet health summary
CREATE MATERIALIZED VIEW IF NOT EXISTS fleet_health_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    vehicle_id,
    AVG(overall_health_score) AS avg_health,
    MAX(failure_probability_72h) AS max_failure_prob,
    SUM(CASE WHEN anomaly_detected THEN 1 ELSE 0 END) AS anomaly_count
FROM fleet_health_predictions
GROUP BY bucket, vehicle_id;

-- Daily OTP summary
CREATE MATERIALIZED VIEW IF NOT EXISTS otp_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS bucket,
    COUNT(*) AS total_predictions,
    AVG(eta_mae_minutes) AS avg_mae,
    AVG(congestion_probability) AS avg_congestion
FROM eta_predictions
GROUP BY bucket;

COMMENT ON TABLE vehicle_telemetry IS 'CAN bus + IoT telemetry at 1-10Hz, 500-vehicle fleet';
COMMENT ON TABLE fleet_health_predictions IS 'LSTM+XGBoost+IsoForest ensemble predictions, 72hr lead time';
COMMENT ON TABLE compliance_events IS 'Hash-chained immutable audit log, 7-year retention';
