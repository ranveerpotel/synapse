"""
SYNAPSE Shared Pydantic Schemas
Canonical data models used across all microservices.
"""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import uuid


# ── Enumerations ────────────────────────────────────────────────────────────

class AlertSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ActionType(str, Enum):
    REROUTE = "REROUTE"
    REASSIGN_LOAD = "REASSIGN_LOAD"
    SCHEDULE_MAINTENANCE = "SCHEDULE_MAINTENANCE"
    TRIGGER_BREAK = "TRIGGER_BREAK"
    SUPPLIER_ESCALATION = "SUPPLIER_ESCALATION"
    MODE_SHIFT = "MODE_SHIFT"
    HOLD_SHIPMENT = "HOLD_SHIPMENT"
    ADJUST_DEPARTURE = "ADJUST_DEPARTURE"


class TransportMode(str, Enum):
    ROAD = "ROAD"
    AIR = "AIR"
    RAIL = "RAIL"
    SEA = "SEA"


class DisruptionType(str, Enum):
    MECHANICAL_FAILURE = "MECHANICAL_FAILURE"
    DRIVER_FATIGUE = "DRIVER_FATIGUE"
    TRAFFIC_CONGESTION = "TRAFFIC_CONGESTION"
    WEATHER = "WEATHER"
    SUPPLIER_DELAY = "SUPPLIER_DELAY"
    PORT_CONGESTION = "PORT_CONGESTION"
    ROAD_CLOSURE = "ROAD_CLOSURE"
    LABOR_SHORTAGE = "LABOR_SHORTAGE"


# ── Telemetry Schemas ────────────────────────────────────────────────────────

class CANBusSignal(BaseModel):
    """Raw CAN bus signal from vehicle OBD-II interface."""
    vehicle_id: str
    timestamp: datetime
    engine_rpm: float = Field(ge=0, le=8000)
    torque_nm: float = Field(ge=0, le=3000)
    oil_pressure_kpa: float = Field(ge=0, le=700)
    coolant_temp_c: float = Field(ge=-40, le=150)
    tire_pressure_fl_kpa: float = Field(ge=0, le=1000)
    tire_pressure_fr_kpa: float = Field(ge=0, le=1000)
    tire_pressure_rl_kpa: float = Field(ge=0, le=1000)
    tire_pressure_rr_kpa: float = Field(ge=0, le=1000)
    vibration_rms_g: float = Field(ge=0, le=50)
    fuel_level_pct: float = Field(ge=0, le=100)
    odometer_km: float = Field(ge=0)
    speed_kmh: float = Field(ge=0, le=200)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    fault_codes: List[str] = Field(default_factory=list)
    harsh_brake_event: bool = False
    harsh_accel_event: bool = False


class FreightIoTSignal(BaseModel):
    """Freight-level IoT sensor reading."""
    shipment_id: str
    vehicle_id: str
    timestamp: datetime
    temperature_c: float
    humidity_pct: float
    shock_g: float
    door_open: bool
    gps_lat: float
    gps_lon: float


# ── Driver State Schemas ────────────────────────────────────────────────────

class DriverPhysiologicalSignal(BaseModel):
    """Wearable device physiological signal."""
    driver_id: str
    timestamp: datetime
    heart_rate_bpm: float = Field(ge=30, le=220)
    hrv_ms: float = Field(ge=0, le=200)          # Heart Rate Variability
    skin_conductance_us: float = Field(ge=0)       # Galvanic skin response
    stress_index: float = Field(ge=0, le=1)        # Normalized [0,1]


class DriverVisionSignal(BaseModel):
    """In-cab camera derived behavioral signal."""
    driver_id: str
    vehicle_id: str
    timestamp: datetime
    eye_closure_rate_pct: float = Field(ge=0, le=100)
    gaze_deviation_deg: float = Field(ge=0, le=90)
    head_pose_yaw_deg: float = Field(ge=-90, le=90)
    head_pose_pitch_deg: float = Field(ge=-90, le=90)
    yawn_detected: bool = False
    distraction_detected: bool = False
    micro_sleep_detected: bool = False


class DriverStateScore(BaseModel):
    """Fused driver risk assessment."""
    driver_id: str
    vehicle_id: str
    timestamp: datetime
    fatigue_probability: float = Field(ge=0, le=1)
    distraction_probability: float = Field(ge=0, le=1)
    stress_index: float = Field(ge=0, le=1)
    hos_risk_score: float = Field(ge=0, le=1)    # Hours of Service risk
    cumulative_driving_hours: float = Field(ge=0, le=70)
    remaining_drive_hours: float = Field(ge=0, le=11)
    risk_level: AlertSeverity
    recommended_action: Optional[str] = None


# ── Fleet Health Schemas ────────────────────────────────────────────────────

class ComponentHealth(BaseModel):
    """Individual vehicle component health state."""
    component_id: str
    component_name: str                            # e.g., "left_rear_tire", "engine_oil"
    degradation_score: float = Field(ge=0, le=1)  # 0=new, 1=failed
    failure_probability_72h: float = Field(ge=0, le=1)
    estimated_remaining_life_km: float = Field(ge=0)
    anomaly_score: float = Field(ge=0, le=1)
    last_maintenance_km: float = Field(ge=0)


class VehicleHealthState(BaseModel):
    """Complete vehicle health assessment from ensemble models."""
    vehicle_id: str
    timestamp: datetime
    overall_health_score: float = Field(ge=0, le=1)   # 1=perfect
    failure_probability_72h: float = Field(ge=0, le=1)
    components: List[ComponentHealth]
    anomaly_detected: bool = False
    anomaly_type: Optional[str] = None
    maintenance_required: bool = False
    estimated_failure_window_hours: Optional[float] = None
    roc_auc_confidence: float = Field(ge=0, le=1, default=0.89)


# ── Routing & ETA Schemas ───────────────────────────────────────────────────

class RouteNode(BaseModel):
    """A node in the logistics network graph."""
    node_id: str
    node_type: str  # "hub", "customer", "supplier", "port"
    latitude: float
    longitude: float
    name: str


class RouteEdge(BaseModel):
    """A transportation lane between nodes."""
    from_node: str
    to_node: str
    distance_km: float
    mode: TransportMode
    congestion_factor: float = Field(ge=0, le=5, default=1.0)
    estimated_time_hours: float


class ETAPrediction(BaseModel):
    """ETA prediction from GNN + Transformer model."""
    shipment_id: str
    vehicle_id: str
    origin_node: str
    destination_node: str
    predicted_eta: datetime
    eta_mae_minutes: float
    eta_confidence_interval_minutes: float
    congestion_probability: float = Field(ge=0, le=1)
    recommended_route: List[str]
    multi_objective_cost: Dict[str, float]  # time, fuel, carbon, HOS costs


# ── Supply Chain Schemas ────────────────────────────────────────────────────

class SupplierNode(BaseModel):
    """Supply chain network node (Tier 1/2/3)."""
    supplier_id: str
    tier: int = Field(ge=1, le=3)
    name: str
    country: str
    lead_time_days: float
    reliability_score: float = Field(ge=0, le=1)


class SupplyChainDisruptionForecast(BaseModel):
    """Predicted supply chain disruption from Multivariate Transformer."""
    forecast_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime
    affected_supplier_ids: List[str]
    disruption_type: DisruptionType
    delay_probability: float = Field(ge=0, le=1)
    estimated_delay_days: float
    confidence: float = Field(ge=0, le=1)
    propagation_risk_tier2: float = Field(ge=0, le=1)
    propagation_risk_tier3: float = Field(ge=0, le=1)
    mape: float                                    # Model accuracy metric


# ── Fused State Vector ──────────────────────────────────────────────────────

class FusedStateVector(BaseModel):
    """
    150-dimensional cross-domain state vector for MODE-DDR.
    Assembled by insight-fusion-svc from all domain predictions.
    """
    state_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime

    # Fleet health subspace (~30D)
    fleet_degradation_scores: List[float]          # Per component per vehicle
    fleet_failure_probabilities: List[float]
    fleet_anomaly_scores: List[float]

    # Driver subspace (~20D)
    driver_fatigue_probabilities: List[float]      # Per active driver
    driver_distraction_scores: List[float]
    driver_stress_indices: List[float]
    driver_hos_risk_scores: List[float]

    # Routing/network subspace (~30D)
    eta_errors: List[float]                        # Per active shipment
    lane_congestion_features: List[float]
    emissions_forecasts: List[float]

    # Supply chain subspace (~30D)
    supplier_delay_probabilities: List[float]
    lead_time_forecasts: List[float]
    disruption_propagation_indicators: List[float]

    # Environmental subspace (~10D)
    weather_indices: List[float]
    traffic_signals: List[float]
    external_disruption_indicators: List[float]

    # Metadata
    active_vehicles: int
    active_drivers: int
    active_shipments: int
    fleet_utilization_pct: float


# ── Prescriptive Action Schemas ─────────────────────────────────────────────

class PrescriptiveAction(BaseModel):
    """A single ranked prescriptive action from MODE-DDR."""
    action_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    action_type: ActionType
    priority_rank: int = Field(ge=1)
    target_id: str                                 # vehicle_id, driver_id, shipment_id

    # Multi-objective impact scores
    cost_impact: float                             # Normalized [-1, 1], negative=saving
    service_impact: float                          # Negative=improvement
    safety_impact: float                           # Negative=improvement
    carbon_impact_kg_co2e: float
    compliance_risk_delta: float

    # Composite reward score
    reward_score: float

    # Human-readable explanation
    description: str
    rationale: str
    shap_top_features: Dict[str, float]           # Feature importance

    # Action parameters
    parameters: Dict[str, Any] = Field(default_factory=dict)

    # Constraints satisfied
    hos_compliant: bool = True
    maintenance_compliant: bool = True
    feasibility_checked: bool = True


class ModeDDRDecision(BaseModel):
    """Complete MODE-DDR decision output for a disruption event."""
    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime
    state_id: str
    disruption_type: Optional[DisruptionType] = None
    ranked_actions: List[PrescriptiveAction]       # Top-K ranked actions
    resolution_latency_ms: float
    pareto_front_quality: float                    # Hypervolume indicator
    human_approval_required: bool = False
    auto_execute: bool = False


# ── Carbon / Emissions Schemas ──────────────────────────────────────────────

class ShipmentCarbonReport(BaseModel):
    """Per-shipment Scope 3 emissions calculation (ISO 14083 / GLEC)."""
    shipment_id: str
    vehicle_id: str
    calculation_timestamp: datetime

    # Direct emissions (Scope 1)
    fuel_consumed_liters: float
    emission_factor_kg_per_liter: float = 2.68   # Diesel default
    direct_co2e_kg: float

    # Upstream Scope 3 (Tier 2-3 spend-based)
    supplier_spend_usd: float
    spend_multiplier_kg_per_usd: float
    upstream_co2e_kg: float

    # Totals
    total_co2e_kg: float
    co2e_per_km: float
    baseline_co2e_kg: float
    reduction_vs_baseline_pct: float

    # Compliance
    iso_14083_compliant: bool = True
    glec_framework_version: str = "3.0"


# ── Audit / Compliance Schemas ──────────────────────────────────────────────

class ComplianceEvent(BaseModel):
    """Immutable compliance event for blockchain audit ledger."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime
    event_type: str                               # "HOS_LOG", "LOAD_ASSIGNMENT", "CARBON_CALC"
    entity_id: str                                # driver_id, shipment_id, vehicle_id
    entity_type: str
    payload: Dict[str, Any]
    hash_previous: Optional[str] = None          # Blockchain chain linkage
    digital_signature: Optional[str] = None
    verified: bool = False
