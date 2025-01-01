"""
MODE-DDR Gymnasium Environment
Multi-Objective Markov Decision Process for logistics orchestration.
State: 150D fused vector | Actions: 50-100 operational decisions
Reward: -(w_cost*C + w_service*S + w_emissions*E) + bonus
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SynapseLogisticsEnv(gym.Env):
    """
    SYNAPSE Multi-Objective MDP environment.

    Implements FMCSA HOS hard constraints and ISO 14083 emissions tracking.
    Used for MODE-DDR training via PPO + DDPG.

    State space: ~150-180 dimensional vector (partially observable)
    Action space: 50-100 discrete + continuous actions
    Episode: 24-72 simulated hours
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    # Multi-objective reward weights (default from paper: 0.4/0.4/0.2)
    W_COST = 0.4
    W_SERVICE = 0.4
    W_EMISSIONS = 0.2

    # HOS limits (FMCSA)
    MAX_DRIVE_HOURS = 11.0
    MAX_DUTY_HOURS = 14.0
    HOS_VIOLATION_PENALTY = -10.0

    # Episode duration
    MAX_STEPS = 144  # 24hr at 10-min intervals

    def __init__(
        self,
        n_vehicles: int = 10,
        n_drivers: int = 10,
        n_shipments: int = 20,
        n_suppliers: int = 15,
        reward_weights: Optional[Dict[str, float]] = None,
        partial_observability: float = 0.8,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.n_vehicles = n_vehicles
        self.n_drivers = n_drivers
        self.n_shipments = n_shipments
        self.n_suppliers = n_suppliers
        self.partial_obs = partial_observability
        self.render_mode = render_mode

        if reward_weights:
            self.W_COST = reward_weights.get("cost", 0.4)
            self.W_SERVICE = reward_weights.get("service", 0.4)
            self.W_EMISSIONS = reward_weights.get("emissions", 0.2)

        # ── State space ────────────────────────────────────────────
        # Fleet: 3 signals * n_vehicles
        # Driver: 4 signals * n_drivers
        # Routing: 3 signals * n_shipments
        # Supply chain: 3 signals * n_suppliers
        # Environmental: 10 signals
        # Metadata: 4 scalars
        self.state_dim = (
            3 * n_vehicles + 4 * n_drivers + 3 * n_shipments
            + 3 * n_suppliers + 10 + 4
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        # ── Action space ────────────────────────────────────────────
        # 8 discrete action types × targets = ~100 actions
        # Encoded as MultiDiscrete for training
        self.n_action_types = 8
        self.action_space = spaces.MultiDiscrete([
            self.n_action_types,   # Action type
            max(n_vehicles, n_drivers, n_shipments),  # Target entity
        ])

        # ── Internal state ─────────────────────────────────────────
        self._step_count = 0
        self._episode_cost = 0.0
        self._episode_emissions = 0.0
        self._episode_delays = 0.0
        self._hos_hours: np.ndarray = None
        self._component_health: np.ndarray = None
        self._shipment_etas: np.ndarray = None
        self._supplier_risks: np.ndarray = None
        self._disruption_active = False
        self._disruption_type: Optional[str] = None

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._step_count = 0
        self._episode_cost = 0.0
        self._episode_emissions = 0.0
        self._episode_delays = 0.0

        # Initialize fleet (70% utilization, Weibull health distribution)
        self._component_health = self.np_random.weibull(2.0, size=(self.n_vehicles, 3))
        self._component_health = np.clip(self._component_health * 0.4, 0.05, 0.95)

        # Driver HOS hours (randomized start state)
        self._hos_hours = self.np_random.uniform(0.0, 5.0, size=self.n_drivers)

        # Shipment ETAs (hours to delivery, 4-48h range)
        self._shipment_etas = self.np_random.uniform(4.0, 48.0, size=self.n_shipments)

        # Supplier risks
        self._supplier_risks = self.np_random.beta(2, 8, size=(self.n_suppliers, 3))

        # Randomly inject a disruption (30% chance on episode start)
        self._disruption_active = self.np_random.random() < 0.30
        self._disruption_type = (
            self.np_random.choice(["TRAFFIC", "WEATHER", "MECHANICAL", "SUPPLIER"])
            if self._disruption_active else None
        )

        obs = self._build_observation()
        info = {
            "step": 0, "disruption": self._disruption_active,
            "disruption_type": self._disruption_type,
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action_type = int(action[0])
        target_idx = int(action[1]) % max(self.n_vehicles, self.n_drivers, self.n_shipments)

        # Apply action and compute effects
        cost_delta, service_delta, emissions_delta = self._apply_action(action_type, target_idx)

        # Advance simulation (10-minute step)
        self._advance_simulation()

        # Compute multi-objective reward
        reward = self._compute_reward(cost_delta, service_delta, emissions_delta)

        # Accumulate episode metrics
        self._episode_cost += cost_delta
        self._episode_emissions += emissions_delta
        self._episode_delays += max(0, service_delta)

        # Check termination conditions
        self._step_count += 1
        terminated = self._step_count >= self.MAX_STEPS
        truncated = self._check_critical_failure()

        obs = self._build_observation()
        info = {
            "step": self._step_count,
            "cost_delta": cost_delta,
            "service_delta": service_delta,
            "emissions_delta": emissions_delta,
            "reward": reward,
            "episode_cost": self._episode_cost,
            "episode_emissions": self._episode_emissions,
            "disruption_resolved": not self._disruption_active,
        }
        return obs, float(reward), terminated, truncated, info

    def _apply_action(self, action_type: int, target_idx: int) -> Tuple[float, float, float]:
        """
        Apply the selected action and return (cost_delta, service_delta, emissions_delta).
        Negative values = improvement.
        """
        action_effects = {
            0: self._action_reroute,
            1: self._action_reassign_load,
            2: self._action_schedule_maintenance,
            3: self._action_trigger_break,
            4: self._action_supplier_escalation,
            5: self._action_mode_shift,
            6: self._action_hold_shipment,
            7: self._action_noop,
        }
        action_fn = action_effects.get(action_type, self._action_noop)
        return action_fn(target_idx)

    def _action_reroute(self, shipment_idx: int) -> Tuple[float, float, float]:
        """Reroute shipment to avoid congestion."""
        idx = shipment_idx % self.n_shipments
        fuel_surcharge = self.np_random.uniform(5, 15)   # +$5-15 fuel cost
        time_saving = self.np_random.uniform(0.2, 1.5)   # -12-90 min delay
        co2_delta = self.np_random.uniform(-10, 5)        # +/-kg CO2
        if self._disruption_active and self._disruption_type == "TRAFFIC":
            time_saving *= 2.0
            self._disruption_active = False
        self._shipment_etas[idx] -= time_saving
        return fuel_surcharge / 1000.0, -time_saving / 48.0, co2_delta / 10000.0

    def _action_reassign_load(self, driver_idx: int) -> Tuple[float, float, float]:
        """Reassign load to different driver or 3PL."""
        idx = driver_idx % self.n_drivers
        reassign_cost = self.np_random.uniform(20, 80)
        delay = self.np_random.uniform(-0.5, 1.0)
        self._hos_hours[idx] = max(0.0, self._hos_hours[idx] - 2.0)
        return reassign_cost / 1000.0, delay / 48.0, 0.01

    def _action_schedule_maintenance(self, vehicle_idx: int) -> Tuple[float, float, float]:
        """Schedule preventive maintenance for a vehicle."""
        idx = vehicle_idx % self.n_vehicles
        maint_cost = self.np_random.uniform(200, 1000)
        delay = self.np_random.uniform(2.0, 6.0)  # Vehicle out of service
        # Prevent future breakdown (reduce degradation)
        self._component_health[idx] *= 0.5
        if self._disruption_active and self._disruption_type == "MECHANICAL":
            self._disruption_active = False
            delay *= 0.5
        return maint_cost / 10000.0, delay / 48.0, -0.005

    def _action_trigger_break(self, driver_idx: int) -> Tuple[float, float, float]:
        """Trigger mandatory rest break for fatigued driver."""
        idx = driver_idx % self.n_drivers
        break_cost = 15.0 / 1000.0   # Delay cost
        break_time_hours = 0.5
        self._hos_hours[idx] = max(0.0, self._hos_hours[idx] - 1.0)
        return break_cost, break_time_hours / 48.0, 0.001

    def _action_supplier_escalation(self, supplier_idx: int) -> Tuple[float, float, float]:
        """Escalate to alternative supplier."""
        idx = supplier_idx % self.n_suppliers
        escalation_cost = self.np_random.uniform(100, 500)
        delay_saved = self.np_random.uniform(0.5, 3.0)
        self._supplier_risks[idx] *= 0.7
        if self._disruption_active and self._disruption_type == "SUPPLIER":
            self._disruption_active = False
        return escalation_cost / 10000.0, -delay_saved / 48.0, 0.02

    def _action_mode_shift(self, shipment_idx: int) -> Tuple[float, float, float]:
        """Shift transport mode (road → air/rail)."""
        idx = shipment_idx % self.n_shipments
        air_surcharge = self.np_random.uniform(500, 2000)
        time_saving = self.np_random.uniform(12.0, 36.0)
        co2_penalty = self.np_random.uniform(50, 200)   # Air = higher emissions
        self._shipment_etas[idx] -= time_saving
        return air_surcharge / 10000.0, -time_saving / 48.0, co2_penalty / 10000.0

    def _action_hold_shipment(self, shipment_idx: int) -> Tuple[float, float, float]:
        """Hold shipment until disruption clears."""
        idx = shipment_idx % self.n_shipments
        hold_delay = self.np_random.uniform(1.0, 4.0)
        self._shipment_etas[idx] += hold_delay
        return 0.005, hold_delay / 48.0, -0.002

    def _action_noop(self, _: int) -> Tuple[float, float, float]:
        """No-op action — monitor and wait."""
        return 0.001, 0.005, 0.001

    def _compute_reward(self, cost: float, service: float, emissions: float) -> float:
        """
        Multi-objective scalarized reward.
        r_t = -(w_cost * C_t + w_service * S_t + w_emissions * E_t) + bonus_t
        """
        # Penalize HOS violations (hard constraint)
        hos_penalty = 0.0
        if np.any(self._hos_hours > self.MAX_DRIVE_HOURS):
            hos_penalty = self.HOS_VIOLATION_PENALTY

        # Service penalty (quadratic for large delays)
        service_normalized = (service ** 2) if service > 0 else service

        r = -(
            self.W_COST * cost
            + self.W_SERVICE * service_normalized
            + self.W_EMISSIONS * emissions
        ) + hos_penalty

        # Resolution bonus
        if not self._disruption_active and self._step_count < 5:
            r += 1.0  # Fast resolution bonus

        return float(r)

    def _advance_simulation(self) -> None:
        """Advance simulation by one 10-minute time step."""
        dt_hours = 10.0 / 60.0

        # Advance HOS hours (all drivers on duty)
        active_mask = self._hos_hours < self.MAX_DRIVE_HOURS
        self._hos_hours += dt_hours * active_mask

        # Advance ETAs
        self._shipment_etas = np.maximum(0.0, self._shipment_etas - dt_hours)

        # Degrade vehicle components (Weibull wear)
        wear_rate = self.np_random.exponential(0.001, size=self._component_health.shape)
        self._component_health = np.clip(self._component_health + wear_rate, 0.0, 1.0)

        # Stochastic disruption injection (Poisson process)
        if not self._disruption_active and self.np_random.random() < 0.02:
            self._disruption_active = True
            self._disruption_type = self.np_random.choice(
                ["TRAFFIC", "WEATHER", "MECHANICAL", "SUPPLIER"]
            )

    def _build_observation(self) -> np.ndarray:
        """Construct the observation vector with partial observability."""
        obs_parts = []

        # Fleet subspace
        obs_parts.extend(self._component_health.flatten())          # n_vehicles * 3

        # Driver subspace
        obs_parts.extend(self._hos_hours / self.MAX_DRIVE_HOURS)    # n_drivers (normalized)
        fatigue_proxy = np.clip(self._hos_hours / self.MAX_DRIVE_HOURS, 0, 1)
        obs_parts.extend(fatigue_proxy)                             # n_drivers
        stress_proxy = fatigue_proxy * 0.7 + self.np_random.normal(0, 0.05, self.n_drivers)
        obs_parts.extend(np.clip(stress_proxy, 0, 1))              # n_drivers
        hos_risk = np.clip(self._hos_hours / self.MAX_DRIVE_HOURS, 0, 1)
        obs_parts.extend(hos_risk)                                  # n_drivers

        # Routing subspace
        eta_norm = np.clip(self._shipment_etas / 48.0, 0, 1)
        obs_parts.extend(eta_norm)                                  # n_shipments
        congestion = self.np_random.uniform(0.1, 0.6, self.n_shipments)
        obs_parts.extend(congestion)                               # n_shipments
        emissions_est = congestion * 0.5
        obs_parts.extend(emissions_est)                            # n_shipments

        # Supply chain subspace
        obs_parts.extend(self._supplier_risks.flatten())           # n_suppliers * 3

        # Environmental (10 signals)
        env = self.np_random.uniform(0.0, 0.3, 10)
        if self._disruption_active:
            if self._disruption_type == "WEATHER":
                env[:3] += 0.5
            elif self._disruption_type == "TRAFFIC":
                env[3:6] += 0.4
        obs_parts.extend(np.clip(env, 0, 1))

        # Metadata (4 scalars)
        obs_parts.extend([
            self._step_count / self.MAX_STEPS,                     # Time progress
            float(self._disruption_active),                        # Disruption flag
            np.mean(self._component_health),                       # Fleet health mean
            np.mean(self._hos_hours) / self.MAX_DRIVE_HOURS,       # Mean HOS utilization
        ])

        obs = np.array(obs_parts, dtype=np.float32)

        # Apply partial observability: add noise to some signals (Tier 2/3 visibility)
        if self.partial_obs < 1.0:
            noise_mask = self.np_random.random(len(obs)) > self.partial_obs
            obs[noise_mask] += self.np_random.normal(0, 0.1, noise_mask.sum())
            obs = np.clip(obs, 0.0, 1.0)

        # Pad or trim to exact state_dim
        if len(obs) < self.state_dim:
            obs = np.pad(obs, (0, self.state_dim - len(obs)))
        return obs[:self.state_dim]

    def _check_critical_failure(self) -> bool:
        """Check for terminal failure conditions."""
        delayed_pct = np.mean(self._shipment_etas > 48.0)
        if delayed_pct > 0.5:
            return True
        if np.any(self._component_health > 0.95):
            return True
        return False

    def render(self) -> None:
        if self.render_mode == "human":
            print(f"\n[SYNAPSE Env] Step {self._step_count}/{self.MAX_STEPS}")
            print(f"  Fleet health: {1 - np.mean(self._component_health):.2%}")
            print(f"  HOS max: {np.max(self._hos_hours):.1f}h / {self.MAX_DRIVE_HOURS}h")
            print(f"  Disruption: {self._disruption_type if self._disruption_active else 'None'}")
            print(f"  Shipments on-time: {np.mean(self._shipment_etas < 24):.1%}")
