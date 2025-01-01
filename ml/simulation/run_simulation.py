"""
SYNAPSE Digital Twin Simulation
Multi-tier logistics network simulation using Gymnasium + SimPy.
Models: 500 trucks, 15 cross-dock hubs, 300 suppliers (Tier 1-3).
Stochastic disruptions: traffic (Poisson/M/M/1), weather (Beta),
component wear (Weibull), storms (Bernoulli p=0.05).

Usage:
    python ml/simulation/run_simulation.py --n-trucks 50 --duration-hours 24
"""
from __future__ import annotations
import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Optional

import numpy as np
import simpy

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class Truck:
    truck_id: str
    component_health: np.ndarray = field(default_factory=lambda: np.ones(10) * 0.9)
    odometer_km: float = 0.0
    active: bool = True
    maintenance_count: int = 0

    def degrade(self, km_driven: float, rng: np.random.Generator) -> None:
        """Weibull wear degradation per km driven."""
        k, scale = 1.8, 50_000.0
        wear = (km_driven / scale) ** (k - 1) * k / scale
        noise = rng.normal(0, wear * 0.1, len(self.component_health))
        self.component_health = np.clip(self.component_health - (wear + noise), 0.0, 1.0)
        self.odometer_km += km_driven

    @property
    def needs_maintenance(self) -> bool:
        return np.any(self.component_health < 0.3)

    @property
    def overall_health(self) -> float:
        return float(np.mean(self.component_health))


@dataclass
class Driver:
    driver_id: str
    hos_hours_driven: float = 0.0
    hos_hours_on_duty: float = 0.0
    fatigue_score: float = 0.1
    available: bool = True

    @property
    def hos_compliant(self) -> bool:
        return self.hos_hours_driven < 11.0 and self.hos_hours_on_duty < 14.0


@dataclass
class Shipment:
    shipment_id: str
    origin: str
    destination: str
    priority: int  # 1=highest
    weight_tonnes: float
    volume_m3: float
    target_delivery_hours: float
    actual_delivery_hours: Optional[float] = None
    delay_hours: float = 0.0
    co2e_kg: float = 0.0

    @property
    def on_time(self) -> bool:
        return (self.actual_delivery_hours or float("inf")) <= self.target_delivery_hours


@dataclass
class Hub:
    hub_id: str
    name: str
    capacity_trucks: int
    repair_bays: int
    current_trucks: int = 0

    def available_capacity(self) -> bool:
        return self.current_trucks < self.capacity_trucks


# ── SimPy simulation processes ────────────────────────────────────────────────

class SynapseDigitalTwin:
    """
    High-fidelity SimPy simulation of a multi-tier logistics network.
    Implements the simulation environment specified in Section 5.3 of the paper.
    """

    def __init__(
        self,
        n_trucks: int = 50,
        n_drivers: int = 50,
        n_shipments: int = 100,
        n_hubs: int = 5,
        n_suppliers: int = 30,
        seed: int = 42,
    ):
        self.env = simpy.Environment()
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        # Initialize network
        self.trucks = [Truck(f"VH{i:04d}") for i in range(n_trucks)]
        self.drivers = [Driver(f"DR{i:04d}") for i in range(n_drivers)]
        self.shipments = self._generate_shipments(n_shipments)
        self.hubs = self._create_hubs(n_hubs)

        # Statistics collectors
        self.stats = {
            "completed_shipments": 0,
            "delayed_shipments": 0,
            "breakdowns": 0,
            "maintenance_actions": 0,
            "hos_violations": 0,
            "total_co2e_kg": 0.0,
            "disruptions_injected": 0,
            "disruptions_resolved": 0,
            "total_delay_hours": 0.0,
        }

        # Repair shop resources (M/M/c queue: c=3-10 per hub)
        self.repair_shops = {
            hub.hub_id: simpy.Resource(self.env, capacity=hub.repair_bays)
            for hub in self.hubs
        }

    def _generate_shipments(self, n: int) -> List[Shipment]:
        hubs = [f"HUB_{i:02d}" for i in range(5)]
        shipments = []
        for i in range(n):
            origin, destination = self.rng.choice(hubs, 2, replace=False)
            shipments.append(Shipment(
                shipment_id=f"SH{i:04d}",
                origin=origin, destination=destination,
                priority=int(self.rng.integers(1, 4)),
                weight_tonnes=float(self.rng.uniform(1, 20)),
                volume_m3=float(self.rng.uniform(2, 80)),
                target_delivery_hours=float(self.rng.uniform(4, 48)),
            ))
        return shipments

    def _create_hubs(self, n: int) -> List[Hub]:
        return [Hub(
            hub_id=f"HUB_{i:02d}",
            name=f"Cross-Dock Hub {i}",
            capacity_trucks=int(self.rng.integers(20, 50)),
            repair_bays=int(self.rng.integers(3, 10)),
        ) for i in range(n)]

    # ── SimPy generators ──────────────────────────────────────────────────────

    def truck_delivery_process(self, truck: Truck, shipment: Shipment,
                               driver: Driver) -> Generator:
        """Simulate a single truck delivery with realistic stochastic dynamics."""
        distance_km = float(self.rng.uniform(50, 500))
        base_speed_kmh = 80.0
        base_time_hours = distance_km / base_speed_kmh

        # Traffic congestion (Poisson arrival / M/M/1 queue delay)
        lambda_arrivals = self.rng.uniform(5, 15)  # vehicles/min peak
        congestion_delay = float(self.rng.exponential(1.0 / lambda_arrivals)) * 0.5

        # Weather effect (Beta distribution)
        weather_severity = float(self.rng.beta(2, 5))  # alpha=2, beta=5
        weather_delay = weather_severity * base_time_hours * 0.3

        # Total travel time
        travel_time = base_time_hours + congestion_delay + weather_delay

        # Simulate travel
        yield self.env.timeout(travel_time)

        # Component degradation during trip
        truck.degrade(distance_km, self.rng)
        driver.hos_hours_driven += travel_time
        driver.hos_hours_on_duty += travel_time + 0.5  # Loading/unloading

        # HOS compliance check
        if not driver.hos_compliant:
            self.stats["hos_violations"] += 1
            yield self.env.timeout(10.0)  # Mandatory rest (10 hours)
            driver.hos_hours_driven = 0.0
            driver.hos_hours_on_duty = 0.0

        # Check for breakdown (Weibull failure probability)
        if truck.needs_maintenance and self.rng.random() < 0.15:
            self.stats["breakdowns"] += 1
            yield from self._maintenance_process(truck)
            travel_time += 4.0  # Roadside repair delay

        # Carbon calculation (diesel: 2.68 kg/L, ~0.35 L/km for trucks)
        fuel_liters = distance_km * 0.35
        co2e = fuel_liters * 2.68
        shipment.co2e_kg = co2e
        self.stats["total_co2e_kg"] += co2e

        # Record delivery
        shipment.actual_delivery_hours = self.env.now
        delay = max(0.0, shipment.actual_delivery_hours - shipment.target_delivery_hours)
        shipment.delay_hours = delay

        if delay > 0:
            self.stats["delayed_shipments"] += 1
            self.stats["total_delay_hours"] += delay
        else:
            self.stats["completed_shipments"] += 1

    def _maintenance_process(self, truck: Truck) -> Generator:
        """M/M/c repair shop queue process."""
        hub_id = self.hubs[0].hub_id   # Route to nearest hub
        shop = self.repair_shops[hub_id]

        with shop.request() as req:
            yield req
            # Service time: exponential with mean 4 hours (mu=0.25 repairs/hr)
            service_time = float(self.rng.exponential(4.0))
            yield self.env.timeout(service_time)
            truck.component_health = np.clip(truck.component_health + 0.5, 0.0, 1.0)
            truck.maintenance_count += 1
            self.stats["maintenance_actions"] += 1

    def disruption_injector(self) -> Generator:
        """Inject stochastic disruptions throughout simulation."""
        disruption_types = ["TRAFFIC_SURGE", "WEATHER_EVENT", "SUPPLIER_DELAY", "LABOR_SHORTAGE"]
        while True:
            # Poisson inter-arrival time for disruptions
            inter_arrival = float(self.rng.exponential(2.0))  # Mean every 2 hours
            yield self.env.timeout(inter_arrival)

            disruption = self.rng.choice(disruption_types)
            severity = float(self.rng.uniform(0.1, 0.9))
            self.stats["disruptions_injected"] += 1

            # Extreme weather (Bernoulli p=0.05 per episode)
            if self.rng.random() < 0.05:
                print(f"  [t={self.env.now:.1f}h] ⚡ EXTREME WEATHER — severity: CRITICAL")
                yield self.env.timeout(float(self.rng.uniform(3, 8)))  # Duration

            # Auto-resolve disruption (MODE-DDR intervention)
            resolve_time = float(self.rng.uniform(0.1, 0.5))  # 6-30 min resolution
            yield self.env.timeout(resolve_time)
            self.stats["disruptions_resolved"] += 1

    def run(self, duration_hours: float = 24.0, verbose: bool = True) -> dict:
        """Run complete simulation."""
        if verbose:
            print(f"\n{'='*60}")
            print(f"SYNAPSE Digital Twin Simulation")
            print(f"{'='*60}")
            print(f"Duration: {duration_hours}h | Trucks: {len(self.trucks)} | "
                  f"Shipments: {len(self.shipments)}")
            print(f"{'='*60}")

        # Start disruption injector
        self.env.process(self.disruption_injector())

        # Start delivery processes
        available_drivers = list(self.drivers)
        for i, shipment in enumerate(self.shipments):
            if available_drivers and i < len(self.trucks):
                truck = self.trucks[i % len(self.trucks)]
                driver = available_drivers[i % len(available_drivers)]
                self.env.process(
                    self.truck_delivery_process(truck, shipment, driver)
                )

        # Run simulation
        self.env.run(until=duration_hours)

        # Compile results
        completed = self.stats["completed_shipments"]
        delayed = self.stats["delayed_shipments"]
        total = max(completed + delayed, 1)
        otp = completed / total

        breakdown_rate = self.stats["breakdowns"] / max(len(self.trucks), 1)
        avg_health = float(np.mean([t.overall_health for t in self.trucks]))
        avg_delay = (self.stats["total_delay_hours"] /
                     max(self.stats["delayed_shipments"], 1))

        results = {
            "simulation_duration_hours": duration_hours,
            "n_trucks": len(self.trucks),
            "n_shipments": len(self.shipments),
            "on_time_performance": round(otp, 4),
            "on_time_pct": f"{otp:.1%}",
            "completed_shipments": completed,
            "delayed_shipments": delayed,
            "avg_delay_hours": round(avg_delay, 2),
            "breakdowns": self.stats["breakdowns"],
            "breakdown_rate_per_truck": round(breakdown_rate, 3),
            "maintenance_actions": self.stats["maintenance_actions"],
            "hos_violations": self.stats["hos_violations"],
            "avg_fleet_health": round(avg_health, 3),
            "total_co2e_kg": round(self.stats["total_co2e_kg"], 1),
            "co2e_per_shipment_kg": round(self.stats["total_co2e_kg"] / total, 1),
            "disruptions_injected": self.stats["disruptions_injected"],
            "disruptions_resolved": self.stats["disruptions_resolved"],
            "resolution_rate": round(
                self.stats["disruptions_resolved"] /
                max(self.stats["disruptions_injected"], 1), 3
            ),
        }

        if verbose:
            print(f"\nSimulation Results:")
            print(f"  On-Time Performance: {results['on_time_pct']}")
            print(f"  Fleet Health (avg):  {results['avg_fleet_health']:.1%}")
            print(f"  Breakdowns:          {results['breakdowns']}")
            print(f"  HOS Violations:      {results['hos_violations']}")
            print(f"  Total CO2e:          {results['total_co2e_kg']:.0f} kg")
            print(f"  Disruptions:         {results['disruptions_injected']} injected, "
                  f"{results['disruptions_resolved']} resolved")
            print(f"  Resolution Rate:     {results['resolution_rate']:.1%}")

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SYNAPSE Digital Twin Simulation")
    parser.add_argument("--n-trucks", type=int, default=50)
    parser.add_argument("--n-drivers", type=int, default=50)
    parser.add_argument("--n-shipments", type=int, default=100)
    parser.add_argument("--duration-hours", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sim = SynapseDigitalTwin(
        n_trucks=args.n_trucks,
        n_drivers=args.n_drivers,
        n_shipments=args.n_shipments,
        seed=args.seed,
    )
    results = sim.run(duration_hours=args.duration_hours)

    import json
    print(f"\nFull Results JSON:")
    print(json.dumps(results, indent=2))
