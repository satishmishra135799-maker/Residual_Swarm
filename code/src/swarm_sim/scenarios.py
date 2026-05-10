from __future__ import annotations

from .config import SimulationConfig


def scenario_config(name: str, policy_mode: str) -> SimulationConfig:
    if name == "hard":
        return SimulationConfig(
            policy_mode=policy_mode,
            num_uavs=6,
            num_victims=4,
            sensing_radius=95.0,
            communication_radius=180.0,
            drift_x=4.0,
            drift_y=2.5,
            steps=110,
            seed=11,
            victim_drift_spread=1.2,
            message_confidence_threshold=0.10,
            message_change_threshold=0.015,
            message_stale_steps=4,
            victim_separation=0.48,
            victim_spawn_margin=0.28,
        )
    if name == "medium":
        return SimulationConfig(
            policy_mode=policy_mode,
            num_uavs=5,
            num_victims=3,
            sensing_radius=120.0,
            communication_radius=220.0,
            drift_x=3.0,
            drift_y=1.7,
            steps=95,
            seed=9,
            victim_drift_spread=0.9,
            victim_separation=0.42,
            victim_spawn_margin=0.24,
        )
    if name == "swarm_heavy":
        return SimulationConfig(
            policy_mode=policy_mode,
            num_uavs=7,
            num_victims=5,
            sensing_radius=110.0,
            communication_radius=210.0,
            drift_x=3.6,
            drift_y=2.1,
            steps=120,
            seed=13,
            victim_drift_spread=1.3,
            message_confidence_threshold=0.11,
            message_change_threshold=0.018,
            message_stale_steps=4,
            victim_separation=0.56,
            victim_spawn_margin=0.34,
        )
    return SimulationConfig(
        policy_mode=policy_mode,
        victim_separation=0.34,
        victim_spawn_margin=0.22,
    )
