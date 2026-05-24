from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    width: float = 1000.0
    height: float = 1000.0
    num_uavs: int = 4
    num_victims: int = 2
    sensing_radius: float = 140.0
    communication_radius: float = 260.0
    max_speed: float = 28.0
    victim_noise_std: float = 3.0
    drift_x: float = 2.0
    drift_y: float = 1.0
    victim_spawn_margin: float = 0.16
    victim_separation: float = 0.18
    victim_drift_spread: float = 0.7
    steps: int = 80
    seed: int = 7
    track_distance: float = 80.0
    grid_size: int = 20
    belief_blend: float = 0.35
    negative_observation_decay: float = 0.35
    message_confidence_threshold: float = 0.14
    message_change_threshold: float = 0.025
    message_stale_steps: int = 6
    policy_mode: str = "belief_sparse_comm"
    coverage_decay: float = 0.96
    handoff_success_window: int = 6
    packet_drop_rate: float = 0.0
    gps_noise_std: float = 0.0
    detection_decay_rate: float = 0.0
    actuation_delay: float = 0.0
