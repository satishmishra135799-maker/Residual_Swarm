from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .belief import belief_peak_xy
from .config import SimulationConfig
from .coverage import coverage_gap_target
from .simulator import MaritimeSwarmSimulator


@dataclass
class AgentObservation:
    agent_id: int
    vector: list[float]
    detection_flag: bool
    message_neighbors: int
    belief_peak: list[float]
    belief_confidence: float
    coverage_target: list[float]


class SwarmLearningEnv:
    """Learning-ready wrapper with reset/step observations and rewards."""

    def __init__(self, config: SimulationConfig, *, control_mode: str = "direct", residual_scale: float = 0.35, confidence_gate_threshold: float = 0.08) -> None:
        self.simulator = MaritimeSwarmSimulator(config)
        self.control_mode = control_mode
        self.residual_scale = residual_scale
        self.confidence_gate_threshold = confidence_gate_threshold
        self.last_actions = np.zeros((config.num_uavs, 2), dtype=float)

    @property
    def config(self) -> SimulationConfig:
        return self.simulator.config

    def _build_observations(self) -> list[AgentObservation]:
        detections, detected_indices = self.simulator._detect()
        links = self.simulator._links()
        neighbor_count = {idx: 0 for idx in range(self.config.num_uavs)}
        for i, j in links:
            neighbor_count[i] += 1
            neighbor_count[j] += 1

        global_peak_xy, global_peak_conf = belief_peak_xy(
            np.mean(np.stack(self.simulator.beliefs, axis=0), axis=0),
            self.config,
        )
        detections_now, detected_indices_now = self.simulator._detect()
        peaks_now, peak_confidences_now = self.simulator._belief_peaks()
        gap_targets_now = [
            coverage_gap_target(self.simulator.coverage, self.config, agent_idx, self.config.num_uavs)
            for agent_idx in range(self.config.num_uavs)
        ]
        _, current_assignments = self.simulator._builtin_actions(
            detections_now,
            detected_indices_now,
            peaks_now,
            peak_confidences_now,
            gap_targets_now,
        )

        observations: list[AgentObservation] = []
        for idx in range(self.config.num_uavs):
            peak_xy, peak_confidence = belief_peak_xy(self.simulator.beliefs[idx], self.config)
            gap_target = coverage_gap_target(
                self.simulator.coverage,
                self.config,
                idx,
                self.config.num_uavs,
            )
            assigned_victim = current_assignments[idx]
            position = self.simulator.uav_positions[idx]
            if self.config.gps_noise_std > 0.0:
                obs_position = position + self.simulator.rng.normal(0.0, self.config.gps_noise_std, size=2)
            else:
                obs_position = position
            velocity = self.simulator.uav_velocities[idx]
            cell_x = min(int(position[0] / (self.config.width / self.config.grid_size)), self.config.grid_size - 1)
            cell_y = min(int(position[1] / (self.config.height / self.config.grid_size)), self.config.grid_size - 1)
            local_coverage_value = float(self.simulator.coverage[cell_y, cell_x])
            peak_delta = (peak_xy - obs_position) / np.array([self.config.width, self.config.height], dtype=float)
            gap_delta = (gap_target - obs_position) / np.array([self.config.width, self.config.height], dtype=float)
            global_peak_delta = (global_peak_xy - obs_position) / np.array([self.config.width, self.config.height], dtype=float)
            last_action = self.last_actions[idx]
            step_fraction = float(self.simulator.steps_taken / max(self.config.steps, 1))
            lane_fraction = float((idx + 0.5) / self.config.num_uavs)
            handoff_target_flag = 1.0 if idx in self.simulator.victim_owner_ids.tolist() else 0.0
            assigned_victim_norm = (
                float((assigned_victim + 1) / max(self.config.num_victims, 1))
                if assigned_victim >= 0
                else 0.0
            )
            vector = [
                float(obs_position[0] / self.config.width),
                float(obs_position[1] / self.config.height),
                float(velocity[0] / max(self.config.max_speed, 1.0)),
                float(velocity[1] / max(self.config.max_speed, 1.0)),
                float(peak_xy[0] / self.config.width),
                float(peak_xy[1] / self.config.height),
                float(peak_confidence),
                float(gap_target[0] / self.config.width),
                float(gap_target[1] / self.config.height),
                float(neighbor_count[idx] / max(self.config.num_uavs - 1, 1)),
                float(detections[idx]),
                float(peak_delta[0]),
                float(peak_delta[1]),
                float(gap_delta[0]),
                float(gap_delta[1]),
                float(global_peak_delta[0]),
                float(global_peak_delta[1]),
                float(global_peak_conf),
                float(local_coverage_value),
                float(last_action[0]),
                float(last_action[1]),
                step_fraction,
                lane_fraction,
                assigned_victim_norm,
                handoff_target_flag,
            ]
            observations.append(
                AgentObservation(
                    agent_id=idx,
                    vector=vector,
                    detection_flag=bool(detections[idx]),
                    message_neighbors=neighbor_count[idx],
                    belief_peak=peak_xy.round(3).tolist(),
                    belief_confidence=float(round(peak_confidence, 4)),
                    coverage_target=gap_target.round(3).tolist(),
                )
            )
        return observations

    def reset(self) -> list[AgentObservation]:
        self.simulator.reset()
        self.last_actions = np.zeros((self.config.num_uavs, 2), dtype=float)
        return self._build_observations()

    def step(self, actions: np.ndarray | None = None) -> tuple[list[AgentObservation], list[float], bool, dict[str, Any]]:
        applied_actions = actions
        if actions is not None and self.control_mode == "residual":
            expert = self.simulator.current_builtin_actions() / max(self.config.max_speed, 1.0)
            applied_actions = np.clip(expert + self.residual_scale * np.asarray(actions, dtype=float), -1.0, 1.0)
        elif actions is not None and self.control_mode == "gated_residual":
            expert = self.simulator.current_builtin_actions() / max(self.config.max_speed, 1.0)
            residual = np.asarray(actions, dtype=float)
            # gate per UAV: only apply residual if belief confidence exceeds threshold
            confidences = np.array([obs.belief_confidence for obs in self._build_observations()])
            gates = (confidences >= self.confidence_gate_threshold).astype(float)
            applied_actions = np.clip(expert + self.residual_scale * gates[:, None] * residual, -1.0, 1.0)
        elif actions is not None and self.control_mode == "soft_gated_residual":
            expert = self.simulator.current_builtin_actions() / max(self.config.max_speed, 1.0)
            residual = np.asarray(actions, dtype=float)
            confidences = np.array([obs.belief_confidence for obs in self._build_observations()])
            # soft gate: sigmoid(k * (c - tau)), smoothly interpolates 0->1 around threshold
            k = 20.0
            gates = 1.0 / (1.0 + np.exp(-k * (confidences - self.confidence_gate_threshold)))
            applied_actions = np.clip(expert + self.residual_scale * gates[:, None] * residual, -1.0, 1.0)
        elif actions is not None and self.control_mode == "learnable_tau_gated":
            # tau is passed via confidence_gate_threshold, extracted from theta by training script
            expert = self.simulator.current_builtin_actions() / max(self.config.max_speed, 1.0)
            residual = np.asarray(actions, dtype=float)
            confidences = np.array([obs.belief_confidence for obs in self._build_observations()])
            k = 20.0
            gates = 1.0 / (1.0 + np.exp(-k * (confidences - self.confidence_gate_threshold)))
            applied_actions = np.clip(expert + self.residual_scale * gates[:, None] * residual, -1.0, 1.0)

        record = self.simulator.step(external_actions=applied_actions)
        if applied_actions is None:
            self.last_actions = self.simulator.uav_velocities / max(self.config.max_speed, 1.0)
        else:
            self.last_actions = np.asarray(applied_actions, dtype=float)
        observations = self._build_observations()
        reward = self._team_reward(record)
        rewards = [reward for _ in range(self.config.num_uavs)]
        done = self.simulator.steps_taken >= self.config.steps
        info: dict[str, Any] = {
            "step": record.step,
            "events": record.event_markers,
            "tracking_active": record.tracking_active,
            "used_external_actions": actions is not None,
            "control_mode": self.control_mode,
            "summary_so_far": self.simulator.summary(),
        }
        return observations, rewards, done, info

    def _team_reward(self, record: Any) -> float:
        reward = 0.0
        if "first_detection" in record.event_markers:
            reward += 5.0
        reward += 2.5 * sum(1 for marker in record.event_markers if marker.startswith("first_detection_v"))
        if record.tracking_active:
            reward += 0.5 + 0.4 * record.tracked_victim_count
        reward += 1.75 * sum(1 for marker in record.event_markers if marker.startswith("handoff_success"))
        if any(marker.startswith("track_loss") for marker in record.event_markers):
            reward -= 1.25 * sum(1 for marker in record.event_markers if marker.startswith("track_loss"))
        if any(marker.startswith("handoff_fail") for marker in record.event_markers):
            reward -= 1.0 * sum(1 for marker in record.event_markers if marker.startswith("handoff_fail"))
        reward += 0.25 * sum(record.detected_victim_counts)
        reward -= 0.01 * sum(1 for sent in record.transmitted_by if sent)
        return reward
