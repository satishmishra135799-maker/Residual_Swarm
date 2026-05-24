from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .belief import belief_peak_xy, drift_predict, fuse_messages, uniform_belief, update_with_observation
from .config import SimulationConfig
from .coverage import coverage_gap_target, initial_coverage, update_coverage
from .policies import BeliefAwarePolicy, FrontierCoveragePolicy, HybridSearchPolicy, SweepSearchPolicy


@dataclass
class StepRecord:
    step: int
    uav_positions: list[list[float]]
    victim_positions: list[list[float]]
    detections: list[bool]
    detected_victim_counts: list[int]
    assigned_victim_ids: list[int]
    victim_owner_ids: list[int]
    tracked_victim_count: int
    handoff_count: int
    handoff_success_count: int
    handoff_failure_count: int
    links: list[list[int]]
    transmitted_by: list[bool]
    peak_confidences: list[float]
    average_belief_peak: list[float]
    coverage_gap_target: list[float]
    tracking_active: bool
    event_markers: list[str]


class MaritimeSwarmSimulator:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.policy = self._build_policy()
        self.reset()

    def _build_policy(self) -> BeliefAwarePolicy | FrontierCoveragePolicy | HybridSearchPolicy | SweepSearchPolicy:
        if self.config.policy_mode == "sweep_only":
            return SweepSearchPolicy(
                max_speed=self.config.max_speed,
                width=self.config.width,
                height=self.config.height,
                num_uavs=self.config.num_uavs,
            )
        if self.config.policy_mode == "frontier_cover":
            return FrontierCoveragePolicy(
                max_speed=self.config.max_speed,
                width=self.config.width,
                height=self.config.height,
                num_uavs=self.config.num_uavs,
            )
        if self.config.policy_mode == "hybrid_frontier_belief":
            return HybridSearchPolicy(
                max_speed=self.config.max_speed,
                width=self.config.width,
                height=self.config.height,
                num_uavs=self.config.num_uavs,
            )
        return BeliefAwarePolicy(
            max_speed=self.config.max_speed,
            width=self.config.width,
            height=self.config.height,
            num_uavs=self.config.num_uavs,
        )

    def reset(self) -> None:
        self.steps_taken = 0
        lane_width = self.config.width / self.config.num_uavs
        self.uav_positions = np.array(
            [
                [lane_width * (idx + 0.5), self.config.height * 0.1]
                for idx in range(self.config.num_uavs)
            ],
            dtype=float,
        )
        self.uav_velocities = np.zeros_like(self.uav_positions)
        self.victim_positions = self._initial_victim_positions()
        self.victim_drifts = self._victim_drift_vectors()
        self.history: list[StepRecord] = []
        self.first_detection_step: int | None = None
        self.total_detections = 0
        self.total_tracked_victims = 0
        self.track_loss_count = 0
        self.reacquisition_delays: list[int] = []
        self.handoff_count = 0
        self.handoff_success_count = 0
        self.handoff_failure_count = 0
        self.last_tracking_mask = np.zeros(self.config.num_victims, dtype=bool)
        self.pending_reacquisition_start: list[int | None] = [None for _ in range(self.config.num_victims)]
        self.victim_owner_ids = np.full(self.config.num_victims, -1, dtype=int)
        self.pending_handoffs: list[dict[str, int] | None] = [None for _ in range(self.config.num_victims)]
        self.total_messages_sent = 0
        self.total_messages_delivered = 0
        self.beliefs = [uniform_belief(self.config.grid_size) for _ in range(self.config.num_uavs)]
        self.coverage = initial_coverage(self.config.grid_size)
        self.last_peak_confidences = [0.0 for _ in range(self.config.num_uavs)]
        self.last_message_steps = [-self.config.message_stale_steps for _ in range(self.config.num_uavs)]
        self.victim_first_detection_steps: list[int | None] = [None for _ in range(self.config.num_victims)]
        self.victim_last_seen_steps = np.full(self.config.num_victims, -1, dtype=int)
        self.victim_estimates = self.victim_positions.copy()
        self.victim_known_mask = np.zeros(self.config.num_victims, dtype=bool)

    def _initial_victim_positions(self) -> np.ndarray:
        cols = max(1, int(np.ceil(self.config.num_victims / 2)))
        x_start = self.config.width * 0.22
        x_end = min(
            self.config.width * 0.82,
            x_start + self.config.width * self.config.victim_separation,
        )
        xs = np.linspace(x_start, x_end, cols)
        y_top = self.config.height * 0.22
        y_bottom = min(
            self.config.height * 0.72,
            y_top + self.config.height * self.config.victim_spawn_margin,
        )
        positions = []
        for idx in range(self.config.num_victims):
            col = idx % cols
            row = idx // cols
            base_y = y_top if row == 0 else y_bottom
            if row == 1 and cols > 1:
                x_step = xs[1] - xs[0]
                base_x = min(self.config.width * 0.88, xs[col] + 0.18 * x_step)
            else:
                base_x = xs[col]
            base = np.array([base_x, base_y], dtype=float)
            jitter = self.rng.normal(0.0, 10.0, size=2)
            positions.append(self._clip_position(base + jitter))
        return np.asarray(positions, dtype=float)

    def _victim_drift_vectors(self) -> np.ndarray:
        offsets = np.linspace(-1.0, 1.0, self.config.num_victims)
        drifts = []
        for offset in offsets:
            drifts.append(
                np.array(
                    [
                        self.config.drift_x + offset * self.config.victim_drift_spread,
                        self.config.drift_y + 0.6 * offset * self.config.victim_drift_spread,
                    ],
                    dtype=float,
                )
            )
        return np.asarray(drifts, dtype=float)

    def _clip_position(self, position: np.ndarray) -> np.ndarray:
        return np.clip(position, [0.0, 0.0], [self.config.width, self.config.height])

    def _drift_victims(self) -> None:
        noise = self.rng.normal(0.0, self.config.victim_noise_std, size=(self.config.num_victims, 2))
        self.victim_positions = np.asarray(
            [
                self._clip_position(position + drift + noise[idx])
                for idx, (position, drift) in enumerate(zip(self.victim_positions, self.victim_drifts, strict=False))
            ],
            dtype=float,
        )

    def _pairwise_distances(self) -> np.ndarray:
        return np.linalg.norm(
            self.uav_positions[:, None, :] - self.victim_positions[None, :, :],
            axis=2,
        )

    def _detect(self) -> tuple[np.ndarray, list[list[int]]]:
        distances = self._pairwise_distances()
        if self.config.detection_decay_rate > 0.0:
            ratio = distances / self.config.sensing_radius
            probs = np.exp(-self.config.detection_decay_rate * ratio ** 2)
            probs[distances > self.config.sensing_radius] = 0.0
            visible = self.rng.random(distances.shape) < probs
        else:
            visible = distances <= self.config.sensing_radius
        detections = np.any(visible, axis=1)
        detected_indices = [np.flatnonzero(visible[idx]).tolist() for idx in range(self.config.num_uavs)]
        return detections, detected_indices

    def _tracking_mask(self) -> np.ndarray:
        distances = self._pairwise_distances()
        return np.any(distances <= self.config.track_distance, axis=0)

    def _links(self) -> list[list[int]]:
        links: list[list[int]] = []
        for i in range(self.config.num_uavs):
            for j in range(i + 1, self.config.num_uavs):
                distance = np.linalg.norm(self.uav_positions[i] - self.uav_positions[j])
                if distance <= self.config.communication_radius:
                    links.append([i, j])
        return links

    def _should_transmit(self, idx: int, detected: bool, peak_confidence: float) -> bool:
        confidence_changed = abs(peak_confidence - self.last_peak_confidences[idx]) >= self.config.message_change_threshold
        stale = (self.steps_taken - self.last_message_steps[idx]) >= self.config.message_stale_steps
        return detected or peak_confidence >= self.config.message_confidence_threshold or confidence_changed or stale

    def _belief_peaks(self) -> tuple[list[np.ndarray], list[float]]:
        peaks: list[np.ndarray] = []
        confidences: list[float] = []
        for belief in self.beliefs:
            peak_xy, peak_confidence = belief_peak_xy(belief, self.config)
            peaks.append(peak_xy)
            confidences.append(peak_confidence)
        return peaks, confidences

    def _predict_victim_estimates(self) -> np.ndarray:
        estimates = self.victim_estimates.copy()
        for victim_idx in range(self.config.num_victims):
            if self.victim_known_mask[victim_idx] and self.victim_last_seen_steps[victim_idx] >= 0:
                stale_steps = max(0, self.steps_taken - int(self.victim_last_seen_steps[victim_idx]))
                estimates[victim_idx] = self._clip_position(
                    self.victim_estimates[victim_idx] + stale_steps * self.victim_drifts[victim_idx]
                )
        return estimates

    def _update_victim_memory(self, detected_indices: list[list[int]], event_markers: list[str]) -> None:
        seen_victims = set()
        for indices in detected_indices:
            for victim_idx in indices:
                seen_victims.add(victim_idx)
        for victim_idx in range(self.config.num_victims):
            if victim_idx in seen_victims:
                self.victim_estimates[victim_idx] = self.victim_positions[victim_idx].copy()
                self.victim_known_mask[victim_idx] = True
                self.victim_last_seen_steps[victim_idx] = self.steps_taken
                if self.victim_first_detection_steps[victim_idx] is None:
                    self.victim_first_detection_steps[victim_idx] = self.steps_taken
                    event_markers.append(f"first_detection_v{victim_idx}")
            elif self.victim_known_mask[victim_idx]:
                self.victim_estimates[victim_idx] = self._clip_position(
                    self.victim_estimates[victim_idx] + self.victim_drifts[victim_idx]
                )

    def _explorer_reserve(self, tracked_counts: np.ndarray) -> int:
        coverage_mean = float(np.mean(self.coverage))
        uncertainty = 1.0 - coverage_mean
        active_tracks = int(np.sum(self.victim_known_mask))
        reserve_ratio = 0.3 + 0.35 * uncertainty
        reserve = int(np.ceil(self.config.num_uavs * reserve_ratio))
        if active_tracks >= max(1, self.config.num_uavs // 2):
            reserve = max(1, reserve - 1)
        return int(np.clip(reserve, 1, max(1, self.config.num_uavs - 1)))

    def _victim_priority(
        self,
        victim_idx: int,
        tracked_counts: np.ndarray,
        seen_counts: np.ndarray,
    ) -> float:
        unseen_steps = (
            self.steps_taken + 1
            if self.victim_last_seen_steps[victim_idx] < 0
            else self.steps_taken - int(self.victim_last_seen_steps[victim_idx])
        )
        discovery_bonus = 2.0 if not self.victim_known_mask[victim_idx] else 0.0
        stale_bonus = min(5.0, 0.22 * unseen_steps)
        undertrack_bonus = 1.1 * max(0, 1 - int(tracked_counts[victim_idx]))
        seen_bonus = 0.8 if seen_counts[victim_idx] > 0 else 0.0
        drift_bonus = 0.2 * float(np.linalg.norm(self.victim_drifts[victim_idx]))
        return discovery_bonus + stale_bonus + undertrack_bonus + seen_bonus + drift_bonus

    def _desired_trackers_for_victim(
        self,
        victim_idx: int,
        tracked_counts: np.ndarray,
        seen_counts: np.ndarray,
    ) -> int:
        if not self.victim_known_mask[victim_idx] and seen_counts[victim_idx] == 0:
            return 0

        desired = 1
        unseen_steps = (
            self.steps_taken + 1
            if self.victim_last_seen_steps[victim_idx] < 0
            else self.steps_taken - int(self.victim_last_seen_steps[victim_idx])
        )
        risk = 0
        if tracked_counts[victim_idx] == 0:
            risk += 1
        if unseen_steps >= 6:
            risk += 1
        if np.linalg.norm(self.victim_drifts[victim_idx]) >= 4.5:
            risk += 1
        if self.pending_handoffs[victim_idx] is not None:
            risk += 1
        if seen_counts[victim_idx] >= 2:
            risk += 1

        if risk >= 2:
            desired = 2
        return min(desired, 2)

    def _exploration_target(self, uav_idx: int, gap_target: np.ndarray) -> np.ndarray:
        lane_x = (uav_idx + 0.5) * (self.config.width / self.config.num_uavs)
        # Keep explorers aligned to their lane without coupling vertical motion
        # to the message staleness timer, which caused periodic target reversals.
        lane_y = float(gap_target[1])
        lane_anchor = np.array([lane_x, lane_y], dtype=float)
        target = 0.72 * gap_target + 0.28 * lane_anchor
        return self._clip_position(target)

    def _assignment_targets(
        self,
        detected_indices: list[list[int]],
        gap_targets: list[np.ndarray],
    ) -> tuple[list[np.ndarray | None], list[int]]:
        pairwise = self._pairwise_distances()
        tracked_counts = np.sum(pairwise <= self.config.track_distance, axis=0)
        seen_counts = np.zeros(self.config.num_victims, dtype=int)
        for indices in detected_indices:
            for victim_idx in indices:
                seen_counts[victim_idx] += 1

        predicted_estimates = self._predict_victim_estimates()
        priorities = [
            self._victim_priority(victim_idx, tracked_counts, seen_counts)
            for victim_idx in range(self.config.num_victims)
        ]

        assignments = [-1 for _ in range(self.config.num_uavs)]
        targets: list[np.ndarray | None] = [None for _ in range(self.config.num_uavs)]
        available = set(range(self.config.num_uavs))
        support_counts = np.zeros(self.config.num_victims, dtype=int)

        for uav_idx, indices in enumerate(detected_indices):
            if not indices:
                continue
            ranked_victims = sorted(
                indices,
                key=lambda victim_idx: priorities[victim_idx] - 0.002 * pairwise[uav_idx, victim_idx],
                reverse=True,
            )
            committed = False
            for victim_idx in ranked_victims:
                desired_trackers = self._desired_trackers_for_victim(victim_idx, tracked_counts, seen_counts)
                needs_support = (
                    not self.victim_known_mask[victim_idx]
                    or support_counts[victim_idx] < desired_trackers
                )
                if needs_support:
                    assignments[uav_idx] = int(victim_idx)
                    targets[uav_idx] = predicted_estimates[victim_idx].copy()
                    available.discard(uav_idx)
                    support_counts[victim_idx] += 1
                    committed = True
                    break
            if not committed:
                targets[uav_idx] = self._exploration_target(uav_idx, gap_targets[uav_idx])

        explorer_budget = min(len(available), self._explorer_reserve(tracked_counts))
        assignable_slots = max(0, len(available) - explorer_budget)

        victim_slots: list[int] = []
        for victim_idx in np.argsort(priorities)[::-1]:
            desired_trackers = self._desired_trackers_for_victim(victim_idx, tracked_counts, seen_counts)
            extra_needed = max(0, desired_trackers - int(support_counts[victim_idx]))
            victim_slots.extend([int(victim_idx)] * extra_needed)

        while available and assignable_slots > 0 and victim_slots:
            victim_idx = victim_slots.pop(0)
            best_uav = min(
                available,
                key=lambda uav_idx: np.linalg.norm(predicted_estimates[victim_idx] - self.uav_positions[uav_idx]),
            )
            assignments[best_uav] = int(victim_idx)
            targets[best_uav] = predicted_estimates[victim_idx].copy()
            available.remove(best_uav)
            assignable_slots -= 1
            support_counts[victim_idx] += 1

        for explorer_idx in sorted(available):
            targets[explorer_idx] = self._exploration_target(explorer_idx, gap_targets[explorer_idx])

        return targets, assignments

    def _victim_owner_from_assignments(self, assignments: list[int]) -> np.ndarray:
        owners = np.full(self.config.num_victims, -1, dtype=int)
        pairwise = self._pairwise_distances()
        for victim_idx in range(self.config.num_victims):
            assigned_uavs = [uav_idx for uav_idx, assigned in enumerate(assignments) if assigned == victim_idx]
            if assigned_uavs:
                owners[victim_idx] = min(
                    assigned_uavs,
                    key=lambda uav_idx: pairwise[uav_idx, victim_idx],
                )
        return owners

    def _builtin_actions(
        self,
        detections: np.ndarray,
        detected_indices: list[list[int]],
        peaks: list[np.ndarray],
        peak_confidences: list[float],
        gap_targets: list[np.ndarray],
    ) -> tuple[np.ndarray, list[int]]:
        actions = np.zeros_like(self.uav_positions)
        assigned_targets, assignments = self._assignment_targets(detected_indices, gap_targets)
        for idx in range(self.config.num_uavs):
            local_target_hint = assigned_targets[idx]
            if self.config.policy_mode == "sweep_only":
                action = self.policy.action(idx, self.uav_positions[idx], local_target_hint, self.rng)
            elif self.config.policy_mode == "frontier_cover":
                action = self.policy.action(idx, self.uav_positions[idx], local_target_hint, gap_targets[idx], self.rng)
            elif self.config.policy_mode == "hybrid_frontier_belief":
                action = self.policy.action(
                    idx,
                    self.uav_positions[idx],
                    local_target_hint,
                    peaks[idx],
                    peak_confidences[idx],
                    gap_targets[idx],
                    self.rng,
                )
            else:
                action = self.policy.action(
                    idx,
                    self.uav_positions[idx],
                    local_target_hint,
                    peaks[idx],
                    peak_confidences[idx],
                    gap_targets[idx],
                    self.rng,
                )
            actions[idx] = action
        return actions, assignments

    def current_builtin_actions(self) -> np.ndarray:
        detections, detected_indices = self._detect()
        peaks, peak_confidences = self._belief_peaks()
        gap_targets = [
            coverage_gap_target(self.coverage, self.config, idx, self.config.num_uavs)
            for idx in range(self.config.num_uavs)
        ]
        actions, _ = self._builtin_actions(detections, detected_indices, peaks, peak_confidences, gap_targets)
        return actions

    def _normalize_external_actions(self, external_actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(external_actions, dtype=float)
        if actions.shape != (self.config.num_uavs, 2):
            raise ValueError(
                f"Expected external actions shape {(self.config.num_uavs, 2)}, got {actions.shape}"
            )
        clipped = np.clip(actions, -1.0, 1.0)
        return clipped * self.config.max_speed

    def step(self, external_actions: np.ndarray | None = None) -> StepRecord:
        detections, detected_indices = self._detect()
        peaks, peak_confidences = self._belief_peaks()
        gap_targets = [
            coverage_gap_target(self.coverage, self.config, idx, self.config.num_uavs)
            for idx in range(self.config.num_uavs)
        ]
        builtin_actions, builtin_assignments = self._builtin_actions(detections, detected_indices, peaks, peak_confidences, gap_targets)

        actions = (
            self._normalize_external_actions(external_actions)
            if external_actions is not None
            else builtin_actions
        )
        active_assignments = builtin_assignments
        victim_owner_ids = self._victim_owner_from_assignments(active_assignments)

        for idx in range(self.config.num_uavs):
            action = actions[idx]
            if self.config.actuation_delay > 0.0:
                d = self.config.actuation_delay
                action = (1.0 - d) * action + d * self.uav_velocities[idx]
            self.uav_velocities[idx] = action
            self.uav_positions[idx] = self._clip_position(self.uav_positions[idx] + action)

        self._drift_victims()
        self.coverage = update_coverage(self.coverage, self.uav_positions, self.config)
        detections, detected_indices = self._detect()
        links = self._links()
        link_neighbors: dict[int, list[int]] = {idx: [] for idx in range(self.config.num_uavs)}
        for i, j in links:
            link_neighbors[i].append(j)
            link_neighbors[j].append(i)

        predicted_beliefs = [drift_predict(belief, self.config) for belief in self.beliefs]
        local_beliefs = [
            update_with_observation(
                predicted_beliefs[idx],
                self.uav_positions[idx],
                bool(detections[idx]),
                [self.victim_positions[victim_idx] for victim_idx in detected_indices[idx]],
                self.config,
            )
            for idx in range(self.config.num_uavs)
        ]

        updated_peaks, updated_peak_confidences = [], []
        for belief in local_beliefs:
            peak_xy, peak_confidence = belief_peak_xy(belief, self.config)
            updated_peaks.append(peak_xy)
            updated_peak_confidences.append(peak_confidence)

        transmitted_by = []
        outgoing_messages: dict[int, np.ndarray] = {}
        for idx in range(self.config.num_uavs):
            send = False
            if self.config.policy_mode in {"belief_sparse_comm", "hybrid_frontier_belief"}:
                send = self._should_transmit(idx, bool(detections[idx]), updated_peak_confidences[idx])
            transmitted_by.append(send)
            if send:
                outgoing_messages[idx] = local_beliefs[idx]
                self.total_messages_sent += 1
                self.last_message_steps[idx] = self.steps_taken

        fused_beliefs = []
        for idx in range(self.config.num_uavs):
            incoming = []
            for neighbor in link_neighbors[idx]:
                if neighbor in outgoing_messages:
                    # packet drop: skip delivery with probability packet_drop_rate
                    if self.config.packet_drop_rate > 0.0 and self.rng.random() < self.config.packet_drop_rate:
                        continue
                    incoming.append(outgoing_messages[neighbor])
                    self.total_messages_delivered += 1
            fused_beliefs.append(fuse_messages(local_beliefs[idx], incoming))
        self.beliefs = fused_beliefs
        _, final_peak_confidences = self._belief_peaks()
        self.last_peak_confidences = final_peak_confidences

        if np.any(detections) and self.first_detection_step is None:
            self.first_detection_step = self.steps_taken
            event_markers = ["first_detection"]
        else:
            event_markers = []
        self._update_victim_memory(detected_indices, event_markers)
        self.total_detections += sum(len(indices) for indices in detected_indices)
        tracking_mask = self._tracking_mask()
        tracking_active = bool(np.any(tracking_mask))
        tracked_victim_count = int(np.sum(tracking_mask))
        self.total_tracked_victims += tracked_victim_count
        for victim_idx in range(self.config.num_victims):
            was_tracking = bool(self.last_tracking_mask[victim_idx])
            is_tracking = bool(tracking_mask[victim_idx])
            if was_tracking and not is_tracking:
                self.track_loss_count += 1
                self.pending_reacquisition_start[victim_idx] = self.steps_taken
                event_markers.append(f"track_loss_v{victim_idx}")
            if (not was_tracking) and is_tracking and self.pending_reacquisition_start[victim_idx] is not None:
                self.reacquisition_delays.append(self.steps_taken - self.pending_reacquisition_start[victim_idx])
                self.pending_reacquisition_start[victim_idx] = None
                event_markers.append(f"reacquired_v{victim_idx}")

        for victim_idx in range(self.config.num_victims):
            previous_owner = int(self.victim_owner_ids[victim_idx])
            new_owner = int(victim_owner_ids[victim_idx])
            if (
                previous_owner >= 0
                and new_owner >= 0
                and previous_owner != new_owner
                and self.victim_known_mask[victim_idx]
            ):
                self.handoff_count += 1
                self.pending_handoffs[victim_idx] = {
                    "new_owner": new_owner,
                    "deadline": self.steps_taken + self.config.handoff_success_window,
                }
                event_markers.append(f"handoff_v{victim_idx}_{previous_owner}_to_{new_owner}")

        pairwise = self._pairwise_distances()
        for victim_idx in range(self.config.num_victims):
            pending = self.pending_handoffs[victim_idx]
            if pending is None:
                continue
            owner = pending["new_owner"]
            owner_tracking = owner >= 0 and pairwise[owner, victim_idx] <= self.config.track_distance
            if owner_tracking:
                self.handoff_success_count += 1
                self.pending_handoffs[victim_idx] = None
                event_markers.append(f"handoff_success_v{victim_idx}")
            elif self.steps_taken >= pending["deadline"]:
                self.handoff_failure_count += 1
                self.pending_handoffs[victim_idx] = None
                event_markers.append(f"handoff_fail_v{victim_idx}")

        self.victim_owner_ids = victim_owner_ids
        self.last_tracking_mask = tracking_mask
        if any(transmitted_by):
            event_markers.append("message_tx")

        avg_belief = np.mean(np.stack(self.beliefs, axis=0), axis=0)
        avg_peak_xy, _ = belief_peak_xy(avg_belief, self.config)
        global_gap_target = coverage_gap_target(self.coverage, self.config)

        record = StepRecord(
            step=self.steps_taken,
            uav_positions=self.uav_positions.round(3).tolist(),
            victim_positions=self.victim_positions.round(3).tolist(),
            detections=detections.tolist(),
            detected_victim_counts=[len(indices) for indices in detected_indices],
            assigned_victim_ids=active_assignments,
            victim_owner_ids=victim_owner_ids.tolist(),
            tracked_victim_count=tracked_victim_count,
            handoff_count=self.handoff_count,
            handoff_success_count=self.handoff_success_count,
            handoff_failure_count=self.handoff_failure_count,
            links=links,
            transmitted_by=transmitted_by,
            peak_confidences=[round(float(value), 4) for value in final_peak_confidences],
            average_belief_peak=avg_peak_xy.round(3).tolist(),
            coverage_gap_target=global_gap_target.round(3).tolist(),
            tracking_active=tracking_active,
            event_markers=event_markers,
        )
        self.history.append(record)
        self.steps_taken += 1
        return record

    def run(self) -> dict[str, Any]:
        self.reset()
        for _ in range(self.config.steps):
            self.step()
        return self.summary()

    def summary(self) -> dict[str, Any]:
        detection_rate = self.total_detections / (self.config.steps * self.config.num_victims * self.config.num_uavs)
        mean_reacquisition_delay = None
        if self.reacquisition_delays:
            mean_reacquisition_delay = round(float(np.mean(self.reacquisition_delays)), 4)
        return {
            "steps": self.config.steps,
            "num_uavs": self.config.num_uavs,
            "num_victims": self.config.num_victims,
            "first_detection_step": self.first_detection_step,
            "discovered_victim_ratio": round(
                float(sum(step is not None for step in self.victim_first_detection_steps) / self.config.num_victims),
                4,
            ),
            "detection_rate": round(float(detection_rate), 4),
            "tracking_ratio": round(float(self.total_tracked_victims / (self.config.steps * self.config.num_victims)), 4),
            "track_loss_count": self.track_loss_count,
            "handoff_count": self.handoff_count,
            "handoff_success_count": self.handoff_success_count,
            "handoff_failure_count": self.handoff_failure_count,
            "handoff_success_rate": round(
                float(self.handoff_success_count / self.handoff_count), 4
            ) if self.handoff_count > 0 else None,
            "reacquisition_delay_mean": mean_reacquisition_delay,
            "messages_sent": self.total_messages_sent,
            "messages_delivered": self.total_messages_delivered,
            "coverage_mean": round(float(np.mean(self.coverage)), 4),
            "final_victim_positions": self.victim_positions.round(3).tolist(),
        }

    def export_history(self, path: Path) -> None:
        payload = []
        for record in self.history:
            payload.append(
                {
                    "step": record.step,
                    "uav_positions": record.uav_positions,
                    "victim_positions": record.victim_positions,
                    "detections": record.detections,
                    "detected_victim_counts": record.detected_victim_counts,
                    "assigned_victim_ids": record.assigned_victim_ids,
                    "victim_owner_ids": record.victim_owner_ids,
                    "tracked_victim_count": record.tracked_victim_count,
                    "handoff_count": record.handoff_count,
                    "handoff_success_count": record.handoff_success_count,
                    "handoff_failure_count": record.handoff_failure_count,
                    "links": record.links,
                    "transmitted_by": record.transmitted_by,
                    "peak_confidences": record.peak_confidences,
                    "average_belief_peak": record.average_belief_peak,
                    "coverage_gap_target": record.coverage_gap_target,
                    "tracking_active": record.tracking_active,
                    "event_markers": record.event_markers,
                }
            )
        path.write_text(json.dumps(payload, indent=2))
