from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class RandomSearchPolicy:
    max_speed: float

    def action(self, position: np.ndarray, target_hint: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
        if target_hint is not None:
            delta = target_hint - position
            norm = np.linalg.norm(delta)
            if norm > 1e-6:
                return (delta / norm) * self.max_speed

        angle = rng.uniform(0.0, 2.0 * math.pi)
        return np.array([math.cos(angle), math.sin(angle)]) * self.max_speed


@dataclass
class SweepSearchPolicy:
    max_speed: float
    width: float
    height: float
    num_uavs: int

    def action(
        self,
        uav_index: int,
        position: np.ndarray,
        target_hint: np.ndarray | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if target_hint is not None:
            delta = target_hint - position
            norm = np.linalg.norm(delta)
            if norm > 1e-6:
                return (delta / norm) * self.max_speed

        lane_x = (uav_index + 0.5) * (self.width / self.num_uavs)
        target_y = self.height * 0.9 if (int(position[1] / max(self.height * 0.25, 1.0)) % 2 == 0) else self.height * 0.1
        target = np.array([lane_x, target_y])
        delta = target - position
        norm = np.linalg.norm(delta)
        if norm < 20.0:
            jitter = rng.normal(0.0, 10.0, size=2)
            target = np.clip(target + jitter, [0.0, 0.0], [self.width, self.height])
            delta = target - position
            norm = np.linalg.norm(delta)
        if norm > 1e-6:
            return (delta / norm) * self.max_speed
        return np.zeros(2)


@dataclass
class BeliefAwarePolicy:
    max_speed: float
    width: float
    height: float
    num_uavs: int

    def action(
        self,
        uav_index: int,
        position: np.ndarray,
        target_hint: np.ndarray | None,
        belief_peak: np.ndarray,
        peak_confidence: float,
        coverage_target: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if target_hint is not None:
            delta = target_hint - position
            norm = np.linalg.norm(delta)
            if norm > 1e-6:
                return (delta / norm) * self.max_speed

        lane_x = (uav_index + 0.5) * (self.width / self.num_uavs)
        target_y = self.height * 0.85 if (int(position[1] / max(self.height * 0.25, 1.0)) % 2 == 0) else self.height * 0.15
        sweep_target = np.array([lane_x, target_y])
        if peak_confidence > 0.06:
            target = 0.6 * belief_peak + 0.25 * coverage_target + 0.15 * sweep_target
        else:
            target = 0.55 * coverage_target + 0.45 * sweep_target
        delta = target - position
        norm = np.linalg.norm(delta)
        if norm < 20.0:
            jitter = rng.normal(0.0, 8.0, size=2)
            target = np.clip(target + jitter, [0.0, 0.0], [self.width, self.height])
            delta = target - position
            norm = np.linalg.norm(delta)
        if norm > 1e-6:
            return (delta / norm) * self.max_speed
        return np.zeros(2)


@dataclass
class FrontierCoveragePolicy:
    max_speed: float
    width: float
    height: float
    num_uavs: int

    def action(
        self,
        uav_index: int,
        position: np.ndarray,
        target_hint: np.ndarray | None,
        coverage_target: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if target_hint is not None:
            delta = target_hint - position
            norm = np.linalg.norm(delta)
            if norm > 1e-6:
                return (delta / norm) * self.max_speed

        lane_x = (uav_index + 0.5) * (self.width / self.num_uavs)
        lane_anchor = np.array([lane_x, coverage_target[1]])
        target = 0.7 * coverage_target + 0.3 * lane_anchor
        delta = target - position
        norm = np.linalg.norm(delta)
        if norm < 20.0:
            jitter = rng.normal(0.0, 8.0, size=2)
            target = np.clip(target + jitter, [0.0, 0.0], [self.width, self.height])
            delta = target - position
            norm = np.linalg.norm(delta)
        if norm > 1e-6:
            return (delta / norm) * self.max_speed
        return np.zeros(2)


@dataclass
class HybridSearchPolicy:
    max_speed: float
    width: float
    height: float
    num_uavs: int

    def action(
        self,
        uav_index: int,
        position: np.ndarray,
        target_hint: np.ndarray | None,
        belief_peak: np.ndarray,
        peak_confidence: float,
        coverage_target: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if target_hint is not None:
            delta = target_hint - position
            norm = np.linalg.norm(delta)
            if norm > 1e-6:
                return (delta / norm) * self.max_speed

        lane_x = (uav_index + 0.5) * (self.width / self.num_uavs)
        lane_anchor = np.array([lane_x, coverage_target[1]])

        if peak_confidence > 0.08:
            target = 0.55 * belief_peak + 0.30 * coverage_target + 0.15 * lane_anchor
        else:
            target = 0.75 * coverage_target + 0.25 * lane_anchor

        delta = target - position
        norm = np.linalg.norm(delta)
        if norm < 20.0:
            jitter = rng.normal(0.0, 8.0, size=2)
            target = np.clip(target + jitter, [0.0, 0.0], [self.width, self.height])
            delta = target - position
            norm = np.linalg.norm(delta)
        if norm > 1e-6:
            return (delta / norm) * self.max_speed
        return np.zeros(2)
