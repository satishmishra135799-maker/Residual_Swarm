from __future__ import annotations

import numpy as np

from .config import SimulationConfig


def uniform_belief(grid_size: int) -> np.ndarray:
    belief = np.ones((grid_size, grid_size), dtype=float)
    return belief / belief.sum()


def belief_peak_xy(belief: np.ndarray, config: SimulationConfig) -> tuple[np.ndarray, float]:
    row, col = np.unravel_index(int(np.argmax(belief)), belief.shape)
    cell_w = config.width / config.grid_size
    cell_h = config.height / config.grid_size
    peak = np.array([(col + 0.5) * cell_w, (row + 0.5) * cell_h], dtype=float)
    return peak, float(belief[row, col])


def drift_predict(belief: np.ndarray, config: SimulationConfig) -> np.ndarray:
    shift_x = int(round(config.drift_x / max(config.width / config.grid_size, 1.0)))
    shift_y = int(round(config.drift_y / max(config.height / config.grid_size, 1.0)))
    shifted = np.roll(belief, shift=(shift_y, shift_x), axis=(0, 1))

    blurred = (
        shifted
        + np.roll(shifted, 1, axis=0)
        + np.roll(shifted, -1, axis=0)
        + np.roll(shifted, 1, axis=1)
        + np.roll(shifted, -1, axis=1)
    ) / 5.0
    return normalize((1.0 - config.belief_blend) * shifted + config.belief_blend * blurred)


def update_with_observation(
    belief: np.ndarray,
    position: np.ndarray,
    detected: bool,
    observed_positions: list[np.ndarray],
    config: SimulationConfig,
) -> np.ndarray:
    updated = belief.copy()
    mask = visible_mask(position, config)
    if detected and observed_positions:
        yy, xx = np.indices(updated.shape)
        gaussian = np.zeros_like(updated)
        for observed_position in observed_positions:
            target_row, target_col = point_to_cell(observed_position, config)
            gaussian += np.exp(-((yy - target_row) ** 2 + (xx - target_col) ** 2) / 3.0)
        updated = normalize(0.25 * updated + 0.75 * normalize(gaussian))
    else:
        updated[mask] *= config.negative_observation_decay
        updated = normalize(updated)
    return updated


def fuse_messages(local_belief: np.ndarray, incoming_beliefs: list[np.ndarray]) -> np.ndarray:
    if not incoming_beliefs:
        return local_belief

    fused = local_belief.copy()
    for incoming in incoming_beliefs:
        fused *= np.maximum(incoming, 1e-9)
    return normalize(fused)


def visible_mask(position: np.ndarray, config: SimulationConfig) -> np.ndarray:
    cell_w = config.width / config.grid_size
    cell_h = config.height / config.grid_size
    yy, xx = np.indices((config.grid_size, config.grid_size))
    centers_x = (xx + 0.5) * cell_w
    centers_y = (yy + 0.5) * cell_h
    distances = np.sqrt((centers_x - position[0]) ** 2 + (centers_y - position[1]) ** 2)
    return distances <= config.sensing_radius


def point_to_cell(point: np.ndarray, config: SimulationConfig) -> tuple[int, int]:
    cell_w = config.width / config.grid_size
    cell_h = config.height / config.grid_size
    col = int(np.clip(point[0] / cell_w, 0, config.grid_size - 1))
    row = int(np.clip(point[1] / cell_h, 0, config.grid_size - 1))
    return row, col


def normalize(belief: np.ndarray) -> np.ndarray:
    total = float(np.sum(belief))
    if total <= 0.0:
        return uniform_belief(belief.shape[0])
    return belief / total
