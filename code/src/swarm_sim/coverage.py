from __future__ import annotations

import numpy as np

from .config import SimulationConfig


def initial_coverage(grid_size: int) -> np.ndarray:
    return np.zeros((grid_size, grid_size), dtype=float)


def update_coverage(coverage: np.ndarray, positions: np.ndarray, config: SimulationConfig) -> np.ndarray:
    updated = coverage * config.coverage_decay
    cell_w = config.width / config.grid_size
    cell_h = config.height / config.grid_size
    yy, xx = np.indices(updated.shape)
    centers_x = (xx + 0.5) * cell_w
    centers_y = (yy + 0.5) * cell_h

    for position in positions:
        distances = np.sqrt((centers_x - position[0]) ** 2 + (centers_y - position[1]) ** 2)
        mask = distances <= config.sensing_radius
        updated[mask] = 1.0
    return updated


def coverage_gap_target(
    coverage: np.ndarray,
    config: SimulationConfig,
    uav_index: int | None = None,
    num_uavs: int | None = None,
) -> np.ndarray:
    if uav_index is not None and num_uavs:
        lane_width = coverage.shape[1] / num_uavs
        start = int(round(uav_index * lane_width))
        end = int(round((uav_index + 1) * lane_width))
        end = max(end, start + 1)
        local = coverage[:, start:end]
        local_row, local_col = np.unravel_index(int(np.argmin(local)), local.shape)
        row = local_row
        col = start + local_col
    else:
        row, col = np.unravel_index(int(np.argmin(coverage)), coverage.shape)
    cell_w = config.width / config.grid_size
    cell_h = config.height / config.grid_size
    return np.array([(col + 0.5) * cell_w, (row + 0.5) * cell_h], dtype=float)
