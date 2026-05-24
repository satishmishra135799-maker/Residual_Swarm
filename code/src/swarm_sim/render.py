from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .simulator import MaritimeSwarmSimulator


def _render_svg_from_record(
    simulator: MaritimeSwarmSimulator,
    path: Path,
    record_index: int,
    *,
    title: str | None = None,
) -> None:
    width = simulator.config.width
    height = simulator.config.height
    latest = simulator.history[record_index]
    sensed = latest.detections
    belief_slice_count = min(record_index + 1, len(simulator.history))
    avg_belief = np.mean(np.stack(simulator.beliefs, axis=0), axis=0)
    cell_w = width / simulator.config.grid_size
    cell_h = height / simulator.config.grid_size

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="700" height="700">',
        '<rect width="100%" height="100%" fill="#dff3ff" />',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="none" stroke="#0b3d91" stroke-width="6" />',
    ]

    if title:
        parts.append(
            f'<text x="30" y="45" font-size="30" fill="#0b1f33" font-weight="bold">{title}</text>'
        )
    parts.append(
        f'<text x="30" y="82" font-size="22" fill="#0b1f33">step={latest.step} tracking={"yes" if latest.tracking_active else "no"} victims={len(latest.victim_positions)} belief_frames={belief_slice_count}</text>'
    )
    if latest.event_markers:
        parts.append(
            f'<text x="30" y="112" font-size="22" fill="#b22222">events: {", ".join(latest.event_markers)}</text>'
        )

    for row in range(simulator.config.grid_size):
        for col in range(simulator.config.grid_size):
            value = float(avg_belief[row, col])
            opacity = min(0.45, value * simulator.config.grid_size * simulator.config.grid_size * 0.4)
            if opacity <= 0.02:
                continue
            x = col * cell_w
            y = row * cell_h
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="#ff9f1c" opacity="{opacity:.3f}" />'
            )

    for i, pos in enumerate(latest.uav_positions):
        x, y = pos
        color = "#008000" if sensed[i] else "#1f77b4"
        parts.append(
            f'<circle cx="{x}" cy="{y}" r="{simulator.config.sensing_radius}" fill="{color}" opacity="0.08" />'
        )
        parts.append(f'<circle cx="{x}" cy="{y}" r="12" fill="{color}" />')
        parts.append(
            f'<text x="{x + 16}" y="{y - 16}" font-size="22" fill="#111">UAV {i} ({latest.peak_confidences[i]:.2f}, seen={latest.detected_victim_counts[i]})</text>'
        )
        if latest.transmitted_by[i]:
            parts.append(f'<circle cx="{x}" cy="{y}" r="18" fill="none" stroke="#111" stroke-width="3" />')

    for i, j in latest.links:
        xi, yi = latest.uav_positions[i]
        xj, yj = latest.uav_positions[j]
        parts.append(
            f'<line x1="{xi}" y1="{yi}" x2="{xj}" y2="{yj}" stroke="#7f7f7f" stroke-dasharray="10 10" stroke-width="4" />'
        )

    victim_palette = ["#d62728", "#c2185b", "#ef6c00", "#6a1b9a", "#ad1457", "#00897b"]
    for idx, victim_position in enumerate(latest.victim_positions):
        vx, vy = victim_position
        victim_color = victim_palette[idx % len(victim_palette)]
        parts.append(f'<circle cx="{vx}" cy="{vy}" r="14" fill="{victim_color}" />')
        parts.append(
            f'<text x="{vx + 18}" y="{vy - 18}" font-size="20" fill="#111">Victim {idx}</text>'
        )
    bx, by = latest.average_belief_peak
    parts.append(f'<circle cx="{bx}" cy="{by}" r="12" fill="none" stroke="#ff9f1c" stroke-width="4" />')
    parts.append(
        f'<text x="{bx + 18}" y="{by + 24}" font-size="22" fill="#7a4e00">Belief peak</text>'
    )
    gx, gy = latest.coverage_gap_target
    parts.append(f'<circle cx="{gx}" cy="{gy}" r="10" fill="none" stroke="#6a4c93" stroke-width="4" />')
    parts.append(
        f'<text x="{gx + 18}" y="{gy + 20}" font-size="20" fill="#4b2e69">Coverage gap</text>'
    )

    parts.append("</svg>")
    path.write_text("\n".join(parts))


def render_svg(simulator: MaritimeSwarmSimulator, path: Path) -> None:
    _render_svg_from_record(simulator, path, len(simulator.history) - 1)


def render_replay_frames(
    simulator: MaritimeSwarmSimulator,
    output_dir: Path,
    *,
    every_n_steps: int = 5,
    include_last: bool = True,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_indices = list(range(0, len(simulator.history), every_n_steps))
    if include_last and simulator.history and frame_indices[-1] != len(simulator.history) - 1:
        frame_indices.append(len(simulator.history) - 1)

    frame_paths: list[Path] = []
    for frame_number, record_index in enumerate(frame_indices):
        path = output_dir / f"frame_{frame_number:03d}.svg"
        _render_svg_from_record(
            simulator,
            path,
            record_index,
            title="Maritime Swarm Replay",
        )
        frame_paths.append(path)
    return frame_paths


def write_replay_manifest(frame_paths: Iterable[Path], manifest_path: Path) -> None:
    lines = [str(path) for path in frame_paths]
    manifest_path.write_text("\n".join(lines) + ("\n" if lines else ""))
