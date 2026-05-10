from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from experiment_config import PAPER_SEEDS

DEFAULT_SEEDS = PAPER_SEEDS
NUMERIC_METRICS = [
    "detection_rate",
    "tracking_ratio",
    "discovered_victim_ratio",
    "track_loss_count",
    "handoff_count",
    "handoff_success_count",
    "handoff_failure_count",
    "handoff_success_rate",
    "messages_sent",
    "messages_delivered",
    "coverage_mean",
]


def mean_std_ci(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    sample_count = int(array.size)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if sample_count > 1 else 0.0
    ci_half_width = 1.96 * std / math.sqrt(sample_count) if sample_count > 1 else 0.0
    return {
        "count": sample_count,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(float(np.min(array)), 4),
        "max": round(float(np.max(array)), 4),
        "ci95_low": round(mean - ci_half_width, 4),
        "ci95_high": round(mean + ci_half_width, 4),
    }


def summarize_runs(runs: list[dict]) -> dict:
    summary: dict[str, float | int | None | dict] = {
        "run_count": len(runs),
    }
    for metric in NUMERIC_METRICS:
        values = [float(run[metric]) for run in runs if run.get(metric) is not None]
        summary[metric] = mean_std_ci(values) if values else None

    first_detection_values = [float(run["first_detection_step"]) for run in runs if run["first_detection_step"] is not None]
    summary["first_detection_step"] = mean_std_ci(first_detection_values) if first_detection_values else None

    reacquisition_values = [
        float(run["reacquisition_delay_mean"])
        for run in runs
        if run["reacquisition_delay_mean"] is not None
    ]
    summary["reacquisition_delay_mean"] = mean_std_ci(reacquisition_values) if reacquisition_values else None
    summary["runs_with_detection"] = sum(1 for run in runs if run["first_detection_step"] is not None)
    return summary


def parse_seeds(argv: list[str]) -> list[int]:
    if not argv:
        return DEFAULT_SEEDS
    return [int(value) for value in argv]


def summary_mean(summary: dict, metric: str) -> float | None:
    metric_summary = summary.get(metric)
    if metric_summary is None:
        return None
    return metric_summary["mean"]


def summary_std(summary: dict, metric: str) -> float | None:
    metric_summary = summary.get(metric)
    if metric_summary is None:
        return None
    return metric_summary["std"]


def summary_ci(summary: dict, metric: str) -> tuple[float, float] | None:
    metric_summary = summary.get(metric)
    if metric_summary is None:
        return None
    return metric_summary["ci95_low"], metric_summary["ci95_high"]
