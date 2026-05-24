from __future__ import annotations

PAPER_SEEDS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39,
               43, 47, 51, 55, 59, 63, 67, 71, 75, 79,
               83, 87, 91, 95, 99, 103, 107, 111, 115, 119]
TRAIN_SEEDS = [3, 7, 11, 15, 19, 23, 27]
VALIDATION_SEEDS = [3, 7, 11, 15, 19]

SCENARIOS = ["default", "medium", "hard"]
BASELINE_POLICIES = ["sweep_only", "frontier_cover", "hybrid_frontier_belief", "belief_sparse_comm"]
PAPER_BASE_POLICY = "hybrid_frontier_belief"

FROZEN_RESIDUAL_SCALE = 0.25
ABLATION_SCALES = [0.0, 0.1, 0.25, 0.4]

CONSISTENCY_METRICS = [
    "detection_rate",
    "tracking_ratio",
    "coverage_mean",
    "first_detection_step",
]

CONSISTENCY_TOLERANCE = 1e-9
