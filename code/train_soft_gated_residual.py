from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from experiment_config import FROZEN_RESIDUAL_SCALE, PAPER_BASE_POLICY, SCENARIOS, TRAIN_SEEDS, VALIDATION_SEEDS
from swarm_sim.env_interface import SwarmLearningEnv
from swarm_sim.mlp_policy import policy_action, theta_dim
from swarm_sim.scenarios import scenario_config

OUTPUT_DIR = ROOT / "outputs"
SCENARIO_WEIGHTS = {"default": 0.8, "medium": 1.0, "hard": 1.35}
ITERATIONS = int(os.environ.get("SWARM_TRAIN_ITERATIONS", "50"))
POPULATION_SIZE = int(os.environ.get("SWARM_TRAIN_POPULATION", "12"))
ELITE_COUNT = int(os.environ.get("SWARM_TRAIN_ELITES", "4"))
CONFIDENCE_GATE_THRESHOLD = 0.08


def rollout_score(theta: np.ndarray, scenario_name: str, seed: int) -> tuple[float, dict]:
    env = SwarmLearningEnv(
        type(scenario_config(scenario_name, PAPER_BASE_POLICY))(
            **{**scenario_config(scenario_name, PAPER_BASE_POLICY).__dict__, "seed": seed}
        ),
        control_mode="soft_gated_residual",
        residual_scale=FROZEN_RESIDUAL_SCALE,
        confidence_gate_threshold=CONFIDENCE_GATE_THRESHOLD,
    )
    observations = env.reset()
    total_reward = 0.0
    done = False
    final_info = {}
    while not done:
        actions = policy_action(theta, observations)
        observations, rewards, done, info = env.step(actions)
        total_reward += float(np.mean(rewards))
        final_info = info
    shaped_score = total_reward
    summary = final_info["summary_so_far"]
    shaped_score += 70.0 * float(summary.get("tracking_ratio", 0.0))
    shaped_score += 45.0 * float(summary.get("detection_rate", 0.0))
    shaped_score += 20.0 * float(summary.get("coverage_mean", 0.0))
    first_detection = summary.get("first_detection_step")
    if first_detection is not None:
        shaped_score += max(0.0, 25.0 - float(first_detection))
    shaped_score -= 1.5 * float(summary.get("track_loss_count", 0.0))
    return shaped_score * SCENARIO_WEIGHTS[scenario_name], summary


def evaluate(theta: np.ndarray) -> tuple[float, dict]:
    scores, summaries = [], {s: [] for s in SCENARIOS}
    for scenario in SCENARIOS:
        for seed in TRAIN_SEEDS:
            score, summary = rollout_score(theta, scenario, seed)
            scores.append(score)
            summaries[scenario].append(summary)
    return float(np.mean(scores)), summaries


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(93)
    mean = np.zeros(theta_dim(), dtype=float)
    std = np.full(theta_dim(), 0.12)
    best_theta = mean.copy()
    best_score, best_summaries = evaluate(best_theta)
    history = []

    for iteration in range(ITERATIONS):
        population = mean + std * rng.normal(size=(POPULATION_SIZE, theta_dim()))
        scored = [(evaluate(c)[0], c, evaluate(c)[1]) for c in population]
        # re-evaluate properly (avoid double eval above)
        scored = []
        for c in population:
            s, summ = evaluate(c)
            scored.append((s, c, summ))
        scored.sort(key=lambda x: x[0], reverse=True)
        elites = scored[:ELITE_COUNT]
        elite_thetas = np.stack([e[1] for e in elites])
        mean = np.mean(elite_thetas, axis=0)
        std = np.maximum(np.std(elite_thetas, axis=0), 0.03)

        if elites[0][0] > best_score:
            best_score, best_theta, best_summaries = elites[0][0], elites[0][1].copy(), elites[0][2]

        hard_track = float(np.mean([s.get("tracking_ratio", 0) for s in best_summaries["hard"]]))
        history.append({"iteration": iteration, "best_score": round(best_score, 4),
                        "hard_tracking": round(hard_track, 4)})
        print(json.dumps(history[-1]), flush=True)

    # final eval on validation seeds
    final_scores, final_summaries = [], {s: [] for s in SCENARIOS}
    for scenario in SCENARIOS:
        for seed in VALIDATION_SEEDS:
            score, summary = rollout_score(best_theta, scenario, seed)
            final_scores.append(score)
            final_summaries[scenario].append(summary)

    payload = {
        "best_score": round(float(np.mean(final_scores)), 4),
        "history": history,
        "train_scenarios": SCENARIOS,
        "train_seeds": TRAIN_SEEDS,
        "confidence_gate_threshold": CONFIDENCE_GATE_THRESHOLD,
        "final_summaries": final_summaries,
    }
    (OUTPUT_DIR / "soft_gated_mlp_training.json").write_text(json.dumps(payload, indent=2))
    np.save(OUTPUT_DIR / "soft_gated_mlp_theta.npy", best_theta)
    print("Soft-gated residual training finished.")


if __name__ == "__main__":
    main()
