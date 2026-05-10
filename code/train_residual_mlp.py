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
TRAIN_SCENARIOS = SCENARIOS
EVAL_SCENARIOS = SCENARIOS
FINAL_SEEDS = VALIDATION_SEEDS
SCENARIO_WEIGHTS = {"default": 0.8, "medium": 1.0, "hard": 1.35}
ITERATIONS = int(os.environ.get("SWARM_TRAIN_ITERATIONS", "10"))
POPULATION_SIZE = int(os.environ.get("SWARM_TRAIN_POPULATION", "12"))
ELITE_COUNT = int(os.environ.get("SWARM_TRAIN_ELITES", "4"))


def rollout_score(theta: np.ndarray, scenario_name: str, seed: int) -> tuple[float, dict]:
    env = SwarmLearningEnv(
        type(scenario_config(scenario_name, PAPER_BASE_POLICY))(
            **{**scenario_config(scenario_name, PAPER_BASE_POLICY).__dict__, "seed": seed}
        ),
        control_mode="residual",
        residual_scale=FROZEN_RESIDUAL_SCALE,
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


def evaluate(theta: np.ndarray, scenario_names: list[str], seeds: list[int]) -> tuple[float, dict[str, list[dict]]]:
    scores = []
    summaries_by_scenario: dict[str, list[dict]] = {name: [] for name in scenario_names}
    for scenario_name in scenario_names:
        for seed in seeds:
            score, summary = rollout_score(theta, scenario_name, seed)
            scores.append(score)
            summaries_by_scenario[scenario_name].append(summary)
    return float(np.mean(scores)), summaries_by_scenario


def scenario_metrics(summaries_by_scenario: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for scenario_name, summaries in summaries_by_scenario.items():
        metrics[scenario_name] = {
            "detection_rate_mean": round(float(np.mean([summary.get("detection_rate", 0.0) for summary in summaries])), 4),
            "tracking_ratio_mean": round(float(np.mean([summary.get("tracking_ratio", 0.0) for summary in summaries])), 4),
            "coverage_mean": round(float(np.mean([summary.get("coverage_mean", 0.0) for summary in summaries])), 4),
        }
    return metrics


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(91)
    mean = np.zeros(theta_dim(), dtype=float)
    std = np.full(theta_dim(), 0.12)
    best_theta = mean.copy()
    best_score, best_summaries = evaluate(best_theta, TRAIN_SCENARIOS, TRAIN_SEEDS)
    history: list[dict] = []

    for iteration in range(ITERATIONS):
        population = mean + std * rng.normal(size=(POPULATION_SIZE, theta_dim()))
        scored = []
        for candidate in population:
            score, summaries = evaluate(candidate, TRAIN_SCENARIOS, TRAIN_SEEDS)
            scored.append((score, candidate, summaries))
        scored.sort(key=lambda item: item[0], reverse=True)
        elites = scored[:ELITE_COUNT]
        elite_thetas = np.stack([item[1] for item in elites], axis=0)
        mean = np.mean(elite_thetas, axis=0)
        std = np.maximum(np.std(elite_thetas, axis=0), 0.03)

        if elites[0][0] > best_score:
            best_score = elites[0][0]
            best_theta = elites[0][1].copy()
            best_summaries = elites[0][2]

        history.append({
            "iteration": iteration,
            "best_score": round(best_score, 4),
            "scenario_metrics": scenario_metrics(best_summaries),
        })
        print(
            json.dumps(
                {
                    "iteration": iteration,
                    "best_score": round(best_score, 4),
                    "hard_tracking": history[-1]["scenario_metrics"]["hard"]["tracking_ratio_mean"],
                }
            ),
            flush=True,
        )

    final_score, final_summaries = evaluate(best_theta, EVAL_SCENARIOS, FINAL_SEEDS)
    payload = {
        "best_score": round(final_score, 4),
        "history": history,
        "train_scenarios": TRAIN_SCENARIOS,
        "train_seeds": TRAIN_SEEDS,
        "frozen_residual_scale": FROZEN_RESIDUAL_SCALE,
        "final_scenario_metrics": scenario_metrics(final_summaries),
        "final_summaries": final_summaries,
    }
    (OUTPUT_DIR / "residual_mlp_training.json").write_text(json.dumps(payload, indent=2))
    np.save(OUTPUT_DIR / "residual_mlp_theta.npy", best_theta)
    print("Residual MLP training finished.")


if __name__ == "__main__":
    main()
