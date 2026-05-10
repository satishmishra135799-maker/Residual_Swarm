from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from experiment_config import FROZEN_RESIDUAL_SCALE, PAPER_BASE_POLICY, PAPER_SEEDS, SCENARIOS
from evaluation_utils import summarize_runs
from swarm_sim.env_interface import SwarmLearningEnv
from swarm_sim.mlp_policy import policy_action
from swarm_sim.scenarios import scenario_config
from swarm_sim.simulator import MaritimeSwarmSimulator

OUTPUT_DIR = ROOT / "outputs"
CONFIDENCE_GATE_THRESHOLD = 0.08


def run_hybrid(scenario: str, seed: int) -> dict:
    cfg = scenario_config(scenario, PAPER_BASE_POLICY)
    cfg = type(cfg)(**{**cfg.__dict__, "seed": seed})
    return MaritimeSwarmSimulator(cfg).run()


def run_policy(theta: np.ndarray, scenario: str, seed: int, mode: str) -> dict:
    cfg = scenario_config(scenario, PAPER_BASE_POLICY)
    cfg = type(cfg)(**{**cfg.__dict__, "seed": seed})
    env = SwarmLearningEnv(cfg, control_mode=mode,
                           residual_scale=FROZEN_RESIDUAL_SCALE,
                           confidence_gate_threshold=CONFIDENCE_GATE_THRESHOLD)
    obs = env.reset()
    done = False
    while not done:
        obs, _, done, info = env.step(policy_action(theta, obs))
    return info["summary_so_far"]


def run_pure_mlp(theta: np.ndarray, scenario: str, seed: int) -> dict:
    """Pure MLP: direct action, no base controller."""
    cfg = scenario_config(scenario, PAPER_BASE_POLICY)
    cfg = type(cfg)(**{**cfg.__dict__, "seed": seed})
    env = SwarmLearningEnv(cfg, control_mode="direct")
    obs = env.reset()
    done = False
    while not done:
        obs, _, done, info = env.step(policy_action(theta, obs))
    return info["summary_so_far"]


def main() -> None:
    fixed_theta = np.load(OUTPUT_DIR / "residual_mlp_theta.npy")
    gated_theta = np.load(OUTPUT_DIR / "gated_residual_mlp_theta.npy")
    pure_theta = np.load(OUTPUT_DIR / "pure_mlp_theta.npy")

    payload: dict = {}
    print(f"{'Scenario':<10} {'Method':<12} {'Tracking':>10} {'Miss':>8} {'FirstDet':>10} {'Coverage':>10}")
    print("-" * 55)

    for scenario in SCENARIOS:
        payload[scenario] = {}
        methods = [
            ("hybrid",   lambda s, seed: run_hybrid(s, seed)),
            ("pure_mlp", lambda s, seed, t=pure_theta: run_pure_mlp(t, s, seed)),
            ("fixed",    lambda s, seed, t=fixed_theta: run_policy(t, s, seed, "residual")),
            ("gated",    lambda s, seed, t=gated_theta: run_policy(t, s, seed, "gated_residual")),
        ]
        for label, runner in methods:
            runs = [runner(scenario, seed) for seed in PAPER_SEEDS]
            summary = summarize_runs(runs)
            payload[scenario][label] = {"summary": summary, "runs": runs}
            tr = summary["tracking_ratio"]["mean"]
            fd = summary["first_detection_step"]["mean"] if summary["first_detection_step"] else float("nan")
            cov = summary["coverage_mean"]["mean"]
            print(f"{scenario:<10} {label:<12} {tr:>10.4f} {1-tr:>8.4f} {fd:>10.2f} {cov:>10.4f}")

    path = OUTPUT_DIR / "full_baseline_comparison.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()
