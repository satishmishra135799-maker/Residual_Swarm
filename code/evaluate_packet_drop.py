"""Robustness evaluation under packet drop rates: 0%, 20%, 40%, 60%."""
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
DROP_RATES = [0.0, 0.2, 0.4, 0.6]


def run_hybrid(scenario: str, seed: int, drop_rate: float) -> dict:
    cfg = scenario_config(scenario, PAPER_BASE_POLICY)
    cfg = type(cfg)(**{**cfg.__dict__, "seed": seed, "packet_drop_rate": drop_rate})
    return MaritimeSwarmSimulator(cfg).run()


def run_policy(theta: np.ndarray, scenario: str, seed: int, mode: str, drop_rate: float) -> dict:
    cfg = scenario_config(scenario, PAPER_BASE_POLICY)
    cfg = type(cfg)(**{**cfg.__dict__, "seed": seed, "packet_drop_rate": drop_rate})
    env = SwarmLearningEnv(cfg, control_mode=mode,
                           residual_scale=FROZEN_RESIDUAL_SCALE,
                           confidence_gate_threshold=0.08)
    obs = env.reset()
    done = False
    while not done:
        obs, _, done, info = env.step(policy_action(theta, obs))
    return info["summary_so_far"]


def main() -> None:
    fixed_theta = np.load(OUTPUT_DIR / "residual_mlp_theta.npy")
    soft_theta  = np.load(OUTPUT_DIR / "soft_gated_mlp_theta.npy")

    payload = {}
    print(f"{'Drop':>6} {'Scenario':>10} {'Method':>12} {'Tracking':>10} {'FirstDet':>10}")
    print("-" * 55)

    for drop in DROP_RATES:
        payload[str(drop)] = {}
        for scenario in SCENARIOS:
            payload[str(drop)][scenario] = {}
            for label, runner in [
                ("hybrid",  lambda s, sd, d=drop: run_hybrid(s, sd, d)),
                ("fixed",   lambda s, sd, d=drop, t=fixed_theta: run_policy(t, s, sd, "residual", d)),
                ("soft",    lambda s, sd, d=drop, t=soft_theta:  run_policy(t, s, sd, "soft_gated_residual", d)),
            ]:
                runs = [runner(scenario, seed) for seed in PAPER_SEEDS]
                summary = summarize_runs(runs)
                payload[str(drop)][scenario][label] = summary
                tr = summary["tracking_ratio"]["mean"]
                fd = summary["first_detection_step"]["mean"]
                print(f"{int(drop*100):>5}% {scenario:>10} {label:>12} {tr:>10.4f} {fd:>10.2f}")

    path = OUTPUT_DIR / "packet_drop_robustness.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()
