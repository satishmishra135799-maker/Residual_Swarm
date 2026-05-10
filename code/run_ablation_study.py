from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from experiment_config import ABLATION_SCALES, PAPER_BASE_POLICY, PAPER_SEEDS, SCENARIOS
from evaluation_utils import parse_seeds, summarize_runs
from swarm_sim.env_interface import SwarmLearningEnv
from swarm_sim.mlp_policy import policy_action
from swarm_sim.scenarios import scenario_config
from swarm_sim.simulator import MaritimeSwarmSimulator


OUTPUT_DIR = ROOT / "outputs"


def run_builtin(scenario_name: str, policy_mode: str, seed: int) -> dict:
    config = scenario_config(scenario_name, policy_mode)
    config = type(config)(**{**config.__dict__, "seed": seed})
    simulator = MaritimeSwarmSimulator(config)
    simulator.run()
    return simulator.summary()


def run_residual(theta: np.ndarray, scenario_name: str, seed: int, residual_scale: float) -> dict:
    config = scenario_config(scenario_name, PAPER_BASE_POLICY)
    config = type(config)(**{**config.__dict__, "seed": seed})
    env = SwarmLearningEnv(config, control_mode="residual", residual_scale=residual_scale)
    observations = env.reset()
    done = False
    final_info = {}
    while not done:
        actions = policy_action(theta, observations)
        observations, rewards, done, info = env.step(actions)
        final_info = info
    return final_info["summary_so_far"]


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    theta = np.load(OUTPUT_DIR / "residual_mlp_theta.npy")
    seeds = parse_seeds(sys.argv[1:]) or PAPER_SEEDS

    methods = {
        "frontier_cover": lambda s, seed: run_builtin(s, "frontier_cover", seed),
        PAPER_BASE_POLICY: lambda s, seed: run_builtin(s, PAPER_BASE_POLICY, seed),
        "residual_scale_0.0": lambda s, seed: run_builtin(s, PAPER_BASE_POLICY, seed),
    }
    for scale in ABLATION_SCALES:
        if scale == 0.0:
            continue
        methods[f"residual_scale_{scale}"] = lambda s, seed, current_scale=scale: run_residual(theta, s, seed, current_scale)

    payload: dict[str, dict[str, dict]] = {}
    for scenario_name in SCENARIOS:
        payload[scenario_name] = {}
        for method_name, runner in methods.items():
            runs = [runner(scenario_name, seed) for seed in seeds]
            payload[scenario_name][method_name] = {
                "summary": summarize_runs(runs),
                "runs": runs,
            }

    path = OUTPUT_DIR / "ablation_study.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"Ablation study written to {path}")


if __name__ == "__main__":
    main()
