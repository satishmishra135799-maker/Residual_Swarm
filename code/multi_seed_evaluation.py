from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from experiment_config import BASELINE_POLICIES, PAPER_SEEDS, SCENARIOS
from evaluation_utils import DEFAULT_SEEDS, parse_seeds, summarize_runs
from swarm_sim.scenarios import scenario_config
from swarm_sim.simulator import MaritimeSwarmSimulator


POLICIES = BASELINE_POLICIES


def run_once(scenario_name: str, policy_mode: str, seed: int) -> dict:
    config = scenario_config(scenario_name, policy_mode)
    config = type(config)(**{**config.__dict__, "seed": seed})
    simulator = MaritimeSwarmSimulator(config)
    return simulator.run()


def main() -> None:
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    seeds = parse_seeds(sys.argv[1:]) or PAPER_SEEDS

    aggregated: dict[str, dict[str, dict]] = {}
    detailed: dict[str, dict[str, list[dict]]] = {}
    for scenario_name in SCENARIOS:
        aggregated[scenario_name] = {}
        detailed[scenario_name] = {}
        for policy_mode in POLICIES:
            runs = [run_once(scenario_name, policy_mode, seed) for seed in seeds]
            detailed[scenario_name][policy_mode] = runs
            aggregated[scenario_name][policy_mode] = summarize_runs(runs)

    (output_dir / "multi_seed_summary.json").write_text(json.dumps(aggregated, indent=2))
    (output_dir / "multi_seed_detailed.json").write_text(json.dumps(detailed, indent=2))
    print("Multi-seed evaluation completed.")
    print(f"Seeds: {seeds if seeds else DEFAULT_SEEDS}")
    print(json.dumps(aggregated, indent=2))
    print(f"Summary file: {output_dir / 'multi_seed_summary.json'}")


if __name__ == "__main__":
    main()
