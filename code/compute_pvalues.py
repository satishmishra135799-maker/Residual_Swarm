"""Compute paired t-tests for all metrics across all method pairs."""
from __future__ import annotations
import json, sys, numpy as np
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from experiment_config import FROZEN_RESIDUAL_SCALE, PAPER_BASE_POLICY, PAPER_SEEDS, SCENARIOS
from swarm_sim.env_interface import SwarmLearningEnv
from swarm_sim.mlp_policy import policy_action
from swarm_sim.scenarios import scenario_config
from swarm_sim.simulator import MaritimeSwarmSimulator

OUTPUT_DIR = ROOT / "outputs"


def run_hybrid(scenario, seed):
    cfg = scenario_config(scenario, PAPER_BASE_POLICY)
    cfg = type(cfg)(**{**cfg.__dict__, "seed": seed})
    return MaritimeSwarmSimulator(cfg).run()


def run_policy(theta, scenario, seed, mode, tau=0.08):
    cfg = scenario_config(scenario, PAPER_BASE_POLICY)
    cfg = type(cfg)(**{**cfg.__dict__, "seed": seed})
    env = SwarmLearningEnv(cfg, control_mode=mode,
                           residual_scale=FROZEN_RESIDUAL_SCALE,
                           confidence_gate_threshold=tau)
    obs = env.reset()
    done = False
    while not done:
        obs, _, done, info = env.step(policy_action(theta, obs))
    return info["summary_so_far"]


def collect(scenario, mode, theta=None, tau=0.08):
    results = []
    for seed in PAPER_SEEDS:
        r = run_hybrid(scenario, seed) if mode == "hybrid" else run_policy(theta, scenario, seed, mode, tau)
        results.append(r)
    return results


def pval(a, b):
    _, p = stats.ttest_rel(a, b)
    return p


def main():
    fixed_t = np.load(OUTPUT_DIR / "residual_mlp_theta.npy")
    gated_t = np.load(OUTPUT_DIR / "gated_residual_mlp_theta.npy")

    print(f"{'Scenario':<10} {'Metric':<20} {'Hybrid_mean':>12} {'Fixed_mean':>12} {'p-value':>10} {'Sig':>5}")
    print("-" * 75)

    for scenario in SCENARIOS:
        hybrid_runs = collect(scenario, "hybrid")
        fixed_runs  = collect(scenario, "residual", fixed_t)
        gated_runs  = collect(scenario, "gated_residual", gated_t)

        for metric in ["tracking_ratio", "first_detection_step", "coverage_mean"]:
            h_vals = [r.get(metric) for r in hybrid_runs if r.get(metric) is not None]
            f_vals = [r.get(metric) for r in fixed_runs  if r.get(metric) is not None]
            g_vals = [r.get(metric) for r in gated_runs  if r.get(metric) is not None]

            if len(h_vals) == len(f_vals) == 30:
                p = pval(h_vals, f_vals)
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
                print(f"{scenario:<10} {metric:<20} {np.mean(h_vals):>12.4f} {np.mean(f_vals):>12.4f} {p:>10.4f} {sig:>5}")

    # Save full results
    payload = {}
    for scenario in SCENARIOS:
        payload[scenario] = {}
        hybrid_runs = collect(scenario, "hybrid")
        fixed_runs  = collect(scenario, "residual", fixed_t)
        for metric in ["tracking_ratio", "first_detection_step", "coverage_mean", "reacquisition_delay_mean"]:
            h = [r.get(metric) for r in hybrid_runs if r.get(metric) is not None]
            f = [r.get(metric) for r in fixed_runs  if r.get(metric) is not None]
            if h and f and len(h) == len(f):
                _, p = stats.ttest_rel(h, f)
                payload[scenario][metric] = {
                    "hybrid_mean": round(float(np.mean(h)), 4),
                    "fixed_mean":  round(float(np.mean(f)), 4),
                    "p_value":     round(float(p), 6),
                }

    (OUTPUT_DIR / "pvalue_analysis.json").write_text(json.dumps(payload, indent=2))
    print(f"\nSaved to {OUTPUT_DIR / 'pvalue_analysis.json'}")


if __name__ == "__main__":
    main()
