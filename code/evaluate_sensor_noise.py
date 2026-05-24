"""
Robustness evaluation under sensor noise and actuation delay.
Tests trained models (hybrid, fixed residual, soft-gated) under:
- GPS noise: 0, 2, 5m std
- Detection decay: 0, 0.5, 1.0 (probability falls off with distance)
- Actuation delay: 0, 0.2, 0.4 (first-order velocity lag)

Models are NOT retrained — this tests generalization to unseen conditions.
"""
from __future__ import annotations
import json, sys, time, dataclasses
from pathlib import Path
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

# Noise conditions to test
GPS_NOISE_LEVELS = [0.0, 2.0, 5.0]          # meters std
DETECTION_DECAY_LEVELS = [0.0, 0.5, 1.0]    # exponential decay rate
ACTUATION_DELAY_LEVELS = [0.0, 0.2, 0.4]    # lag coefficient


def run_hybrid(scenario: str, seed: int, gps_noise: float, det_decay: float, act_delay: float) -> dict:
    cfg = scenario_config(scenario, PAPER_BASE_POLICY)
    cfg = dataclasses.replace(cfg, seed=seed,
                              gps_noise_std=gps_noise,
                              detection_decay_rate=det_decay,
                              actuation_delay=act_delay)
    return MaritimeSwarmSimulator(cfg).run()


def run_policy(theta: np.ndarray, scenario: str, seed: int, mode: str,
               gps_noise: float, det_decay: float, act_delay: float) -> dict:
    cfg = scenario_config(scenario, PAPER_BASE_POLICY)
    cfg = dataclasses.replace(cfg, seed=seed,
                              gps_noise_std=gps_noise,
                              detection_decay_rate=det_decay,
                              actuation_delay=act_delay)
    env = SwarmLearningEnv(cfg, control_mode=mode,
                           residual_scale=FROZEN_RESIDUAL_SCALE,
                           confidence_gate_threshold=0.08)
    obs = env.reset()
    done = False
    while not done:
        obs, _, done, info = env.step(policy_action(theta, obs))
    return info["summary_so_far"]


def run_combined_noise_study():
    """Test all methods under combined realistic noise conditions."""
    fixed_theta = np.load(OUTPUT_DIR / "residual_mlp_theta.npy")
    soft_theta = np.load(OUTPUT_DIR / "soft_gated_mlp_theta.npy")

    # Combined noise profiles (realistic scenarios)
    noise_profiles = [
        {"label": "ideal",    "gps": 0.0, "det": 0.0, "act": 0.0},
        {"label": "mild",     "gps": 2.0, "det": 0.3, "act": 0.1},
        {"label": "moderate", "gps": 3.0, "det": 0.5, "act": 0.2},
        {"label": "severe",   "gps": 5.0, "det": 1.0, "act": 0.4},
    ]

    payload = {}
    print(f"\n{'='*70}")
    print("ROBUSTNESS STUDY: Sensor Noise & Actuation Delay")
    print(f"{'='*70}")
    print(f"{'Profile':>10} {'Scenario':>8} {'Method':>8} {'Track':>8} {'FDet':>6} {'Cov':>6}")
    print("-" * 70)

    for profile in noise_profiles:
        label = profile["label"]
        payload[label] = {}
        for scenario in SCENARIOS:
            payload[label][scenario] = {}
            for method_name, runner in [
                ("hybrid", lambda s, sd: run_hybrid(s, sd, profile["gps"], profile["det"], profile["act"])),
                ("fixed",  lambda s, sd: run_policy(fixed_theta, s, sd, "residual",
                                                     profile["gps"], profile["det"], profile["act"])),
                ("soft",   lambda s, sd: run_policy(soft_theta, s, sd, "soft_gated_residual",
                                                     profile["gps"], profile["det"], profile["act"])),
            ]:
                runs = [runner(scenario, seed) for seed in PAPER_SEEDS]
                summary = summarize_runs(runs)
                payload[label][scenario][method_name] = summary
                tr = summary["tracking_ratio"]["mean"]
                fd = summary["first_detection_step"]["mean"] if summary["first_detection_step"] else -1
                cov = summary["coverage_mean"]["mean"]
                print(f"{label:>10} {scenario:>8} {method_name:>8} {tr:>8.3f} {fd:>6.1f} {cov:>6.3f}")

    return payload


def run_individual_noise_study():
    """Test each noise dimension independently (isolate effects)."""
    fixed_theta = np.load(OUTPUT_DIR / "residual_mlp_theta.npy")
    soft_theta = np.load(OUTPUT_DIR / "soft_gated_mlp_theta.npy")

    scenario = "hard"  # Most challenging, most informative
    payload = {"gps_noise": {}, "detection_decay": {}, "actuation_delay": {}}

    print(f"\n{'='*70}")
    print("INDIVIDUAL NOISE DIMENSIONS (hard scenario only)")
    print(f"{'='*70}")

    # GPS noise sweep
    print(f"\n--- GPS Noise (σ meters) ---")
    print(f"{'GPS σ':>8} {'Hybrid':>8} {'Fixed':>8} {'Soft':>8}")
    for gps in GPS_NOISE_LEVELS:
        results = {}
        for method_name, runner in [
            ("hybrid", lambda sd: run_hybrid(scenario, sd, gps, 0.0, 0.0)),
            ("fixed",  lambda sd: run_policy(fixed_theta, scenario, sd, "residual", gps, 0.0, 0.0)),
            ("soft",   lambda sd: run_policy(soft_theta, scenario, sd, "soft_gated_residual", gps, 0.0, 0.0)),
        ]:
            runs = [runner(seed) for seed in PAPER_SEEDS]
            summary = summarize_runs(runs)
            results[method_name] = summary
        payload["gps_noise"][str(gps)] = results
        print(f"{gps:>8.1f} {results['hybrid']['tracking_ratio']['mean']:>8.3f} "
              f"{results['fixed']['tracking_ratio']['mean']:>8.3f} "
              f"{results['soft']['tracking_ratio']['mean']:>8.3f}")

    # Detection decay sweep
    print(f"\n--- Detection Decay Rate ---")
    print(f"{'Rate':>8} {'Hybrid':>8} {'Fixed':>8} {'Soft':>8}")
    for det in DETECTION_DECAY_LEVELS:
        results = {}
        for method_name, runner in [
            ("hybrid", lambda sd: run_hybrid(scenario, sd, 0.0, det, 0.0)),
            ("fixed",  lambda sd: run_policy(fixed_theta, scenario, sd, "residual", 0.0, det, 0.0)),
            ("soft",   lambda sd: run_policy(soft_theta, scenario, sd, "soft_gated_residual", 0.0, det, 0.0)),
        ]:
            runs = [runner(seed) for seed in PAPER_SEEDS]
            summary = summarize_runs(runs)
            results[method_name] = summary
        payload["detection_decay"][str(det)] = results
        print(f"{det:>8.1f} {results['hybrid']['tracking_ratio']['mean']:>8.3f} "
              f"{results['fixed']['tracking_ratio']['mean']:>8.3f} "
              f"{results['soft']['tracking_ratio']['mean']:>8.3f}")

    # Actuation delay sweep
    print(f"\n--- Actuation Delay (lag coefficient) ---")
    print(f"{'Delay':>8} {'Hybrid':>8} {'Fixed':>8} {'Soft':>8}")
    for act in ACTUATION_DELAY_LEVELS:
        results = {}
        for method_name, runner in [
            ("hybrid", lambda sd: run_hybrid(scenario, sd, 0.0, 0.0, act)),
            ("fixed",  lambda sd: run_policy(fixed_theta, scenario, sd, "residual", 0.0, 0.0, act)),
            ("soft",   lambda sd: run_policy(soft_theta, scenario, sd, "soft_gated_residual", 0.0, 0.0, act)),
        ]:
            runs = [runner(seed) for seed in PAPER_SEEDS]
            summary = summarize_runs(runs)
            results[method_name] = summary
        payload["actuation_delay"][str(act)] = results
        print(f"{act:>8.1f} {results['hybrid']['tracking_ratio']['mean']:>8.3f} "
              f"{results['fixed']['tracking_ratio']['mean']:>8.3f} "
              f"{results['soft']['tracking_ratio']['mean']:>8.3f}")

    return payload


def main():
    t0 = time.time()

    combined = run_combined_noise_study()
    individual = run_individual_noise_study()

    results = {
        "combined_profiles": combined,
        "individual_sweeps": individual,
        "config": {
            "seeds": PAPER_SEEDS,
            "n_seeds": len(PAPER_SEEDS),
            "scenarios": SCENARIOS,
            "gps_noise_levels": GPS_NOISE_LEVELS,
            "detection_decay_levels": DETECTION_DECAY_LEVELS,
            "actuation_delay_levels": ACTUATION_DELAY_LEVELS,
        }
    }

    out_path = OUTPUT_DIR / "sensor_noise_robustness.json"
    out_path.write_text(json.dumps(results, indent=2))

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
