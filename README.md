# Residual Coordination for Maritime Multi-UAV Search and Tracking Under Sparse Communication

**Author:** Satish Kumar Mishra, Netaji Subhas University of Technology, Delhi
**Target:** IEEE Robotics and Automation Letters (RA-L)

## Contents

```
paper/
  main.tex                      -- IEEE RA-L submission manuscript
  figures/                      -- all paper figures (method overview, IPPO comparison, etc.)
  fig_visualization.pdf         -- trajectory visualization
  fig_training_curve.pdf        -- ES training convergence

code/
  # Core training
  train_residual_mlp.py         -- train fixed residual policy (ES, 100 iterations)
  train_gated_residual.py       -- train binary-gated residual policy
  train_soft_gated_residual.py  -- train soft-gated residual policy
  train_ippo.py                 -- Independent PPO baseline (20,165 params)
  train_qmix_fair.py            -- QMIX baseline (346K params, fair setup)

  # Evaluation
  compare_all_methods.py        -- 30-seed evaluation of all methods
  run_ablation_study.py         -- residual scale ablation (alpha in {0,0.1,0.25,0.4})
  run_ippo_lr_sensitivity.py    -- IPPO learning rate sweep (3e-4, 1e-4, 3e-5)
  evaluate_packet_drop.py       -- robustness under 0-60% packet drop
  evaluate_sensor_noise.py      -- robustness under GPS noise, detection decay, actuation delay
  multi_seed_evaluation.py      -- full 30-seed benchmark
  compute_pvalues.py            -- paired t-tests for statistical significance

  # Utilities
  evaluation_utils.py           -- shared evaluation utilities
  experiment_config.py          -- seeds, scenarios, hyperparameters
  requirements.txt              -- numpy, torch (for IPPO/QMIX only)

  # Simulator
  src/swarm_sim/
    simulator.py                -- MaritimeSwarmSimulator (sensor noise, actuation delay)
    env_interface.py            -- SwarmLearningEnv (residual control modes, GPS noise)
    config.py                   -- SimulationConfig dataclass
    scenarios.py                -- scenario configurations (default, medium, hard)
    mlp_policy.py               -- 25-16-2 MLP forward pass (450 params)
    policies.py                 -- 4 handcrafted policies
    belief.py                   -- 3-stage belief update
    coverage.py                 -- coverage grid decay
    render.py                   -- visualization

outputs/
  # Trained weights
  residual_mlp_theta.npy        -- trained fixed residual weights (100 iter)
  gated_residual_mlp_theta.npy  -- trained binary gate weights
  soft_gated_mlp_theta.npy      -- trained soft gate weights
  pure_mlp_theta.npy            -- pure MLP baseline weights
  ippo_params.npy               -- trained IPPO weights

  # Evaluation results
  multi_seed_summary.json       -- 30-seed main results
  pvalue_analysis.json          -- statistical significance tests
  ippo_lr_sensitivity.json      -- IPPO learning rate sweep results
  qmix_fair_results.json        -- QMIX fair evaluation results
  packet_drop_robustness.json   -- robustness under packet drop
  sensor_noise_robustness.json  -- robustness under sensor/actuation noise
```

## Key Results

| Method | Params | Default | Medium | Hard |
|--------|--------|---------|--------|------|
| QMIX (best LR) | 345,978 | 0.000 | 0.013 | 0.025 |
| IPPO (best LR) | 20,165 | 0.553 | 0.589 | 0.505 |
| Hybrid base | 0 | 0.869 | 0.879 | 0.905 |
| Fixed residual (ours) | 450 | 0.877 | 0.904 | **0.912** |
| Soft gate (ours) | 450 | **0.881** | **0.913** | 0.910 |

Our 450-parameter residual achieves 1.65x higher tracking than 20,165-param IPPO and 770x fewer params than QMIX (which fails completely).

## Reproduce Results

Install dependencies:
```bash
pip install -r code/requirements.txt
```

Re-run 30-seed evaluation (uses pre-trained weights):
```bash
cd code
python3 compare_all_methods.py
```

Re-train from scratch (~12 min each):
```bash
python3 train_residual_mlp.py
python3 train_gated_residual.py
python3 train_soft_gated_residual.py
```

Run MARL baselines:
```bash
python3 train_ippo.py
python3 run_ippo_lr_sensitivity.py
python3 train_qmix_fair.py
```

Run robustness studies:
```bash
python3 evaluate_packet_drop.py
python3 evaluate_sensor_noise.py
```

Run ablation and statistics:
```bash
python3 run_ablation_study.py
python3 compute_pvalues.py
```
