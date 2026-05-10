# Residual Coordination for Maritime Multi-UAV Search and Tracking Under Sparse Communication

**Author:** Satish Kumar Mishra, Netaji Subhas University of Technology, Delhi

## Contents

```
paper/
  report_final.tex        — final IEEE-format paper
  fig_visualization.pdf   — Figure 1: trajectory visualization
  fig_training_curve.pdf  — Figure 2: ES training convergence

code/
  train_residual_mlp.py         — train fixed residual policy (50 ES iterations)
  train_gated_residual.py       — train binary-gated residual policy
  train_soft_gated_residual.py  — train soft-gated residual policy
  compare_all_methods.py        — 30-seed evaluation of all 5 methods
  run_ablation_study.py         — residual scale ablation (alpha in {0,0.1,0.25,0.4})
  evaluation_utils.py           — shared evaluation utilities
  experiment_config.py          — seeds, scenarios, hyperparameters
  requirements.txt              — numpy only
  src/swarm_sim/                — simulator source code

outputs/
  residual_mlp_theta.npy        — trained fixed residual weights
  gated_residual_mlp_theta.npy  — trained binary gate weights
  soft_gated_mlp_theta.npy      — trained soft gate weights
  pure_mlp_theta.npy            — pure MLP baseline weights
```

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

Re-train from scratch (50 iterations, ~12 min each):
```bash
python3 train_residual_mlp.py
python3 train_gated_residual.py
python3 train_soft_gated_residual.py
```

Re-run ablation:
```bash
python3 run_ablation_study.py
```
