"""IPPO LR sensitivity: tests lr=3e-4, 1e-4, 3e-5 each for 500 updates."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import torch, numpy as np, torch.optim as optim
from train_ippo_proper import ActorCritic, collect_rollouts, ppo_update, evaluate

OUTPUT_DIR = ROOT / "outputs"
results = {}

for lr in [3e-4, 1e-4, 3e-5]:
    torch.manual_seed(42); np.random.seed(42)
    net = ActorCritic()
    opt = optim.Adam(net.parameters(), lr=lr, eps=1e-5)
    print(f"\n--- LR={lr} ---")
    for upd in range(1, 501):
        obs, acts, lps, advs, rets = collect_rollouts(net)
        ppo_update(net, opt, obs, acts, lps, advs, rets)
        if upd % 100 == 0:
            res = evaluate(net)
            print(f"  upd={upd:3d} | default={res['default']['tracking_ratio']:.3f} "
                  f"medium={res['medium']['tracking_ratio']:.3f} "
                  f"hard={res['hard']['tracking_ratio']:.3f}")
    final = evaluate(net)
    results[str(lr)] = final
    print(f"  FINAL lr={lr}: default={final['default']['tracking_ratio']:.4f} "
          f"medium={final['medium']['tracking_ratio']:.4f} "
          f"hard={final['hard']['tracking_ratio']:.4f}")

with open(OUTPUT_DIR / "ippo_lr_sensitivity.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved → outputs/ippo_lr_sensitivity.json")
