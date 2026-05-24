"""
Fair QMIX baseline — matching IPPO's advantages:
- 25 discrete actions (8 directions × 3 speeds + stay) for finer control
- Collects episodes across ALL scenarios × ALL seeds per update (like IPPO)
- Larger replay buffer (50K transitions)
- Multiple gradient steps per collection round (4, matching IPPO's epochs)
- Gradient clipping at 0.5 (same as IPPO)
- Same total updates (500)
- Same observation space (25-dim)
- Same control mode (direct, no base controller)
"""
from __future__ import annotations
import json, sys, time, dataclasses
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from experiment_config import SCENARIOS, TRAIN_SEEDS, PAPER_BASE_POLICY
from swarm_sim.env_interface import SwarmLearningEnv
from swarm_sim.scenarios import scenario_config

# ── Discrete action space: 25 actions ────────────────────────────────────────
# 8 directions × 3 speed levels (0.33, 0.67, 1.0) + stay = 25 actions
DIRECTIONS = np.array([
    [1.0,    0.0   ],  # E
    [-1.0,   0.0   ],  # W
    [0.0,    1.0   ],  # N
    [0.0,   -1.0   ],  # S
    [0.707,  0.707 ],  # NE
    [-0.707, 0.707 ],  # NW
    [0.707, -0.707 ],  # SE
    [-0.707,-0.707 ],  # SW
], dtype=np.float32)

SPEEDS = [0.33, 0.67, 1.0]
ACTIONS = np.zeros((25, 2), dtype=np.float32)
ACTIONS[0] = [0.0, 0.0]  # stay
idx = 1
for d in DIRECTIONS:
    for s in SPEEDS:
        ACTIONS[idx] = d * s
        idx += 1

N_ACTIONS = len(ACTIONS)  # 25
assert N_ACTIONS == 25

OBS_DIM   = 25
HIDDEN    = 64
GAMMA     = 0.99
BATCH     = 64
BUF_SIZE  = 50000
TARGET_UPDATE = 10
EPS_START = 1.0
EPS_END   = 0.05
EPS_DECAY = 300
GRAD_STEPS_PER_UPDATE = 4
MAX_GRAD  = 0.5
UPDATES   = 500
SCENARIO_WEIGHTS = {"default": 0.8, "medium": 1.0, "hard": 1.35}
OUTPUT_DIR = ROOT / "outputs"


# ── Networks ─────────────────────────────────────────────────────────────────
class AgentQNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),  nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),  nn.ReLU(),
            nn.Linear(HIDDEN, N_ACTIONS)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, obs):
        return self.net(obs)


class MixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim, embed_dim=64):
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, n_agents * embed_dim)
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        B = agent_qs.size(0)
        w1 = torch.abs(self.hyper_w1(states)).view(B, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(B, 1, self.embed_dim)
        hidden = torch.relu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        w2 = torch.abs(self.hyper_w2(states)).view(B, self.embed_dim, 1)
        b2 = self.hyper_b2(states).view(B, 1, 1)
        q_total = torch.bmm(hidden, w2) + b2
        return q_total.squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, actions, reward, next_obs, done, state, next_state):
        self.buf.append((obs, actions, reward, next_obs, done, state, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buf, min(batch_size, len(self.buf)))
        obs, actions, rewards, next_obs, done, state, next_state = zip(*batch)
        return (np.array(obs), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_obs), np.array(done, dtype=np.float32),
                np.array(state), np.array(next_state))

    def __len__(self):
        return len(self.buf)


# ── Helpers ──────────────────────────────────────────────────────────────────
def get_obs_array(obs_list):
    return np.array([o.vector for o in obs_list], dtype=np.float32)

def get_state(obs_array):
    return obs_array.flatten()

def discrete_to_continuous(action_indices, config):
    return ACTIONS[action_indices] * config.max_speed


# ── Training ─────────────────────────────────────────────────────────────────
def train_qmix_fair(lr: float, n_updates: int = UPDATES):
    """
    Train QMIX fairly:
    - Each update collects episodes across ALL scenarios × ALL seeds (like IPPO)
    - Per-scenario agent networks (since num_agents differs)
    - Shared replay buffer per scenario
    """
    # Since QMIX needs fixed n_agents, we train per-scenario but collect all seeds
    # This matches IPPO which also iterates over all scenarios × seeds per update
    
    # We'll use max agents (6, from hard scenario) and track per-scenario
    scenario_configs = {}
    for sc in SCENARIOS:
        cfg = scenario_config(sc, "direct")
        scenario_configs[sc] = cfg

    # Separate networks per scenario (different n_agents)
    nets = {}
    for sc in SCENARIOS:
        cfg = scenario_configs[sc]
        n_agents = cfg.num_uavs
        state_dim = n_agents * OBS_DIM
        q_nets = [AgentQNet() for _ in range(n_agents)]
        tgt_nets = [AgentQNet() for _ in range(n_agents)]
        mixer = MixingNetwork(n_agents, state_dim)
        tgt_mixer = MixingNetwork(n_agents, state_dim)
        for i in range(n_agents):
            tgt_nets[i].load_state_dict(q_nets[i].state_dict())
        tgt_mixer.load_state_dict(mixer.state_dict())
        
        params = list(mixer.parameters())
        for net in q_nets:
            params += list(net.parameters())
        optimizer = optim.Adam(params, lr=lr, eps=1e-5)
        
        nets[sc] = {
            "q_nets": q_nets, "tgt_nets": tgt_nets,
            "mixer": mixer, "tgt_mixer": tgt_mixer,
            "optimizer": optimizer, "params": params,
            "buffer": ReplayBuffer(BUF_SIZE),
            "n_agents": n_agents, "state_dim": state_dim,
        }

    total_params = sum(
        sum(p.numel() for p in nets[sc]["params"])
        for sc in SCENARIOS
    )
    print(f"\n{'='*60}")
    print(f"Fair QMIX | lr={lr} | total_params={total_params}")
    print(f"Actions={N_ACTIONS} | Buffer={BUF_SIZE} | GradSteps={GRAD_STEPS_PER_UPDATE}")
    print(f"Updates={n_updates} | Episodes/update={len(SCENARIOS)*len(TRAIN_SEEDS)}")
    print(f"{'='*60}")

    update_count = 0
    history = []
    t0 = time.time()

    for upd in range(1, n_updates + 1):
        # ── Collect episodes across ALL scenarios × ALL seeds (like IPPO) ──
        for sc in SCENARIOS:
            cfg_base = scenario_configs[sc]
            n_agents = nets[sc]["n_agents"]
            q_nets = nets[sc]["q_nets"]
            buf = nets[sc]["buffer"]
            w = SCENARIO_WEIGHTS[sc]

            for seed in TRAIN_SEEDS:
                cfg_s = dataclasses.replace(cfg_base, seed=seed)
                env = SwarmLearningEnv(cfg_s, control_mode="direct")
                obs_list = env.reset()
                obs = get_obs_array(obs_list)
                state = get_state(obs)

                while True:
                    eps = max(EPS_END, EPS_START - update_count * (EPS_START - EPS_END) / EPS_DECAY)
                    actions = []
                    for i in range(n_agents):
                        if random.random() < eps:
                            actions.append(random.randint(0, N_ACTIONS - 1))
                        else:
                            with torch.no_grad():
                                q = q_nets[i](torch.tensor(obs[i]).unsqueeze(0))
                            actions.append(q.argmax().item())
                    actions = np.array(actions)
                    vels = discrete_to_continuous(actions, cfg_s)
                    next_obs_list, rewards, done, _ = env.step(vels)
                    next_obs = get_obs_array(next_obs_list)
                    next_state = get_state(next_obs)
                    r = float(np.mean(rewards)) * w
                    buf.push(obs, actions, r, next_obs, done, state, next_state)
                    obs, state = next_obs, next_state
                    if done:
                        break

        # ── Multiple gradient steps (like IPPO's 4 epochs) ──
        for sc in SCENARIOS:
            buf = nets[sc]["buffer"]
            if len(buf) < BATCH:
                continue
            
            q_nets = nets[sc]["q_nets"]
            tgt_nets = nets[sc]["tgt_nets"]
            mixer = nets[sc]["mixer"]
            tgt_mixer = nets[sc]["tgt_mixer"]
            optimizer = nets[sc]["optimizer"]
            params = nets[sc]["params"]
            n_agents = nets[sc]["n_agents"]

            for _ in range(GRAD_STEPS_PER_UPDATE):
                obs_b, act_b, rew_b, nobs_b, done_b, s_b, ns_b = buf.sample(BATCH)
                obs_t = torch.tensor(obs_b, dtype=torch.float32)
                nobs_t = torch.tensor(nobs_b, dtype=torch.float32)
                act_t = torch.tensor(act_b, dtype=torch.long)
                rew_t = torch.tensor(rew_b, dtype=torch.float32)
                done_t = torch.tensor(done_b, dtype=torch.float32)
                s_t = torch.tensor(s_b, dtype=torch.float32)
                ns_t = torch.tensor(ns_b, dtype=torch.float32)

                agent_qs = []
                for i in range(n_agents):
                    q = q_nets[i](obs_t[:, i, :])
                    q_a = q.gather(1, act_t[:, i:i+1])
                    agent_qs.append(q_a)
                agent_qs = torch.cat(agent_qs, dim=1)
                q_total = mixer(agent_qs, s_t)

                with torch.no_grad():
                    tgt_agent_qs = []
                    for i in range(n_agents):
                        tq = tgt_nets[i](nobs_t[:, i, :]).max(1, keepdim=True)[0]
                        tgt_agent_qs.append(tq)
                    tgt_agent_qs = torch.cat(tgt_agent_qs, dim=1)
                    q_total_tgt = tgt_mixer(tgt_agent_qs, ns_t)
                    y = rew_t.unsqueeze(1) + GAMMA * q_total_tgt * (1 - done_t.unsqueeze(1))

                loss = nn.functional.mse_loss(q_total, y)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, MAX_GRAD)
                optimizer.step()

        update_count += 1

        # Target network update
        if update_count % TARGET_UPDATE == 0:
            for sc in SCENARIOS:
                for i in range(nets[sc]["n_agents"]):
                    nets[sc]["tgt_nets"][i].load_state_dict(nets[sc]["q_nets"][i].state_dict())
                nets[sc]["tgt_mixer"].load_state_dict(nets[sc]["mixer"].state_dict())

        # ── Evaluate ──
        if upd % 50 == 0 or upd == 1:
            res = {}
            for sc in SCENARIOS:
                res[sc] = evaluate_qmix_fair(nets[sc]["q_nets"], sc)
            elapsed = time.time() - t0
            history.append({"update": upd, "elapsed": round(elapsed), "scenarios": res})
            print(f"  upd={upd:3d}/{n_updates} ({elapsed/60:.1f}min) | "
                  f"def={res['default']['tracking_ratio']:.3f} "
                  f"med={res['medium']['tracking_ratio']:.3f} "
                  f"hard={res['hard']['tracking_ratio']:.3f} | "
                  f"buf={len(nets['hard']['buffer'])}")

    # Final evaluation
    final = {}
    for sc in SCENARIOS:
        final[sc] = evaluate_qmix_fair(nets[sc]["q_nets"], sc)

    return nets, final, history


def evaluate_qmix_fair(q_nets, scenario_name):
    cfg = scenario_config(scenario_name, "direct")
    n_agents = cfg.num_uavs
    tracking_ratios = []
    detection_rates = []
    coverages = []

    for seed in TRAIN_SEEDS:
        cfg_s = dataclasses.replace(cfg, seed=seed)
        env = SwarmLearningEnv(cfg_s, control_mode="direct")
        obs_list = env.reset()
        obs = get_obs_array(obs_list)
        while True:
            actions = []
            for i in range(n_agents):
                with torch.no_grad():
                    q = q_nets[i](torch.tensor(obs[i]).unsqueeze(0))
                actions.append(q.argmax().item())
            vels = discrete_to_continuous(np.array(actions), cfg_s)
            obs_list, _, done, info = env.step(vels)
            obs = get_obs_array(obs_list)
            if done:
                break
        s = info.get('summary_so_far', {})
        tracking_ratios.append(s.get('tracking_ratio', 0))
        detection_rates.append(s.get('detection_rate', 0))
        coverages.append(s.get('coverage_mean', 0))

    return {
        'tracking_ratio': float(np.mean(tracking_ratios)),
        'tracking_std': float(np.std(tracking_ratios)),
        'detection_rate': float(np.mean(detection_rates)),
        'coverage_mean': float(np.mean(coverages)),
    }


# ── LR sensitivity run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    all_results = {}

    for lr in [3e-4, 1e-4, 3e-5]:
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        nets, final, history = train_qmix_fair(lr=lr)
        all_results[str(lr)] = {
            "final": {sc: final[sc] for sc in SCENARIOS},
            "history": history,
        }
        print(f"\n  LR={lr} FINAL: "
              f"def={final['default']['tracking_ratio']:.4f} "
              f"med={final['medium']['tracking_ratio']:.4f} "
              f"hard={final['hard']['tracking_ratio']:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("=== Fair QMIX LR Sensitivity (25 actions, 50K buffer, multi-scenario) ===")
    print(f"{'='*60}")
    for lr, data in all_results.items():
        f = data["final"]
        print(f"  lr={lr}: default={f['default']['tracking_ratio']:.4f} "
              f"medium={f['medium']['tracking_ratio']:.4f} "
              f"hard={f['hard']['tracking_ratio']:.4f}")

    # Compare with IPPO
    print(f"\nFor reference — IPPO best (from paper): "
          f"default=0.553  medium=0.589  hard=0.505")
    print(f"Hybrid base controller:                  "
          f"default=0.869  medium=0.879  hard=0.905")
    print(f"ES residual (450 params):                "
          f"default=0.877  medium=0.913  hard=0.912")

    out = OUTPUT_DIR / "qmix_fair_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved -> {out}")
