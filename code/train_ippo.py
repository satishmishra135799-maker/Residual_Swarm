"""
IPPO with analytic backprop (no torch needed).
Parameter-sharing: all UAVs share one actor-critic MLP.
Runs residual on top of hybrid base controller.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from experiment_config import FROZEN_RESIDUAL_SCALE, PAPER_BASE_POLICY, SCENARIOS, TRAIN_SEEDS
from swarm_sim.env_interface import SwarmLearningEnv
from swarm_sim.scenarios import scenario_config

# ── Hyperparameters ────────────────────────────────────────────────────────────
OBS_DIM   = 25
ACT_DIM   = 2
HIDDEN    = 16
LR        = 3e-4
GAMMA     = 0.99
LAM       = 0.95
CLIP_EPS  = 0.2
ENT_COEF  = 0.01
VF_COEF   = 0.5
EPOCHS    = 4
MINIBATCH = 128
UPDATES   = 300
SCENARIO_WEIGHTS = {"default": 0.8, "medium": 1.0, "hard": 1.35}
OUTPUT_DIR = ROOT / "outputs"

# ── Network weights stored as dict ────────────────────────────────────────────
def init_params(seed=42):
    rng = np.random.default_rng(seed)
    def w(r, c): return rng.standard_normal((r, c)) * np.sqrt(2.0 / c)
    return {
        "W1a": w(HIDDEN, OBS_DIM), "b1a": np.zeros(HIDDEN),
        "W2a": w(ACT_DIM, HIDDEN), "b2a": np.zeros(ACT_DIM),
        "W1v": w(HIDDEN, OBS_DIM), "b1v": np.zeros(HIDDEN),
        "W2v": w(1, HIDDEN),       "b2v": np.zeros(1),
        "log_std": np.full(ACT_DIM, -1.0),
    }

def actor(p, obs):
    """obs: (N, OBS_DIM) -> mean (N, ACT_DIM)"""
    h = np.tanh(obs @ p["W1a"].T + p["b1a"])
    return np.tanh(h @ p["W2a"].T + p["b2a"])

def critic(p, obs):
    """obs: (N, OBS_DIM) -> value (N,)"""
    h = np.tanh(obs @ p["W1v"].T + p["b1v"])
    return (h @ p["W2v"].T + p["b2v"]).squeeze(-1)

def log_prob(p, obs, act):
    """Gaussian log prob. obs: (N, OBS_DIM), act: (N, ACT_DIM) -> (N,)"""
    mean = actor(p, obs)
    std  = np.exp(p["log_std"])
    return -0.5 * np.sum(((act - mean) / (std + 1e-8))**2, axis=1) \
           - np.sum(p["log_std"]) - 0.5 * ACT_DIM * np.log(2 * np.pi)

def entropy(p):
    return float(np.sum(p["log_std"]) + 0.5 * ACT_DIM * (1 + np.log(2 * np.pi)))

# ── Analytic gradients ─────────────────────────────────────────────────────────
def grad_actor(p, obs, act, adv_clip):
    """Policy gradient for clipped PPO (approximate: ignores clip boundary)."""
    N = obs.shape[0]
    mean = actor(p, obs)
    std  = np.exp(p["log_std"])
    # d log_pi / d mean = (act - mean) / std^2
    d_mean = (act - mean) / (std**2 + 1e-8)   # (N, ACT_DIM)
    d_mean *= adv_clip[:, None]                 # weight by advantage

    # backprop through tanh(W2a @ h + b2a)
    h1 = np.tanh(obs @ p["W1a"].T + p["b1a"])  # (N, HIDDEN)
    pre2 = h1 @ p["W2a"].T + p["b2a"]          # (N, ACT_DIM)
    dtanh2 = (1 - np.tanh(pre2)**2)             # (N, ACT_DIM)
    delta2 = d_mean * dtanh2                    # (N, ACT_DIM)

    dW2a = delta2.T @ h1 / N                    # (ACT_DIM, HIDDEN)
    db2a = delta2.mean(0)

    # backprop through tanh(W1a @ obs + b1a)
    delta_h = delta2 @ p["W2a"]                 # (N, HIDDEN)
    dtanh1  = (1 - h1**2)
    delta1  = delta_h * dtanh1                  # (N, HIDDEN)
    dW1a = delta1.T @ obs / N                   # (HIDDEN, OBS_DIM)
    db1a = delta1.mean(0)

    # entropy gradient on log_std
    d_log_std = np.ones(ACT_DIM) * ENT_COEF     # encourage exploration

    return {"W1a": -dW1a, "b1a": -db1a, "W2a": -dW2a, "b2a": -db2a,
            "log_std": -d_log_std}

def grad_critic(p, obs, returns):
    """MSE value loss gradient."""
    N = obs.shape[0]
    h1   = np.tanh(obs @ p["W1v"].T + p["b1v"])  # (N, HIDDEN)
    vpred = (h1 @ p["W2v"].T + p["b2v"]).squeeze(-1)  # (N,)
    err   = vpred - returns                        # (N,)

    delta2 = err[:, None] * np.ones((N, 1))       # (N, 1)
    dW2v   = delta2.T @ h1 / N * VF_COEF * 2
    db2v   = delta2.mean(0) * VF_COEF * 2

    delta_h = delta2 @ p["W2v"]                   # (N, HIDDEN)
    dtanh1  = (1 - h1**2)
    delta1  = delta_h * dtanh1
    dW1v    = delta1.T @ obs / N * VF_COEF * 2
    db1v    = delta1.mean(0) * VF_COEF * 2

    return {"W1v": dW1v, "b1v": db1v, "W2v": dW2v, "b2v": db2v}

# ── Adam optimizer ─────────────────────────────────────────────────────────────
class Adam:
    def __init__(self, params, lr=LR, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k in grads:
            if k not in params: continue
            g = np.clip(grads[k], -0.5, 0.5)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g**2
            mh = self.m[k] / (1 - self.b1**self.t)
            vh = self.v[k] / (1 - self.b2**self.t)
            params[k] = params[k] - self.lr * mh / (np.sqrt(vh) + self.eps)
        return params

# ── GAE ────────────────────────────────────────────────────────────────────────
def gae(rewards, values, dones, last_v):
    n = len(rewards)
    adv = np.zeros(n)
    g = 0.0
    for t in reversed(range(n)):
        nv = last_v if t == n-1 else values[t+1]
        delta = rewards[t] + GAMMA * nv * (1 - dones[t]) - values[t]
        g = delta + GAMMA * LAM * (1 - dones[t]) * g
        adv[t] = g
    return adv, adv + values

# ── Rollout ────────────────────────────────────────────────────────────────────
def rollout(p, seeds=TRAIN_SEEDS[:3]):
    obs_buf, act_buf, lp_buf, adv_buf, ret_buf = [], [], [], [], []
    for sc in SCENARIOS:
        w = SCENARIO_WEIGHTS[sc]
        for seed in seeds:
            base = scenario_config(sc, PAPER_BASE_POLICY)
            cfg  = type(base)(**{**base.__dict__, "seed": seed})
            env  = SwarmLearningEnv(cfg, control_mode="residual",
                                    residual_scale=FROZEN_RESIDUAL_SCALE)
            obs_list = env.reset()
            ep_obs, ep_act, ep_lp, ep_rew, ep_val, ep_done = [], [], [], [], [], []
            done = False
            while not done:
                O = np.array([o.vector for o in obs_list], dtype=np.float32)
                mean = actor(p, O)
                std  = np.exp(p["log_std"])
                noise = np.random.randn(*mean.shape)
                acts  = np.clip(mean + std * noise, -1.0, 1.0)
                lps   = -0.5 * np.sum(((acts - mean)/(std+1e-8))**2, axis=1) \
                        - np.sum(p["log_std"]) - 0.5*ACT_DIM*np.log(2*np.pi)
                vals  = critic(p, O)
                obs_list, rewards, done, info = env.step(acts)
                r = float(np.mean(rewards)) * w
                ep_obs.append(O); ep_act.append(acts); ep_lp.append(lps)
                ep_rew.append(np.full(cfg.num_uavs, r))
                ep_val.append(vals); ep_done.append(np.full(cfg.num_uavs, float(done)))

            last_O  = np.array([o.vector for o in obs_list], dtype=np.float32)
            last_v  = float(critic(p, last_O).mean())
            rew_flat = np.concatenate(ep_rew)
            val_flat = np.concatenate(ep_val)
            don_flat = np.concatenate(ep_done)
            advs, rets = gae(rew_flat, val_flat, don_flat, last_v)
            obs_buf.append(np.concatenate(ep_obs))
            act_buf.append(np.concatenate(ep_act))
            lp_buf.append(np.concatenate(ep_lp).repeat(cfg.num_uavs) if np.concatenate(ep_lp).ndim==1
                          else np.concatenate(ep_lp).flatten())
            adv_buf.append(advs); ret_buf.append(rets)

    return (np.concatenate(obs_buf), np.concatenate(act_buf),
            np.concatenate(lp_buf),  np.concatenate(adv_buf),
            np.concatenate(ret_buf))

# ── PPO update ─────────────────────────────────────────────────────────────────
def update(p, opt, obs, acts, old_lp, advs, rets):
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    N = len(obs)
    for _ in range(EPOCHS):
        idx = np.random.permutation(N)
        for s in range(0, N, MINIBATCH):
            b = idx[s:s+MINIBATCH]
            if len(b) < 8: continue
            O, A, OLP, ADV, RET = obs[b], acts[b], old_lp[b], advs[b], rets[b]

            # importance ratio
            new_lp = log_prob(p, O, A)
            ratio  = np.exp(np.clip(new_lp - OLP, -5, 5))
            clip_r = np.clip(ratio, 1-CLIP_EPS, 1+CLIP_EPS)
            adv_clip = np.where(
                (ADV >= 0) & (ratio > 1+CLIP_EPS) | (ADV < 0) & (ratio < 1-CLIP_EPS),
                0.0, ADV
            )

            ga = grad_actor(p, O, A, adv_clip)
            gc = grad_critic(p, O, RET)
            grads = {k: ga.get(k, 0) + gc.get(k, 0) for k in set(ga)|set(gc)}
            p = opt.step(p, grads)
    return p

# ── Evaluate ───────────────────────────────────────────────────────────────────
def evaluate(p):
    scores = []
    for sc in SCENARIOS:
        for seed in TRAIN_SEEDS:
            base = scenario_config(sc, PAPER_BASE_POLICY)
            cfg  = type(base)(**{**base.__dict__, "seed": seed})
            env  = SwarmLearningEnv(cfg, control_mode="residual",
                                    residual_scale=FROZEN_RESIDUAL_SCALE)
            obs_list = env.reset(); done = False
            while not done:
                O = np.array([o.vector for o in obs_list], dtype=np.float32)
                acts = actor(p, O)   # deterministic
                obs_list, _, done, info = env.step(acts)
            s = info["summary_so_far"]
            scores.append(SCENARIO_WEIGHTS[sc] * (
                70*s.get("tracking_ratio",0) + 45*s.get("detection_rate",0)
                + 20*s.get("coverage_mean",0)))
    return float(np.mean(scores))

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    np.random.seed(42)
    p   = init_params()
    opt = Adam(p)
    history = []
    print(f"IPPO | updates={UPDATES} | obs={OBS_DIM} act={ACT_DIM} hidden={HIDDEN}")

    for upd in range(1, UPDATES+1):
        obs, acts, old_lp, advs, rets = rollout(p)
        p = update(p, opt, obs, acts, old_lp, advs, rets)

        if upd % 20 == 0 or upd == 1:
            score = evaluate(p)
            history.append({"update": upd, "score": score})
            print(f"Update {upd:3d}/{UPDATES}  score={score:.2f}  "
                  f"transitions={len(obs)}")

    # flatten params to numpy array for compatibility with eval scripts
    keys = ["W1a","b1a","W2a","b2a","W1v","b1v","W2v","b2v","log_std"]
    flat = np.concatenate([p[k].flatten() for k in keys])
    np.save(OUTPUT_DIR / "ippo_theta.npy", flat)
    # also save dict
    np.save(OUTPUT_DIR / "ippo_params.npy", p, allow_pickle=True)
    with open(OUTPUT_DIR / "ippo_training.json", "w") as f:
        json.dump({"history": history,
                   "hyperparams": {"lr": LR, "clip": CLIP_EPS,
                                   "epochs": EPOCHS, "updates": UPDATES}}, f, indent=2)
    print(f"\nSaved → {OUTPUT_DIR}/ippo_theta.npy")
    print(f"Final score: {evaluate(p):.2f}")

if __name__ == "__main__":
    main()
