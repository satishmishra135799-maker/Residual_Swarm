from __future__ import annotations

import numpy as np


OBS_DIM = 25
HIDDEN_DIM = 16
ACTION_DIM = 2
TAU_DEFAULT = 0.08  # fixed tau for non-learnable variants


def theta_dim(learnable_tau: bool = False) -> int:
    base = HIDDEN_DIM * OBS_DIM + HIDDEN_DIM + ACTION_DIM * HIDDEN_DIM + ACTION_DIM
    return base + 1 if learnable_tau else base


def unpack(theta: np.ndarray, learnable_tau: bool = False) -> tuple:
    idx = 0
    w1_size = HIDDEN_DIM * OBS_DIM
    w1 = theta[idx : idx + w1_size].reshape(HIDDEN_DIM, OBS_DIM)
    idx += w1_size
    b1 = theta[idx : idx + HIDDEN_DIM]
    idx += HIDDEN_DIM
    w2_size = ACTION_DIM * HIDDEN_DIM
    w2 = theta[idx : idx + w2_size].reshape(ACTION_DIM, HIDDEN_DIM)
    idx += w2_size
    b2 = theta[idx : idx + ACTION_DIM]
    if learnable_tau:
        # tau stored as raw value, mapped to (0,1) via sigmoid
        tau_raw = float(theta[idx])
        tau = float(1.0 / (1.0 + np.exp(-tau_raw)))
        return w1, b1, w2, b2, tau
    return w1, b1, w2, b2


def policy_action(theta: np.ndarray, observations: list,
                  learnable_tau: bool = False) -> np.ndarray:
    if learnable_tau:
        w1, b1, w2, b2, _ = unpack(theta, learnable_tau=True)
    else:
        w1, b1, w2, b2 = unpack(theta)
    actions = []
    for obs in observations:
        x = np.asarray(obs.vector, dtype=float)
        hidden = np.tanh(w1 @ x + b1)
        logits = w2 @ hidden + b2
        actions.append(np.tanh(logits))
    return np.asarray(actions, dtype=float)


def extract_tau(theta: np.ndarray) -> float:
    """Extract learned tau from theta (last element, sigmoid-mapped to (0,1))."""
    tau_raw = float(theta[-1])
    return float(1.0 / (1.0 + np.exp(-tau_raw)))

