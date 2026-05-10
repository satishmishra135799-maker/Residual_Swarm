from __future__ import annotations

import numpy as np


OBS_DIM = 25
HIDDEN_DIM = 16
ACTION_DIM = 2


def theta_dim() -> int:
    return HIDDEN_DIM * OBS_DIM + HIDDEN_DIM + ACTION_DIM * HIDDEN_DIM + ACTION_DIM


def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return w1, b1, w2, b2


def policy_action(theta: np.ndarray, observations: list) -> np.ndarray:
    w1, b1, w2, b2 = unpack(theta)
    actions = []
    for obs in observations:
        x = np.asarray(obs.vector, dtype=float)
        hidden = np.tanh(w1 @ x + b1)
        logits = w2 @ hidden + b2
        actions.append(np.tanh(logits))
    return np.asarray(actions, dtype=float)
