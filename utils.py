import torch
import numpy as np


def is_safe(state, obstacle, n_pos, dang_dist=0.6):
    """
    args:
        state (n_state,)
        obstacle (k_obstacle, n_state)
        n_pos int
    """
    min_dist = np.amin(np.linalg.norm(obstacle[:, :n_pos] - state[:n_pos], axis=1))
    return min_dist > dang_dist