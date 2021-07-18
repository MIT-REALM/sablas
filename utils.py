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


def angle_to_sin_cos(state, angle_dims=[]):
    """ convert the angles in state vector to sin and cos
    args:
        state (n_state,)
        angle_dims: list of dimensions that are angles
    returns:
        state_converted (n_state + len(angle_dims),)
    """
    state_converted = []
    for i in range(len(state)):
        if i in angle_dims:
            state_converted.append(np.cos(state[i]))
            state_converted.append(np.sin(state[i]))
        else:
            state_converted.append(state[i])
    state_converted = np.array(state_converted)
    return state_converted


def angle_to_sin_cos_torch(state, angle_dims=[]):
    """ convert the angles in state vector to sin and cos
    args:
        state (bs, n_state, ...)
        angle_dims: list of dimensions that are angles
    returns:
        state_converted (bs, n_state + len(angle_dims), ...)
    """
    state_converted = []
    for i in range(state.shape[1]):
        if i in angle_dims:
            state_converted.append(torch.cos(state[:, i:i+1]))
            state_converted.append(torch.sin(state[:, i:i+1]))
        else:
            state_converted.append(state[:, i:i+1])
    state_converted = torch.cat(state_converted, dim=1)
    return state_converted