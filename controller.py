import numpy as np 
import torch
from torch import nn

class NominalController(object):

    def __init__(self, u_norm_max=0.5):
        self.u_norm_max = u_norm_max
        self.K = np.array([
            [1, 0, 3, 0],
            [0, 1, 0, 3]
        ])
        return

    def __call__(self, state, goal):
        u_nominal = -self.K.dot(state - goal)
        norm = np.linalg.norm(u_nominal)
        if norm > self.u_norm_max:
            u_nominal = u_nominal / norm * self.u_norm_max
        return u_nominal


class NNController(nn.Module):

    def __init__(self, n_state, k_obstacle, m_control):
        super().__init__()
        self.n_state = n_state
        self.k_obstacle = k_obstacle
        self.m_control = m_control

        self.conv0 = nn.Conv1d(n_state, 64, 1)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.fc0 = nn.Linear(128 + m_control, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, m_control)
        self.activation = nn.ReLU()
        self.output_activation = nn.Tanh()

    def forward(self, state, obstacle, u_nominal):
        """
        args:
            state (bs, n_state)
            obstacle (bs, k_obstacle, n_state)
            u_nominal (bs, m_control)
        returns:
            u (bs, m_control)
        """
        state = torch.unsqueeze(state, 2)    # (bs, n_state, 1)
        obstacle = obstacle.permute(0, 2, 1) # (bs, n_state, k_obstacle)
        
        x = self.activation(self.conv0(state - obstacle))
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))   # (bs, 128, k_obstacle)
        x, _ = torch.max(x, dim=2)              # (bs, 128)
        x = torch.cat([x, u_nominal], dim=1) # (bs, 128 + m_control)
        x = self.activation(self.fc0(x))
        x = self.activation(self.fc1(x))
        x = self.output_activation(self.fc2(x))
        u = x + u_nominal
        return u