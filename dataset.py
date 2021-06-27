import numpy as np


class Dataset(object):

    def __init__(self, n_state, k_obstacle, m_control, buffer_size=10000):
        self.n_state = n_state
        self.k_obstacle = k_obstacle
        self.m_control = m_control
        self.buffer_size = buffer_size
        self.buffer = []

    def add_data(self, state, obstacle, u_nominal, state_next):
        """
        args:
            state (n_state,): state of the agent
            obstacle (k_obstacle, n_state): K obstacles
            u_nominal (m_control,): the nominal control
            state_next (n_state,): state of the agent at the next timestep
        """
        self.buffer.append(
            [np.copy(state).astype(np.float32), 
             np.copy(obstacle).astype(np.float32), 
             np.copy(u_nominal).astype(np.float32), 
             np.copy(state_next).astype(np.float32)])
        self.buffer = self.buffer[-self.buffer_size:]

    def sample_data(self, batch_size):
        indices = np.random.randint(len(self.buffer), size=(batch_size))
        s = np.zeros((batch_size, self.n_state), dtype=np.float32)
        o = np.zeros((batch_size, self.k_obstacle, self.n_state), dtype=np.float32)
        u = np.zeros((batch_size, self.m_control), dtype=np.float32)
        s_next = np.zeros((batch_size, self.n_state), dtype=np.float32)
        for i, ind in enumerate(indices):
            state, obstacle, u_nominal, state_next = self.buffer[ind]
            s[i] = state
            o[i] = obstacle
            u[i] = u_nominal
            s_next[i] = state_next
        return s, o, u, s_next