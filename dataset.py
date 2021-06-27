import numpy as np


class Dataset(object):

    def __init__(self, n_state, k_obstacle, m_control, n_pos, buffer_size=10000, safe_dist=1.0, dang_dist=0.6):
        self.n_state = n_state
        self.k_obstacle = k_obstacle
        self.m_control = m_control
        self.n_pos = n_pos
        self.safe_dist = safe_dist
        self.dang_dist = dang_dist
        self.buffer_size = buffer_size
        self.buffer_safe = []
        self.buffer_dang = []
        self.buffer_mid = []

    def add_data(self, state, obstacle, u_nominal, state_next):
        """
        args:
            state (n_state,): state of the agent
            obstacle (k_obstacle, n_state): K obstacles
            u_nominal (m_control,): the nominal control
            state_next (n_state,): state of the agent at the next timestep
        """
        dist = np.linalg.norm(obstacle[:, :self.n_pos] - state[:self.n_pos], axis=1)
        min_dist = np.amin(dist)
        data = [np.copy(state).astype(np.float32), np.copy(obstacle).astype(np.float32), 
                np.copy(u_nominal).astype(np.float32), np.copy(state_next).astype(np.float32)]
        if min_dist < self.dang_dist:
            self.buffer_dang.append(data)
            self.buffer_dang = self.buffer_dang[-self.buffer_size:]
        elif min_dist > self.safe_dist:
            self.buffer_safe.append(data)
            self.buffer_safe = self.buffer_safe[-self.buffer_size:]
        else:
            self.buffer_mid.append(data)
            self.buffer_mid = self.buffer_mid[-self.buffer_size:]

    def sample_data(self, batch_size):
        num_safe = batch_size // 3
        num_dang = batch_size // 3
        num_mid = batch_size - num_safe - num_dang

        s_safe, o_safe, u_safe, s_next_safe = self.sample_data_from_buffer(num_safe, self.buffer_safe)
        s_dang, o_dang, u_dang, s_next_dang = self.sample_data_from_buffer(num_dang, self.buffer_dang)
        s_mid, o_mid, u_mid, s_next_mid = self.sample_data_from_buffer(num_mid, self.buffer_mid)

        s = np.concatenate([s_safe, s_dang, s_mid], axis=0)
        o = np.concatenate([o_safe, o_dang, o_mid], axis=0)
        u = np.concatenate([u_safe, u_dang, u_mid], axis=0)
        s_next = np.concatenate([s_next_safe, s_next_dang, s_next_mid], axis=0)

        return s, o, u, s_next

    def sample_data_from_buffer(self, batch_size, buffer):
        indices = np.random.randint(len(buffer), size=(batch_size))
        s = np.zeros((batch_size, self.n_state), dtype=np.float32)
        o = np.zeros((batch_size, self.k_obstacle, self.n_state), dtype=np.float32)
        u = np.zeros((batch_size, self.m_control), dtype=np.float32)
        s_next = np.zeros((batch_size, self.n_state), dtype=np.float32)
        for i, ind in enumerate(indices):
            state, obstacle, u_nominal, state_next = buffer[ind]
            s[i] = state
            o[i] = obstacle
            u[i] = u_nominal
            s_next[i] = state_next
        return s, o, u, s_next