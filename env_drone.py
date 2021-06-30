import numpy as np
import torch
import control


class Drone(object):

    def __init__(self, 
                 dt=0.1, 
                 k_obstacle=8, 
                 total_obstacle=500, 
                 env_size=20, 
                 safe_dist=1, 
                 max_steps=500, 
                 max_speed=0.5, 
                 max_theta=np.pi/6,
                 noise_std=0):
        assert total_obstacle >= k_obstacle
        self.dt = dt
        self.k_obstacle = k_obstacle
        self.total_obstacle = total_obstacle
        self.env_size = env_size
        self.safe_dist = safe_dist
        self.max_steps = max_steps
        self.max_speed = max_speed
        self.max_theta = max_theta
        self.noise_std = noise_std

        self.A_nominal = [[0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0]]
        self.A_real = [[0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0]]
        self.B_nominal = [[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 1],
                          [1, 0, 0],
                          [0, 1, 0]]
        self.B_real = [[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 1],
                       [1, 0, 0],
                       [0, 1, 0]]
        self.K = self.get_K()
        self.noise = np.random.normal(size=(8,)) * self.noise_std
        
        
    def reset(self):
        self.obstacle = np.random.uniform(
            low=0, high=self.env_size, size=(self.total_obstacle, 3))
        state = np.random.uniform(low=0, high=self.env_size, size=(3,))
        while np.amin(np.linalg.norm(self.obstacle - state, axis=1)) < self.safe_dist:
            state = np.random.uniform(low=0, high=self.env_size, size=(3,))
        state = np.concatenate([state, np.zeros((5,))])
        self.state = state
        obstacle = self.get_obstacle(state)
        goal = np.random.uniform(low=0, high=self.env_size, size=(3,))
        while np.amin(np.linalg.norm(self.obstacle - goal, axis=1)) < self.safe_dist:
            goal = np.random.uniform(low=0, high=self.env_size, size=(3,))
        goal = np.concatenate([goal, np.zeros((5,))])
        self.goal = goal
        self.num_steps = 0
        return state, obstacle, goal

    def step(self, u):
        dsdt = self.uncertain_dynamics(self.state, u)
        noise = self.get_noise()
        state = self.state + (dsdt + noise) * self.dt
        state[3:6] = np.clip(state[3:6], -self.max_speed, self.max_speed)
        state[6:] = np.clip(state[6:], -self.max_theta, self.max_theta)

        dsdt_nominal = self.nominal_dynamics(self.state, u)
        state_nominal = self.state + dsdt_nominal * self.dt
        state_nominal[3:6] = np.clip(state_nominal[3:6], -self.max_speed, self.max_speed)
        state_nominal[6:] = np.clip(state_nominal[6:], -self.max_theta, self.max_theta)

        obstacle = self.get_obstacle(state)
        goal = self.get_goal(state)
        self.state = state
        done = np.linalg.norm(state[:3] - goal[:3]) < self.safe_dist or self.num_steps > self.max_steps
        self.num_steps = self.num_steps + 1
        return state, state_nominal, obstacle, goal, done

    def uncertain_dynamics(self, state, u):
        """
        args:
            state (n_state,)
            u (m_control,)
        returns:
            dsdt (n_state,)
        """
        A = np.array(self.A_real, dtype=np.float32)
        B = np.array(self.B_real, dtype=np.float32)
        dsdt = A.dot(state) + B.dot(u)

        return dsdt

    def nominal_dynamics(self, state, u):
        """
        args:
            state (n_state,)
            u (m_control,)
        returns:
            dsdt (n_state,)
        """
        A = np.array(self.A_nominal, dtype=np.float32)
        B = np.array(self.B_nominal, dtype=np.float32)
        dsdt = A.dot(state) + B.dot(u)

        return dsdt

    def nominal_dynamics_torch(self, state, u):
        """
        args:
            state (bs, n_state)
            u (bs, m_control)
        returns:
            dsdt (bs, n_state)
        """
        A = np.array(self.A_nominal, dtype=np.float32)
        A_T = torch.from_numpy(A.T)

        B = np.array(self.B_nominal, dtype=np.float32)
        B_T = torch.from_numpy(B.T)

        dsdt = torch.matmul(state, A_T) + torch.matmul(u, B_T)
        
        return dsdt

    def nominal_controller(self, state, goal, u_norm_max=0.5):
        """
        args:
            state (n_state,)
            goal (n_state,)
        returns:
            u_nominal (m_control,)
        """
        K = self.K
        u_nominal = -K.dot(state - goal)
        norm = np.linalg.norm(u_nominal)
        if norm > u_norm_max:
            u_nominal = u_nominal / norm * u_norm_max
        return u_nominal

    def get_obstacle(self, state):
        """
        args:
            state (n_state,)
        returns:
            obstacle (k_obstacle, n_state)
        """
        dist = np.linalg.norm(self.obstacle - state[:3], axis=1)
        argsort = np.argsort(dist)[:self.k_obstacle]
        obstacle = self.obstacle[argsort]
        obstacle = np.concatenate([obstacle, np.zeros((self.k_obstacle, 5))], axis=1)
        return obstacle

    def get_goal(self, state):
        return self.goal

    def get_K(self, state=None):
        Q = np.eye(8)
        R = np.eye(3)
        K, _, _ = control.lqr(self.A_real, self.B_real, Q, R)
        return K

    def get_noise(self):
        if np.random.uniform() < 0.05:
            self.noise = np.random.normal(size=(8,)) * self.noise_std
        return self.noise


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    env = Drone()
    
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_zlim(0, 20)

    state, obstacle, goal = env.reset()
    ax.scatter(env.obstacle[:, 0], env.obstacle[:, 1], env.obstacle[:, 2], color='grey')
    ax.scatter(state[0], state[1], state[2], color='darkred')
    ax.scatter(obstacle[:, 0], obstacle[:, 1], obstacle[:, 2], color='darkblue')
    ax.scatter(goal[0], goal[1], goal[2], color='darkorange')

    plt.show()