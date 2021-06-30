import numpy as np
import torch


class DoubleIntegrator(object):

    def __init__(self, dt=0.1, k_obstacle=8, total_obstacle=10, env_size=20, safe_dist=1, max_steps=300, max_speed=0.5):
        assert total_obstacle >= k_obstacle
        self.dt = dt
        self.k_obstacle = k_obstacle
        self.total_obstacle = total_obstacle
        self.env_size = env_size
        self.safe_dist = safe_dist
        self.max_steps = max_steps
        self.max_speed = max_speed

    def reset(self):
        self.obstacle = np.random.uniform(
            low=0, high=self.env_size, size=(self.total_obstacle, 2))
        state = np.random.uniform(low=0, high=self.env_size, size=(2,))
        while np.amin(np.linalg.norm(self.obstacle - state, axis=1)) < self.safe_dist:
            state = np.random.uniform(low=0, high=self.env_size, size=(2,))
        state = np.concatenate([state, np.zeros((2,))])
        self.state = state
        obstacle = self.get_obstacle(state)
        goal = np.random.uniform(low=0, high=self.env_size, size=(2,))
        while np.amin(np.linalg.norm(self.obstacle - goal, axis=1)) < self.safe_dist:
            goal = np.random.uniform(low=0, high=self.env_size, size=(2,))
        goal = np.concatenate([goal, np.zeros((2,))])
        self.goal = goal
        self.num_steps = 0
        return state, obstacle, goal

    def step(self, u):
        dsdt = self.uncertain_dynamics(self.state, u)
        state = self.state + dsdt * self.dt
        state[2:] = np.clip(state[2:], -self.max_speed, self.max_speed)
        obstacle = self.get_obstacle(state)
        goal = self.get_goal(state)
        self.state = state
        done = np.linalg.norm(state[:2] - goal[:2]) < self.safe_dist or self.num_steps > self.max_steps
        self.num_steps = self.num_steps + 1
        return state, obstacle, goal, done

    def uncertain_dynamics(self, state, u):
        """
        args:
            state (n_state,)
            u (m_control,)
        returns:
            dsdt (n_state,)
        """
        dsdt = np.concatenate([state[2:], u])
        return dsdt

    def nominal_dynamics_torch(self, state, u):
        """
        args:
            state (bs, n_state)
            u (bs, m_control)
        returns:
            dsdt (bs, n_state)
        """
        dsdt = torch.cat([state[:, 2:], u], dim=1)
        return dsdt

    def nominal_controller(self, state, goal, u_norm_max=0.5):
        """
        args:
            state (n_state,)
            goal (n_state,)
        returns:
            u_nominal (m_control,)
        """
        K = np.array([[1, 0, 3, 0], [0, 1, 0, 3]])
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
        dist = np.linalg.norm(self.obstacle - state[:2], axis=1)
        argsort = np.argsort(dist)[:self.k_obstacle]
        obstacle = self.obstacle[argsort]
        obstacle = np.concatenate([obstacle, np.zeros((self.k_obstacle, 2))], axis=1)
        return obstacle

    def get_goal(self, state):
        return self.goal


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    env = DoubleIntegrator()
    plt.figure(figsize=(10, 10))
    plt.xlim(0, 10)
    plt.ylim(0, 10)

    state, obstacle, goal = env.reset()
    plt.scatter(env.obstacle[:, 0], env.obstacle[:, 1], color='grey')
    plt.scatter(state[0], state[1], color='darkred')
    plt.scatter(obstacle[:, 0], obstacle[:, 1], color='darkblue')
    plt.scatter(goal[0], goal[1], color='darkorange')

    plt.show()