import numpy as np
import torch
from network import ControlAffineDynamics
import utils


class Ship(object):

    def __init__(self,
                 dt=0.1, 
                 k_obstacle=8, 
                 total_obstacle=20, 
                 env_size=20, 
                 safe_dist=1, 
                 max_steps=600, 
                 max_speed=np.array([0.3, 0.3, 1.0]),
                 gpu_id=-1,
                 estimated_param=None
                 ):
        assert total_obstacle >= k_obstacle
        self.dt = dt
        self.k_obstacle = k_obstacle
        self.total_obstacle = total_obstacle
        self.env_size = env_size
        self.safe_dist = safe_dist
        self.max_steps = max_steps
        self.max_speed = max_speed

        self.B_real = np.array([[1, 0], [0, 0.1], [0, 0.5]])
        self.B_nominal = np.array([[1, 0], [0, 0.1], [0, 0.5]])

        # if gpu_id >= 0, we use gpu in self.nominal_dynamics_torch
        self.gpu_id = gpu_id

        # prepare the estimated dynamics
        if estimated_param is not None:
            preprocess_func = lambda x: utils.angle_to_sin_cos_torch(x, [2])
            self.dynamics_mlp = ControlAffineDynamics(
                n_state=6, m_control=2, n_extended_state=1, preprocess_func=preprocess_func)
            self.dynamics_mlp.load_state_dict(torch.load(estimated_param))
            if gpu_id >= 0:
                self.dynamics_mlp = self.dynamics_mlp.cuda(gpu_id)
        else:
            self.dynamics_mlp = None

    def reset(self):
        self.obstacle = np.random.uniform(
            low=0, high=self.env_size, size=(self.total_obstacle, 2))
        state = np.random.uniform(low=0, high=self.env_size, size=(2,))
        while np.amin(np.linalg.norm(self.obstacle - state, axis=1)) < self.safe_dist:
            state = np.random.uniform(low=0, high=self.env_size, size=(2,))
        # [x, y, psi, u, v, r]
        state = np.concatenate([state, [np.random.uniform(low=0, high=np.pi)], np.zeros((3,))])
        self.state = state
        obstacle = self.get_obstacle(state)
        goal = np.random.uniform(low=0, high=self.env_size, size=(2,))
        while np.amin(np.linalg.norm(self.obstacle - goal, axis=1)) < self.safe_dist:
            goal = np.random.uniform(low=0, high=self.env_size, size=(2,))
        goal = np.concatenate([goal, np.zeros((4,))])
        self.goal = goal
        self.num_steps = 0
        return state, obstacle, goal

    def step(self, u):
        dsdt = self.uncertain_dynamics(self.state, u)
        state = self.state + dsdt * self.dt
        state[3:] = np.clip(state[3:], -self.max_speed, self.max_speed)

        dsdt_nominal = self.nominal_dynamics(self.state, u)
        state_nominal = self.state + dsdt_nominal * self.dt
        state_nominal[3:] = np.clip(state_nominal[3:], -self.max_speed, self.max_speed)

        obstacle = self.get_obstacle(state)
        goal = self.get_goal(state)
        self.state = state
        done = np.linalg.norm(state[:2] - goal[:2]) < self.safe_dist or self.num_steps > self.max_steps
        self.num_steps = self.num_steps + 1
        return state, state_nominal, obstacle, goal, done

    def uncertain_dynamics(self, state, u):
        """
        args:
            state (n_state,): [eta, nu], where eta = [x, y, psi] and nu = [u, v, r]
            u (m_control,): [T, delta], where T is the thrust and delta is the rudder angle
        returns:
            dsdt (n_state,)
        """
        psi = state[2]
        R = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi),  np.cos(psi), 0],
            [          0,            0, 1]
        ])
        detadt = R.dot(state[3:])
        # drdt = delta, v = alpha * r, dvdt = alpha * delta, alpha = 0.2
        B = self.B_real
        dnudt = B.dot(u)
        dsdt = np.concatenate([detadt, dnudt])
        return dsdt

    def nominal_dynamics(self, state, u):
        """
        args:
            state (n_state,): [eta, nu], where eta = [x, y, psi] and nu = [u, v, r]
            u (m_control,): [T, delta], where T is the thrust and delta is the rudder angle
        returns:
            dsdt (n_state,)
        """
        psi = state[2]
        R = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi),  np.cos(psi), 0],
            [          0,            0, 1]
        ])
        detadt = R.dot(state[3:])
        # drdt = delta, v = alpha * r, dvdt = alpha * delta, alpha = 0.2
        B = self.B_nominal
        dnudt = B.dot(u)
        dsdt = np.concatenate([detadt, dnudt])
        return dsdt

    def nominal_dynamics_torch(self, state, u):
        """
        args:
            state (bs, n_state)
            u (bs, m_control)
        returns:
            dsdt (bs, n_state)
        """
        if self.dynamics_mlp is not None:
            return self.nominal_dynamics_mlp_torch(state, u)
        else:
            return self.nominal_dynamics_analytic_torch(state, u)

    def nominal_dynamics_analytic_torch(self, state, u):
        """
        args:
            state (bs, n_state)
            u (bs, m_control)
        returns:
            dsdt (bs, n_state)
        """
        psi = state[:, 2:3]
        zeros = torch.zeros_like(psi)
        ones = torch.ones_like(psi)

        R = torch.cat([
            torch.cos(psi), -torch.sin(psi), zeros,
            torch.sin(psi), torch.cos(psi), zeros,
            zeros, zeros, ones
        ], dim=1)
        if self.gpu_id >= 0:
            R = R.cuda(self.gpu_id)
        detadt = torch.bmm(R.view(-1, 3, 3), state[:, 3:].view(-1, 3, 1)) # (bs, 3, 1)
        detadt = detadt.squeeze(-1)  # (bs, 3)

        bs = state.shape[0]
        B = self.B_nominal.astype(np.float32).reshape(-1, 3, 2)
        B = torch.from_numpy(B).repeat(bs, 1, 1)
        if self.gpu_id >=0:
            B = B.cuda(self.gpu_id)
        dnudt = torch.bmm(B, u.view(-1, 2, 1)) # (bs, 3, 1)
        dnudt = dnudt.squeeze(-1)  # (bs, 3)
        dsdt = torch.cat([detadt, dnudt], dim=1)
        return dsdt

    def nominal_dynamics_mlp_torch(self, state, u):
        """
        args:
            state (bs, n_state)
            u (bs, m_control)
        returns:
            dsdt (bs, n_state)
        """
        f, B = self.dynamics_mlp(state, u)
        dsdt = f + torch.bmm(B, u.unsqueeze(-1)).squeeze(-1)
        return dsdt

    def nominal_controller(self, state, goal, u_norm_max=0.5):
        """
        args:
            state (n_state,)
            goal (n_state,)
        returns:
            u_nominal (m_control,)
        """
        psi = state[2]
        r = state[5]

        vec = goal[:2] - state[:2]
        vec = vec / np.linalg.norm(vec)

        yaw_error_cos = np.cos(psi) * vec[0] + np.sin(psi) * vec[1]
        yaw_error_sin = np.sin(psi) * vec[0] - np.cos(psi) * vec[1]

        if yaw_error_cos <= 0:
            rudder = -yaw_error_sin / abs(yaw_error_sin) - r
        else:
            rudder = -yaw_error_sin - r
        thrust = yaw_error_cos + 1
        u_nominal = np.clip(np.array([thrust, rudder]), -1, 1)

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
        obstacle = np.concatenate([obstacle, np.zeros((self.k_obstacle, 4))], axis=1)
        return obstacle

    def get_goal(self, state):
        return self.goal


class River(Ship):

    def __init__(self,
                 dt=0.1, 
                 k_obstacle=8, 
                 total_obstacle=100,
                 env_size=20,
                 river_width=2.0,
                 safe_dist=1, 
                 max_steps=600, 
                 max_speed=np.array([0.3, 0.3, 1.0]),
                 gpu_id=-1,
                 estimated_param=None
                 ):
        super(River, self).__init__(
                 dt,
                 k_obstacle,
                 total_obstacle,
                 env_size,
                 safe_dist,
                 max_steps,
                 max_speed,
                 gpu_id,
                 estimated_param
        )
        self.river_width = river_width
    
    def reset(self):
        obstacle_xs_half = np.linspace(0, self.env_size, num=self.total_obstacle//2)
        obstacle_xs = np.concatenate([obstacle_xs_half, obstacle_xs_half])
        obstacle_ys_half = np.sin(obstacle_xs_half / self.env_size * np.pi * 2)
        obstacle_ys = np.concatenate([
            obstacle_ys_half * 1 + self.env_size * 0.5 - self.river_width * 0.5,
            obstacle_ys_half * 1 + self.env_size * 0.5 + self.river_width * 0.5,
        ])
        
        self.obstacle = np.concatenate([obstacle_xs.reshape(-1, 1), obstacle_ys.reshape(-1, 1)], axis=1)
        state = np.array([0, self.env_size * 0.5])
        # [x, y, psi, u, v, r]
        state = np.concatenate([state, [0], np.zeros((3,))])
        self.state = state
        obstacle = self.get_obstacle(state)
        goal = np.array([self.env_size, self.env_size * 0.5])
        goal = np.concatenate([goal, np.zeros((4,))])
        self.goal = goal
        self.num_steps = 0
        return state, obstacle, goal


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    env = Ship()
    fig = plt.figure(figsize=(10, 10))
    #plt.ion()

    plt.xlim(0, 20)
    plt.ylim(0, 20)

    state, obstacle, goal = env.reset()
    plt.scatter(env.obstacle[:, 0], env.obstacle[:, 1], color='grey')
    plt.scatter(state[0], state[1], color='darkred')
    plt.scatter(obstacle[:, 0], obstacle[:, 1], color='darkblue')
    plt.scatter(goal[0], goal[1], color='darkorange')

    for i in range(env.max_steps):
        u = env.nominal_controller(state, goal)
        state, _, _, _, _ = env.step(u)
        plt.scatter(state[0], state[1], color='darkred', alpha=i/1000)
        #fig.canvas.draw()
        #plt.pause(0.1)

        if np.linalg.norm(state[:2] - goal[:2]) < 1.0:
            break

    plt.show()