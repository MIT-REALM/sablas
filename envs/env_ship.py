import os
import sys
sys.path.insert(1, os.path.abspath('.'))

import random
import json
import numpy as np
import cvxpy as cp
import torch
from modules.network import ControlAffineDynamics
from modules import utils
from modules import gurobi_milp
gurobi_milp.setM(1e6)
from modules.gurobi_milp import plan, interpolate


def get_scene(num_agents, obstacle_equations, env_size=100):
    """
    args:
        num_agents (int): The number of agents including NPC and controlled agents.
        obstacle_equations (list of (O, 3)): Every (O, 3) represents an convex hull with
            O facets. The 3 elements are [a, b, c] where ax + by <= c.
        env_size (float): The range of the environment.
    returns:
        area_bound (2, 2): Bound of the environment.
        waypoints (num_agents, num_waypoints, 2): The waypoints for each agent where num_waypoints = 2. 
    """
        
    area_bound = np.array([
        [-10, env_size + 10], [-10, env_size + 10] # xmin, xmax, ymin, ymax
    ])
    safe_states = []
    obstacle_equations = [np.array(o) for o in obstacle_equations]
    while len(safe_states) < num_agents * 2:
        pos = np.random.uniform(size=(2,)) * env_size
        inside_obstacle = False
        for obs in obstacle_equations:
            # obs (O, 3)
            if np.all(obs[:, :2].dot(pos) <= obs[:, 2] + 2.0):
                # pos inside the obstacle
                inside_obstacle = True
                break
        if not inside_obstacle:
            safe_states.append(pos)
    
    init_states = np.array(safe_states[:num_agents])
    goal_states = np.array(safe_states[num_agents:])
    # waypoints (num_agents, num_waypoints, 2) where num_waypoints = 2
    waypoints = np.stack([init_states, goal_states], axis=1)
    obstacles = [[o[:, :2], o[:, 2]] for o in obstacle_equations]
    return obstacles, area_bound, waypoints


def smooth_trajectory(traj, ddx_lim=0.1):
    """
    args:
        traj (n, ndim): trajectory of length n and ndim state space.
    returns:
        smoothed_traj (n, ndim): trajectory after smoothing.
    """
    if traj.shape[0] < 5:
        return traj
    x = cp.Variable(traj.shape)
    dx = x[1:] - x[:-1]
    ddx = dx[1:] - dx[:-1]
    objective = cp.Minimize(cp.sum_squares(x - traj) + cp.sum_squares(dx))
    constraint = [-ddx_lim <= ddx, ddx <= ddx_lim]
    cp.Problem(objective, constraint).solve()
    smooth_traj = x.value
    return smooth_traj


class Ship(object):

    def __init__(self,
                 dt=0.1, 
                 k_obstacle=8, 
                 total_obstacle=20, 
                 env_size=20, 
                 safe_dist=1, 
                 max_steps=2000, 
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
                 max_steps=2000, 
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


class Valley(object):

    def __init__(self,
                 static_obstacle='data/mountains.json',
                 preplanned_traj=None,
                 dt=0.1,
                 k_obstacle=8,
                 env_size=100,
                 num_npc=16,
                 npc_speed=1.0,
                 safe_dist=1,
                 perception_range=12,
                 max_steps=2000, 
                 max_speed=np.array([0.3, 0.3, 1.0]),
                 estimated_param=None):
        assert num_npc >= k_obstacle
        self.dt = dt
        self.k_obstacle = k_obstacle
        self.env_size = env_size
        self.num_npc = num_npc
        self.npc_speed = npc_speed
        self.safe_dist = safe_dist
        self.perception_range = perception_range
        self.max_steps = max_steps
        self.max_speed = max_speed

        self.B_real = np.array([[1, 0], [0, 0.1], [0, 0.5]])
        self.B_nominal = np.array([[1, 0], [0, 0.1], [0, 0.5]])

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

        if preplanned_traj is None:
            # There are num_npc + 1 agents in total, including 1 controlled agent
            # and num_npc uncontrolled agents
            obstacles, area_bound, waypoints = \
                 get_scene(self.num_npc + 1, np.array(json.load(open(static_obstacle, 'r'))), env_size=self.env_size)
            self.reference_traj = []
            reference_traj_as_list = []
            for i in range(self.num_npc + 1):
                print('\n========= Planning for agent {} / {} =========\n'.format(i+1, self.num_npc + 1))
                x_init = np.concatenate([waypoints[i][0], np.zeros(4)])
                path = plan(x_init[:2], waypoints[i][1:], area_bound, obstacles)
                if path is None:
                    print('\n========= Planning failed for agent {} =========\n'.format(i+1))
                    continue
                traj, _ = interpolate(path, dt=2, max_speed=self.max_speed[0])
                traj = smooth_trajectory(traj)
                traj, ts = interpolate(traj, dt=self.dt, max_speed=self.max_speed[0])
                self.reference_traj.append({"traj": traj, "ts": ts})
                reference_traj_as_list.append({"traj": traj.tolist(), "ts": ts.tolist()})
            self.num_npc = len(self.reference_traj) - 1
            json.dump(reference_traj_as_list, open("data/ship_reference_traj.json", "w"), indent=4)
        else:
            reference_traj_as_list = json.load(open(preplanned_traj))
            self.num_npc = len(reference_traj_as_list) - 1
            self.reference_traj = []
            for single_agent_traj in reference_traj_as_list:
                self.reference_traj.append({
                    "traj": np.array(single_agent_traj["traj"]), 
                    "ts": np.array(single_agent_traj["ts"])})
        print('\nFinish preparing reference trajectories. Found {} NPC in total\n'.format(self.num_npc))

    def reset(self):
        self.t = 0
        random.shuffle(self.reference_traj)
        self.state = np.concatenate([self.reference_traj[0]['traj'][0], np.zeros(4)])
        obstacle = self.get_obstacle(self.state)
        goal = self.get_goal(self.state)
        return self.state, obstacle, goal

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
        done = int(self.t / self.dt) == len(self.reference_traj[0]['traj'])
        if np.linalg.norm(goal[:2] - state[:2]) > 20:
            # If too faraway from the reference trajectory, the tracking fails
            done = True
        self.t = min(self.t + self.dt, len(self.reference_traj[0]['traj']) * self.dt)
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
        pos = state[:2]
        ind_max = int(self.npc_speed * self.t / self.dt)
        obstacle = []
        for i in range(1, self.num_npc):
            ind = min(ind_max, len(self.reference_traj[i]['traj']) - 1)
            obstacle_pos = self.reference_traj[i]['traj'][ind][:2]
            if np.linalg.norm(pos - obstacle_pos) < self.perception_range:
                obstacle.append(obstacle_pos)

        # Make the number of surrounding obstacles always equal to self.k_obstacle
        if len(obstacle) < self.k_obstacle:
            if len(obstacle) == 0:
                obstacle.append(state[:2] + np.array([10, 10]))
            while len(obstacle) < self.k_obstacle:
                obstacle.append(obstacle[-1])
        elif len(obstacle) > self.k_obstacle:
            obstacle = np.array(obstacle)
            dist = np.linalg.norm(obstacle - pos, axis=1)
            indices = np.argsort(dist)
            obstacle = obstacle[indices[:self.k_obstacle]]
        else:
            pass
        obstacle = np.concatenate([np.array(obstacle), np.zeros((self.k_obstacle, 4))], axis=1)
        return obstacle

    def get_goal(self, state, forward_t=10.0):
        """
        args:
            state (n_state,): current state of the controlled agent.
            forward_t (float): time to look forward in the reference trajectory
        returns:
            goal (n_state,): the state in the reference trajectory at t + forward_t
        """
        ind = int((self.t + forward_t) / self.dt)
        ind = min(ind, len(self.reference_traj[0]['traj']) - 1)
        goal = self.reference_traj[0]['traj'][ind]
        goal = np.concatenate([goal, np.zeros((4,))])
        return goal


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    env = Valley(num_npc=16)#, preplanned_traj='data/ship_reference_traj.json')
    fig = plt.figure(figsize=(10, 10))
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    for ref in env.reference_traj:
        traj = ref['traj']
        plt.plot(traj[:, 0], traj[:, 1])
    plt.show()