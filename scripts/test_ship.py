import os
import sys
sys.path.insert(1, os.path.abspath('.'))

import numpy as np 
import torch
import json

from envs.env_ship import Ship, River, Valley
from modules.dataset import Dataset
from modules.trainer import Trainer
from modules.network import CBF, NNController
from modules import utils
from modules import config

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

np.set_printoptions(4)


def main(env_name='ship', preplanned_traj=None, npc_speed=0.5, vis=False, save_traj=False, goal_radius=2.0):
    if env_name == 'ship':
        env = Ship(max_steps=2000)
    elif env_name == 'river':
        env = River(max_steps=2000)
    elif env_name == 'valley':
        if save_traj:
            random_permute = False
        else:
            random_permute = True
        env = Valley(preplanned_traj=preplanned_traj, npc_speed=npc_speed, random_permute=random_permute)
    else:
        raise NotImplementedError

    preprocess_func = lambda x: utils.angle_to_sin_cos_torch(x, [2])
    
    nn_controller = NNController(n_state=7, k_obstacle=8, m_control=2, preprocess_func=preprocess_func, output_scale=1.1)
    nn_controller.load_state_dict(torch.load('./data/ship_controller_weights.pth', map_location=torch.device('cpu')))
    nn_controller.eval()

    cbf = CBF(n_state=7, k_obstacle=8, m_control=2, preprocess_func=preprocess_func)
    cbf.load_state_dict(torch.load('./data/ship_cbf_weights.pth', map_location=torch.device('cpu')))
    cbf.eval()

    state, obstacle, goal = env.reset()

    if vis:
        plt.ion()
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

    state_error = np.zeros((6,), dtype=np.float32)
    dt = 0.1

    safety_rate = 0
    goal_reached = 0
    num_episodes = 0
    traj_following_error = 0

    if save_traj:
        all_agent_state_list = []

    for i in range(config.EVAL_STEPS):
        u_nominal = env.nominal_controller(state, goal)
        u = nn_controller(
            torch.from_numpy(state.reshape(1, 6).astype(np.float32)), 
            torch.from_numpy(obstacle.reshape(1, 8, 6).astype(np.float32)),
            torch.from_numpy(u_nominal.reshape(1, 2).astype(np.float32)),
            torch.from_numpy(state_error.reshape(1, 6).astype(np.float32)))
        u = np.squeeze(u.detach().cpu().numpy())
        state_next, state_nominal_next, obstacle_next, goal_next, done = env.step(u)
        #state_next, state_nominal_next, obstacle_next, goal_next, done = env.step(u_nominal)

        is_safe = int(utils.is_safe(state, obstacle, n_pos=2, dang_dist=0.7))
        safety_rate = safety_rate * i / (i+1) + is_safe * 1.0 / (i+1)
        dist = np.linalg.norm(state[:2] - goal[:2])
        traj_following_error = traj_following_error * i / (i+1) + dist / (i+1)

        # error between the true current state and the state obtained from the nominal model
        # this error will be fed to the controller network in the next timestep
        state_error = (state_next - state_nominal_next) / dt

        state = state_next
        obstacle = obstacle_next
        goal = goal_next

        if save_traj:
            all_agent_state = env.get_all_agent_state()
            all_agent_state_list.append(all_agent_state.tolist())

        if done:
            num_episodes = num_episodes + 1
            goal_reached = goal_reached + 1 if dist < goal_radius else goal_reached
            state, obstacle, goal = env.reset()
            print('safety rate: {:.4f}, distance: {:.4f}'.format(safety_rate, dist))

            if save_traj:
                json.dump(all_agent_state_list, open('ship_trajectory.json', 'w'), indent=4)
                print('Trajectory saved to ship_trajectory.json. Finished.')
                break

        if vis and np.mod(i, 10) == 0:
            ax.clear()
            ax.scatter(obstacle[:, 0], obstacle[:, 1], color='grey')
            ax.scatter(state[0], state[1], color='darkred')
            rect = np.array([[0.6, 0.4], [-0.6, 0.4], [-0.6, -0.4], [0.6, -0.4]])
            yaw = state[2]
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            rect = rect.dot(R.T) + state[:2]
            p = PatchCollection([Polygon(rect, True)], alpha=0.1, color='darkred')
            ax.add_collection(p)
            ax.scatter(goal[0], goal[1], color='darkorange')
            if not is_safe:
                plt.scatter(state[0], state[1], color='darkblue')

            if env_name == 'valley':
                ax.set_xlim(state[0] - 10, state[0] + 10)
                ax.set_ylim(state[1] - 10, state[1] + 10)
            else:
                ax.set_xlim(0, 20)
                ax.set_ylim(0, 20)
            fig.canvas.draw()
            plt.pause(0.01)

    goal_reaching_success_rate = goal_reached * 1.0 / num_episodes
    print('Safety rate: {:.4f}, Goal reaching success rate: {:.4f}, Traj following error: {:.4f}'.format(
        safety_rate, goal_reaching_success_rate, traj_following_error))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='ship')
    parser.add_argument('--vis', type=int, default=0)
    parser.add_argument('--preplanned_traj', type=str, default=None)
    parser.add_argument('--npc_speed', type=float, default=0.2)
    parser.add_argument('--save_traj', type=int, default=0)
    args = parser.parse_args()

    main(args.env, args.preplanned_traj, args.npc_speed, args.vis, args.save_traj)
