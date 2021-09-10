import os
import sys
sys.path.insert(1, os.path.abspath('.'))

import json
import numpy as np 
import torch
from envs.env_drone import Drone, City
from modules import config
from modules import utils
from modules.network import CBF, NNController
import matplotlib.pyplot as plt

np.set_printoptions(4)


def main(env_name='drone', preplanned_traj=None, npc_speed=0.5, vis=True, save_traj=False, estimated_param=None, goal_radius=2.0):
    if env_name == 'drone':
        env = Drone(estimated_param=estimated_param)
    elif env_name == 'city':
        if save_traj:
            random_permute = False
        else:
            random_permute = True
        env = City(estimated_param=estimated_param, preplanned_traj=preplanned_traj, npc_speed=npc_speed, random_permute=random_permute)
    else:
        raise NotImplementedError
    
    nn_controller = NNController(n_state=8, k_obstacle=8, m_control=3)
    nn_controller.load_state_dict(torch.load('./data/drone_controller_weights.pth'))
    nn_controller.eval()

    cbf = CBF(n_state=8, k_obstacle=8, m_control=3)
    cbf.load_state_dict(torch.load('./data/drone_cbf_weights.pth'))
    cbf.eval()

    state, obstacle, goal = env.reset()

    if vis:
        plt.ion()
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

    state_error = np.zeros((8,), dtype=np.float32)
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
            torch.from_numpy(state.reshape(1, 8).astype(np.float32)), 
            torch.from_numpy(obstacle.reshape(1, 8, 8).astype(np.float32)),
            torch.from_numpy(u_nominal.reshape(1, 3).astype(np.float32)),
            torch.from_numpy(state_error.reshape(1, 8).astype(np.float32)))
        u = np.squeeze(u.detach().cpu().numpy())
        state_next, state_nominal_next, obstacle_next, goal_next, done = env.step(u)
        #state_next, state_nominal_next, obstacle_next, goal_next, done = env.step(u_nominal)

        is_safe = int(utils.is_safe(state, obstacle, n_pos=3, dang_dist=0.6))
        safety_rate = safety_rate * i / (i+1) + is_safe * 1.0 / (i+1)
        dist = np.linalg.norm(state[:3] - goal[:3])
        traj_following_error = traj_following_error * i / (i+1) + dist / (i+1)

        # Error between the true current state and the state obtained from the nominal model
        # This error will be fed to the controller network in the next timestep
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
            print('Progress: {:.2f}% safety rate: {:.4f}, distance: {:.4f}'.format(
                100 * (i + 1.0) / config.EVAL_STEPS, safety_rate, dist))

            if save_traj:
                json.dump(all_agent_state_list, open('drone_trajectory.json', 'w'), indent=4)
                print('Trajectory saved to drone_trajectory.json. Finished.')
                break

        if vis and np.mod(i, 20) == 0:
            ax.clear()
            if env_name == 'city':
                ax.set_xlim(state[0] - 5, state[0] + 5)
                ax.set_ylim(state[1] - 5, state[1] + 5)
                ax.set_zlim(state[2] - 5, state[2] + 5)
            elif env_name == 'drone':
                ax.set_xlim(0, 20)
                ax.set_ylim(0, 20)
                ax.set_zlim(0, 20)
            else:
                raise NotImplementedError

            ax.scatter(state[0], state[1], state[2], color='darkred')
            ax.scatter(goal[0], goal[1], goal[2], color='darkorange')
            ax.scatter(obstacle[:, 0], obstacle[:, 1], obstacle[:, 2], color='grey')
            if not is_safe:
                plt.scatter(state[0], state[1], state[2], color='darkblue')
            fig.canvas.draw()
            plt.pause(0.01)

    goal_reaching_success_rate = goal_reached * 1.0 / num_episodes
    print('Safety rate: {:.4f}, Goal reaching success rate: {:.4f}, Traj following error: {:.4f}'.format(
        safety_rate, goal_reaching_success_rate, traj_following_error))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='drone')
    parser.add_argument('--preplanned_traj', type=str, default=None)
    parser.add_argument('--npc_speed', type=float, default=0.5)
    parser.add_argument('--vis', type=int, default=0)
    parser.add_argument('--save_traj', type=int, default=0)
    parser.add_argument('--param', type=str, default='./data/estimated_model_drone.npz')
    args = parser.parse_args()

    estimated_param = np.load(open(args.param, 'rb'))
    main(args.env, args.preplanned_traj, args.npc_speed, args.vis, args.save_traj, estimated_param)
