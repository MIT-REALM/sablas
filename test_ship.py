import numpy as np 
import torch
import config
from env_ship import Ship, River
from dataset import Dataset
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from trainer import Trainer
from network import CBF, NNController
import utils

np.set_printoptions(4)


def main(env='ship', vis=True, estimated_param=None):
    if env == 'ship':
        env = Ship(max_steps=2000)
    elif env == 'river':
        env = River(max_steps=2000)
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
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.scatter(env.obstacle[:, 0], env.obstacle[:, 1], color='grey')

    state_error = np.zeros((6,), dtype=np.float32)
    dt = 0.1

    safety_rate = 0.0
    goal_reached = 0.0

    for i in range(config.TRAIN_STEPS):
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

        # error between the true current state and the state obtained from the nominal model
        # this error will be fed to the controller network in the next timestep
        state_error = (state_next - state_nominal_next) / dt

        state = state_next
        obstacle = obstacle_next
        goal = goal_next

        if done:
            dist = np.linalg.norm(state[:2] - goal[:2])
            state, obstacle, goal = env.reset()
            print('safety rate: {:.4f}, distance: {:.4f}'.format(safety_rate, dist))

        if vis and done:
            ax.clear()
            ax.set_xlim(0, 20)
            ax.set_ylim(0, 20)
            ax.scatter(env.obstacle[:, 0], env.obstacle[:, 1], color='grey')

        if vis and np.mod(i, 10) == 0:
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
            fig.canvas.draw()
            plt.pause(0.01)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='ship')
    parser.add_argument('--vis', type=int, default=0)
    parser.add_argument('--param', type=str, default='data/estimated_model_drone.npz')
    args = parser.parse_args()

    main(args.env, args.vis, estimated_param=None)
