import numpy as np 
import torch
import config
from environment import Environment
from dataset import Dataset
from controller import NominalController, NNController
import matplotlib.pyplot as plt
from trainer import Trainer
from cbf import CBF
import utils

np.set_printoptions(4)


def main(vis=True):
    env = Environment()
    nominal_controller = NominalController()

    nn_controller = NNController(n_state=4, k_obstacle=8, m_control=2)
    nn_controller.load_state_dict(torch.load('./data/controller_weights.pth'))
    nn_controller.eval()

    cbf = CBF(n_state=4, k_obstacle=8, m_control=2)
    cbf.load_state_dict(torch.load('./data/cbf_weights.pth'))
    cbf.eval()

    state, obstacle, goal = env.reset()

    if vis:
        plt.ion()
        fig = plt.figure(figsize=(10, 10))
        plt.xlim(0, 20)
        plt.ylim(0, 20)
        plt.scatter(env.obstacle[:, 0], env.obstacle[:, 1], color='grey')

    safety_rate = 0.0

    for i in range(config.TRAIN_STEPS):
        u_nominal = nominal_controller(state, goal)
        u = nn_controller(
            torch.from_numpy(state.reshape(1, 4).astype(np.float32)), 
            torch.from_numpy(obstacle.reshape(1, 8, 4).astype(np.float32)),
            torch.from_numpy(u_nominal.reshape(1, 2).astype(np.float32)))
        u = np.squeeze(u.detach().cpu().numpy())
        state_next, obstacle_next, goal_next, done = env.step(u)

        is_safe = int(utils.is_safe(state, obstacle, n_pos=2, dang_dist=0.6))
        safety_rate = safety_rate * i / (i+1) + is_safe * 1.0 / (i+1)

        state = state_next
        obstacle = obstacle_next
        goal = goal_next

        if done:
            state, obstacle, goal = env.reset()
            print('safety rate: {:.4f}'.format(safety_rate))

        if vis and done:
            plt.clf()
            plt.xlim(0, 20)
            plt.ylim(0, 20)
            plt.scatter(env.obstacle[:, 0], env.obstacle[:, 1], color='grey')

        if vis and np.mod(i, 10) == 0:
            plt.scatter(state[0], state[1], color='darkred')
            plt.scatter(goal[0], goal[1], color='darkorange')
            if not is_safe:
                plt.scatter(state[0], state[1], color='darkblue')
            fig.canvas.draw()
            plt.pause(0.01)
main()
