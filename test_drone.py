import numpy as np 
import torch
import config
from env_drone import Drone
from dataset import Dataset
import matplotlib.pyplot as plt
from trainer import Trainer
from network import CBF, NNController
import utils

np.set_printoptions(4)


def main(vis=True):
    env = Drone()
    
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
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.set_zlim(0, 20)
        ax.scatter(env.obstacle[:, 0], env.obstacle[:, 1], env.obstacle[:, 2], color='grey')

    safety_rate = 0.0

    for i in range(config.TRAIN_STEPS):
        u_nominal = env.nominal_controller(state, goal)
        u = nn_controller(
            torch.from_numpy(state.reshape(1, 8).astype(np.float32)), 
            torch.from_numpy(obstacle.reshape(1, 8, 8).astype(np.float32)),
            torch.from_numpy(u_nominal.reshape(1, 3).astype(np.float32)))
        u = np.squeeze(u.detach().cpu().numpy())
        state_next, obstacle_next, goal_next, done = env.step(u)

        is_safe = int(utils.is_safe(state, obstacle, n_pos=3, dang_dist=0.6))
        safety_rate = safety_rate * i / (i+1) + is_safe * 1.0 / (i+1)

        state = state_next
        obstacle = obstacle_next
        goal = goal_next

        if done:
            state, obstacle, goal = env.reset()
            print('safety rate: {:.4f}'.format(safety_rate))

        if vis and done:
            ax.clear()
            ax.set_xlim(0, 20)
            ax.set_ylim(0, 20)
            ax.set_zlim(0, 20)
            ax.scatter(env.obstacle[:, 0], env.obstacle[:, 1], env.obstacle[:, 2], color='grey')

        if vis and np.mod(i, 10) == 0:
            ax.scatter(state[0], state[1], state[2], color='darkred')
            ax.scatter(goal[0], goal[1], goal[2], color='darkorange')
            if not is_safe:
                plt.scatter(state[0], state[1], state[2], color='darkblue')
            fig.canvas.draw()
            plt.pause(0.01)
main()
