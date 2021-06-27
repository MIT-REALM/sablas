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


def main(vis=False):
    env = Environment()
    nominal_controller = NominalController()
    nn_controller = NNController(n_state=4, k_obstacle=8, m_control=2)
    cbf = CBF(n_state=4, k_obstacle=8, m_control=2)
    dataset = Dataset(n_state=4, k_obstacle=8, m_control=2)
    trainer = Trainer(nn_controller, cbf, dataset, env.nominal_dynamics_torch, n_pos=2)
    state, obstacle, goal = env.reset()

    if vis:
        plt.ion()
        fig = plt.figure(figsize=(10, 10))
        plt.xlim(0, 10)
        plt.ylim(0, 10)
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
        dataset.add_data(state, obstacle, u_nominal, state_next)

        is_safe = int(utils.is_safe(state, obstacle, n_pos=2, dang_dist=0.6))
        safety_rate = safety_rate * (1 - 1e-4) + is_safe * 1e-4

        state = state_next
        obstacle = obstacle_next
        goal = goal_next

        if np.mod(i, config.POLICY_UPDATE_INTERVAL) == 0 and i > 0:
            if np.mod(i // config.POLICY_UPDATE_INTERVAL, 2) == 0:
                loss_np, acc_np = trainer.train_cbf()
                print('step: {}, train h, loss: {:.3f}, safety rate: {:.3f}, acc: {}'.format(i, loss_np, safety_rate, acc_np))

            elif np.mod(i // config.POLICY_UPDATE_INTERVAL, 2) == 1:
                loss_np, acc_np = trainer.train_controller()
                print('step: {}, train u, loss: {:.3f}, safety rate: {:.3f}, acc: {}'.format(i, loss_np, safety_rate, acc_np))

        if done:
            state, obstacle, goal = env.reset()

        if vis and done:
            plt.clf()
            plt.xlim(0, 10)
            plt.ylim(0, 10)
            plt.scatter(env.obstacle[:, 0], env.obstacle[:, 1], color='grey')

        if vis and np.mod(i, 10) == 0:
            plt.scatter(state[0], state[1], color='darkred')
            plt.scatter(goal[0], goal[1], color='darkorange')
            fig.canvas.draw()
            plt.pause(0.01)


main()