import os
import sys
sys.path.insert(1, os.path.abspath('.'))

import numpy as np 
import torch
from envs.env_drone import Drone
from modules import config
from modules import utils
from modules.dataset import Dataset
from modules.trainer import Trainer
from modules.network import CBF, NNController
import matplotlib.pyplot as plt

np.set_printoptions(4)


def main(estimated_param=None):
    env = Drone(estimated_param=estimated_param)

    nn_controller = NNController(n_state=8, k_obstacle=8, m_control=3)
    cbf = CBF(n_state=8, k_obstacle=8, m_control=3)
    dataset = Dataset(n_state=8, k_obstacle=8, m_control=3, n_pos=3)
    trainer = Trainer(nn_controller, cbf, dataset, env.nominal_dynamics_torch, n_pos=3, action_loss_weight=0.1)
    state, obstacle, goal = env.reset()

    state_error = np.zeros((8,), dtype=np.float32)

    safety_rate = 0.0
    goal_reached = 0.0

    dt = 0.1

    for i in range(config.TRAIN_STEPS):
        u_nominal = env.nominal_controller(state, goal)
        u = nn_controller(
            torch.from_numpy(state.reshape(1, 8).astype(np.float32)), 
            torch.from_numpy(obstacle.reshape(1, 8, 8).astype(np.float32)),
            torch.from_numpy(u_nominal.reshape(1, 3).astype(np.float32)),
            torch.from_numpy(state_error.reshape(1, 8).astype(np.float32)))
        u = np.squeeze(u.detach().cpu().numpy())
        state_next, state_nominal_next, obstacle_next, goal_next, done = env.step(u)

        dataset.add_data(state, obstacle, u_nominal, state_next, state_error)

        is_safe = int(utils.is_safe(state, obstacle, n_pos=3, dang_dist=0.6))
        safety_rate = safety_rate * (1 - 1e-4) + is_safe * 1e-4

        # error between the true current state and the state obtained from the nominal model
        # this error will be fed to the controller network in the next timestep
        state_error = (state_next - state_nominal_next) / dt

        state = state_next
        obstacle = obstacle_next
        goal = goal_next

        if np.mod(i, config.POLICY_UPDATE_INTERVAL) == 0 and i > 0:

            loss_np, acc_np = trainer.train_cbf_and_controller()
            print('step: {}, train h and u, loss: {:.3f}, safety rate: {:.3f}, goal reached: {:.3f}, acc: {}'.format(
                i, loss_np, safety_rate, goal_reached, acc_np))

            torch.save(cbf.state_dict(), './data/drone_cbf_weights.pth')
            torch.save(nn_controller.state_dict(), './data/drone_controller_weights.pth')

        if done:
            dist = np.linalg.norm(state[:3] - goal[:3])
            goal_reached = goal_reached * (1-1e-2) + (dist < 2.0) * 1e-2
            state, obstacle, goal = env.reset()

if __name__ == '__main__':
    estimated_param = np.load(open('./data/estimated_model_drone.npz', 'rb'))
    main(estimated_param)
