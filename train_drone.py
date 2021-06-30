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


def main(vis=False):
    env = Drone()

    nn_controller = NNController(n_state=8, k_obstacle=8, m_control=3)
    cbf = CBF(n_state=8, k_obstacle=8, m_control=3)
    dataset = Dataset(n_state=8, k_obstacle=8, m_control=3, n_pos=3)
    trainer = Trainer(nn_controller, cbf, dataset, env.nominal_dynamics_torch, n_pos=3)
    state, obstacle, goal = env.reset()

    safety_rate = 0.0
    goal_reached = 0.0

    for i in range(config.TRAIN_STEPS):
        u_nominal = env.nominal_controller(state, goal)
        u = nn_controller(
            torch.from_numpy(state.reshape(1, 8).astype(np.float32)), 
            torch.from_numpy(obstacle.reshape(1, 8, 8).astype(np.float32)),
            torch.from_numpy(u_nominal.reshape(1, 3).astype(np.float32)))
        u = np.squeeze(u.detach().cpu().numpy())
        state_next, obstacle_next, goal_next, done = env.step(u)

        dataset.add_data(state, obstacle, u_nominal, state_next)

        is_safe = int(utils.is_safe(state, obstacle, n_pos=3, dang_dist=0.6))
        safety_rate = safety_rate * (1 - 1e-4) + is_safe * 1e-4

        state = state_next
        obstacle = obstacle_next
        goal = goal_next

        if np.mod(i, config.POLICY_UPDATE_INTERVAL) == 0 and i > 0:
            if np.mod(i // config.POLICY_UPDATE_INTERVAL, 2) == 0:
                loss_np, acc_np = trainer.train_cbf()
                print('step: {}, train h, loss: {:.3f}, safety rate: {:.3f}, goal reached: {:.3f}, acc: {}'.format(
                    i, loss_np, safety_rate, goal_reached, acc_np))
                torch.save(cbf.state_dict(), './data/drone_cbf_weights.pth')

            elif np.mod(i // config.POLICY_UPDATE_INTERVAL, 2) == 1:
                loss_np, acc_np = trainer.train_controller()
                print('step: {}, train u, loss: {:.3f}, safety rate: {:.3f}, goal reached: {:.3f}, acc: {}'.format(
                    i, loss_np, safety_rate, goal_reached, acc_np))
                torch.save(nn_controller.state_dict(), './data/drone_controller_weights.pth')

        if done:
            dist = np.linalg.norm(state[:3] - goal[:3])
            goal_reached = goal_reached * (1-1e-2) + (dist < 2.0) * 1e-2
            state, obstacle, goal = env.reset()

main()
