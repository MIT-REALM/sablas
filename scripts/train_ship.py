import os
import sys
sys.path.insert(1, os.path.abspath('.'))

import numpy as np 
import torch
from envs.env_ship import Ship
from modules.dataset import Dataset
from modules.trainer import Trainer
from modules.network import CBF, NNController
from modules.utils import angle_to_sin_cos_torch
from modules import config
import matplotlib.pyplot as plt

np.set_printoptions(4)


class ShipTrainer(Trainer):

    def get_mask(self, state, obstacle):
        """
        args:
            state (bs, n_state)
            obstacle (bs, k_obstacle, n_state)
        returns:
            safe_mask (bs, k_obstacle)
            mid_mask  (bs, k_obstacle)
            dang_mask (bs, k_obstacle)
        """
        yaw = state[:, 2:3].unsqueeze(-1)           # (bs, 1, 1)
        yaw_vect = torch.cat([torch.cos(yaw), torch.sin(yaw)], axis=1)
        yaw_vect_perp = torch.cat([-torch.sin(yaw), torch.cos(yaw)], axis=1)

        state = torch.unsqueeze(state, 2)[:, :2]    # (bs, 2, 1)
        obstacle = obstacle.permute(0, 2, 1)[:, :2] # (bs, 2, k_obstacle)
        diff = obstacle - state

        front_dist = torch.abs(torch.sum(diff * yaw_vect, axis=1))
        side_dist = torch.abs(torch.sum(diff * yaw_vect_perp, axis=1))

        dang_mask = torch.logical_and(
                front_dist <= self.dang_dist * 1.2, 
                side_dist <= self.dang_dist * 0.8).float()
        safe_mask = torch.logical_or(
                front_dist >= self.safe_dist * 1.2,
                side_dist >= self.safe_dist * 0.8).float()
        mid_mask = (1 - safe_mask) * (1 - dang_mask)

        return safe_mask, dang_mask, mid_mask


def is_safe_ship(state, obstacle, dang_dist):
    """
    args:
        state (n_state,)
        obstacle (k_obstacle, n_state)
        dang_dist (flost)
    returns:
        is_safe (bool)
    """
    yaw = state[2]
    yaw_vect = np.array([np.cos(yaw), np.sin(yaw)])
    yaw_vect_perp = np.array([-np.sin(yaw), np.cos(yaw)])

    diff = obstacle[:, :2] - state[:2]
    front_dist = np.abs(np.sum(diff * yaw_vect, axis=1))
    side_dist = np.abs(np.sum(diff * yaw_vect_perp, axis=1))
    is_dang = np.logical_and(front_dist < dang_dist * 1.2, side_dist < dang_dist * 0.8)
    is_safe = not np.any(is_dang)
    return is_safe


def main(gpu_id=0, estimated_param=None):
    # the original n_state is 6, but the state includes an angle. we convert the angle
    # to cos and sin before feeding into the controller and CBF, so the state length is 7
    preprocess_func = lambda x: angle_to_sin_cos_torch(x, [2])
    nn_controller = NNController(n_state=7, k_obstacle=8, m_control=2, preprocess_func=preprocess_func, output_scale=1.1)
    cbf = CBF(n_state=7, k_obstacle=8, m_control=2, preprocess_func=preprocess_func)

    use_gpu = torch.cuda.is_available() and gpu_id >= 0
    if use_gpu:
        print('Using GPU {}'.format(gpu_id))
        nn_controller = nn_controller.cuda(gpu_id)
        cbf = cbf.cuda(gpu_id)

    env = Ship(gpu_id=gpu_id if use_gpu else -1, estimated_param=estimated_param)
    # the dataset stores the orignal state representation, where n_state is 6
    dataset = Dataset(n_state=6, k_obstacle=8, m_control=2, n_pos=2, buffer_size=100000)
    trainer = ShipTrainer(nn_controller, cbf, dataset, env.nominal_dynamics_torch, 
                      n_pos=2, safe_dist=3.0, dang_dist=1.0, action_loss_weight=0.02, gpu_id=gpu_id if use_gpu else -1,
                      lr_decay_stepsize=int(config.TRAIN_STEPS/config.POLICY_UPDATE_INTERVAL/3))
    state, obstacle, goal = env.reset()
    add_action_noise = np.random.uniform() > 0.5

    state_error = np.zeros((6,), dtype=np.float32)

    safety_rate = 0.0
    goal_reached = 0.0
    dt = 0.1

    for i in range(config.TRAIN_STEPS):
        u_nominal = env.nominal_controller(state, goal)

        state_torch = torch.from_numpy(state.reshape(1, 6).astype(np.float32))
        obstacle_torch = torch.from_numpy(obstacle.reshape(1, 8, 6).astype(np.float32))
        u_nominal_torch = torch.from_numpy(u_nominal.reshape(1, 2).astype(np.float32))
        state_error_torch = torch.from_numpy(state_error.reshape(1, 6).astype(np.float32))

        if use_gpu:
            state_torch = state_torch.cuda(gpu_id)
            obstacle_torch = obstacle_torch.cuda(gpu_id)
            u_nominal_torch = u_nominal_torch.cuda(gpu_id)
            state_error_torch = state_error_torch.cuda(gpu_id)

        u = nn_controller(state_torch, obstacle_torch, u_nominal_torch, state_error_torch)
        u = np.squeeze(u.detach().cpu().numpy())

        if add_action_noise:
            # add noise to improve the diversity of the training samples
            u = u + np.random.normal(size=(2,)) * 3.0

        state_next, state_nominal_next, obstacle_next, goal_next, done = env.step(u)

        dataset.add_data(state, obstacle, u_nominal, state_next, state_error)

        if not add_action_noise:
            # without action noise, the safety performance is the performance of the
            # neural network controller
            is_safe = int(is_safe_ship(state, obstacle, dang_dist=0.7))
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

            torch.save(cbf.state_dict(), './data/ship_cbf_weights.pth')
            torch.save(nn_controller.state_dict(), './data/ship_controller_weights.pth')

        if done:
            dist = np.linalg.norm(state[:2] - goal[:2])
            if not add_action_noise:
                goal_reached = goal_reached * (1-1e-2) + (dist < 2.0) * 1e-2
            state, obstacle, goal = env.reset()
            add_action_noise = np.random.uniform() > 0.5

if __name__ == '__main__':
    main(estimated_param='./data/estimated_model_ship.pth')
