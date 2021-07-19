import numpy as np 
import torch
import config
from env_ship import Ship
from dataset import Dataset
import matplotlib.pyplot as plt
from trainer import Trainer
from network import CBF, NNController
import utils

np.set_printoptions(4)


def main(gpu_id=0):
    # the original n_state is 6, but the state includes an angle. we convert the angle
    # to cos and sin before feeding into the controller and CBF, so the state length is 7
    preprocess_func = lambda x: utils.angle_to_sin_cos_torch(x, [2])
    nn_controller = NNController(n_state=7, k_obstacle=8, m_control=2, preprocess_func=preprocess_func)
    cbf = CBF(n_state=7, k_obstacle=8, m_control=2, preprocess_func=preprocess_func)

    use_gpu = torch.cuda.is_available() and gpu_id >= 0
    if use_gpu:
        print('Using GPU {}'.format(gpu_id))
        nn_controller = nn_controller.cuda(gpu_id)
        cbf = cbf.cuda(gpu_id)

    env = Ship(gpu_id=gpu_id if use_gpu else -1)
    # the dataset stores the orignal state representation, where n_state is 6
    dataset = Dataset(n_state=6, k_obstacle=8, m_control=2, n_pos=2, buffer_size=100000)
    trainer = Trainer(nn_controller, cbf, dataset, env.nominal_dynamics_torch, 
                      n_pos=2, safe_dist=3.0, dang_dist=0.7, action_loss_weight=0.01, gpu_id=gpu_id if use_gpu else -1)
    state, obstacle, goal = env.reset()
    add_action_noise = np.random.uniform() > 0.5

    state_error = np.zeros((6,), dtype=np.float32)

    safety_rate = 0.0
    goal_reached = 0.0
    dt = 0.1

    for i in range(config.TRAIN_STEPS * 5):
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
            #if np.random.uniform() > 0.5:
            #    # add noise to improve the diversity of the training samples
            #    u = u + np.random.normal(size=(2,)) * 3.0
            #else:
            u = u_nominal

        state_next, state_nominal_next, obstacle_next, goal_next, done = env.step(u)

        dataset.add_data(state, obstacle, u_nominal, state_next, state_error)

        if not add_action_noise:
            # without action noise, the safety performance is the performance of the
            # neural network controller
            is_safe = int(utils.is_safe(state, obstacle, n_pos=2, dang_dist=0.7))
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
    main()
