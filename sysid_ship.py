""" System identification of the ship model. """
import numpy as np
import torch
from env_ship import Ship
from network import ControlAffineDynamics
import utils
from tqdm import tqdm

def main():
    env = Ship()
    state, obstacle, goal = env.reset()

    dt = 0.1
    data_steps = int(1e+6)

    sdot_data = [] # sdot (N, 8)
    su_data = []   # [s, u] (N, 11)

    print('Collecting state samples')
    for i in tqdm(range(data_steps)):
        u_nominal = env.nominal_controller(state, goal)
        u_nominal = u_nominal + np.random.normal(size=u_nominal.shape)
        state_next, state_nominal_next, obstacle_next, goal_next, done = env.step(u_nominal)

        if not np.any(np.abs(state_next[3:]) == env.max_speed):
            sdot_data.append((state_next - state) / dt)
            su_data.append(np.concatenate([state, u_nominal]))

        state = state_next
        obstacle = obstacle_next
        goal = goal_next

        if done:
            state, obstacle, goal = env.reset()

    sdot_data = np.array(sdot_data).astype(np.float32)
    su_data = np.array(su_data).astype(np.float32)
    s_data, u_data = su_data[:, :6], su_data[:, 6:]

    num_samples = sdot_data.shape[0]
    print('Collected {} samples'.format(num_samples))

    print('Fitting model')
    preprocess_func = lambda x: utils.angle_to_sin_cos_torch(x, [2])
    net = ControlAffineDynamics(n_state=6, m_control=2, n_extended_state=1, preprocess_func=preprocess_func)
    optim = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)

    fitting_steps = int(1e+5)
    fitting_batch_size = 1024
    loss_np = 0.0

    for i in tqdm(range(fitting_steps)):
        ind = np.random.randint(num_samples, size=(fitting_batch_size,))
        sdot = torch.from_numpy(sdot_data[ind])
        s = torch.from_numpy(s_data[ind])
        u = torch.from_numpy(u_data[ind])

        f, B = net(s, u)
        sdot_pred = f + torch.bmm(B, u.unsqueeze(-1)).squeeze(-1)

        loss = torch.mean((sdot_pred - sdot)**2)
        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_np = loss_np * (1-1e-4) + loss.detach().cpu().numpy() * 1e-4
        
    print('MSE: {:.5f}'.format(loss_np))
    torch.save(net.state_dict(), './data/estimated_model_ship.pth')

main()