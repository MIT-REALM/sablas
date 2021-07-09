""" System identification of the drone model. """
import numpy as np
from env_drone import Drone

def main():
    env = Drone()
    state, obstacle, goal = env.reset()

    dt = 0.1
    id_steps = 10000

    sdot_data = [] # sdot (N, 8)
    su_data = []   # [s, u] (N, 11)

    for i in range(id_steps):
        u_nominal = env.nominal_controller(state, goal)
        u_nominal = u_nominal + np.random.normal(size=u_nominal.shape) * 0.2
        state_next, state_nominal_next, obstacle_next, goal_next, done = env.step(u_nominal)

        if np.abs(state_next[3:6]).max() < env.max_speed and np.abs(state_next[6:]).max() < env.max_theta:
            sdot_data.append((state_next - state) / dt)
            su_data.append(np.concatenate([state, u_nominal]))

        state = state_next
        obstacle = obstacle_next
        goal = goal_next

        if done:
            state, obstacle, goal = env.reset()

    sdot_data = np.array(sdot_data)
    su_data = np.array(su_data)

    mat = np.linalg.lstsq(su_data, sdot_data)[0].T
    A = mat[:, :8]
    B = mat[:, 8:]

    f = open('data/estimated_model_drone.npz', 'wb')
    np.savez(f, A=A, B=B)

main()