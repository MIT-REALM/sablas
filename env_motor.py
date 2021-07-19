import numpy as np 
import torch 


class Motor(object):
    """ 
    Control two motors such that their angular velocities follow their
    desired reference trajectories, and at the same time the difference
    between their angular velocity should be lower than a given threshold.
    """
    def __init__(self, dt=0.1):
        self.dt = dt 

    def uncertain_dynamics(self, state, u):
        """
        args:
            state (4,): i_0, omega_0, i_1, omega_1
            u (2,): u_0, u_1
        returns:
            dsdt (4,)
        """
        dsdt = np.zeros((4,))
        dsdt[0] = u[0] - state[1] - state[0]  # didt = u - omega - i
        dsdt[1] = state[0] - 0.1 * state[1]   # domegadt = i - T(omega)
        dsdt[2] = u[1] - state[3] - state[2]  # didt = u - omega - i
        dsdt[3] = state[2] - 0.1 * state[3]   # domegadt = i - T(omega)
        return dsdt 

    def nominal_dynamics(self, state, u):
        """
        args:
            state (4,): i_0, omega_0, i_1, omega_1
            u (2,): u_0, u_1
        returns:
            dsdt (4,)
        """
        dsdt = np.zeros((4,))
        dsdt[0] = u[0] - state[1] - state[0]  # didt = u - omega - i
        dsdt[1] = state[0] - 0.1 * state[1]   # domegadt = i - T(omega)
        dsdt[2] = u[1] - state[3] - state[2]  # didt = u - omega - i
        dsdt[3] = state[2] - 0.1 * state[3]   # domegadt = i - T(omega)
        return dsdt

    def nominal_dynamics_torch(self, state, u):
        """
        args:
            state (bs, 4): i_0, omega_0, i_1, omega_1
            u (bs, 2): u_0, u_1
        returns:
            dsdt (bs, 4)
        """
        i_0, omega_0, i_1, omega_1 = torch.split(state, 4, dim=1)
        u_0, u_1 = torch.split(u, 2, dim=1)

        dsdt_0 = u_0 - omega_0 - i_0
        dsdt_1 = i_0 - 0.1 * omega_0
        dsdt_2 = u_1 - omega_1 - i_1
        dsdt_3 = i_1 - 0.1 * omega_1

        dsdt = torch.cat([dsdt_0, dsdt_1, dsdt_2, dsdt_3], dim=1)
        return dsdt

    def nominal_controller(self, state, )
    