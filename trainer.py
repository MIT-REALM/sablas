import torch
from torch import nn
import numpy as np


class Trainer(object):

    def __init__(self, controller, cbf, dataset, nominal_dynamics, n_pos, dt=0.1, safe_dist=1, dang_dist=0.6):
        self.controller = controller
        self.cbf = cbf
        self.dataset = dataset
        self.nominal_dynamics = nominal_dynamics
        self.controller_optimizer = torch.optim.Adam(
            self.controller.parameters(), lr=1e-4, weight_decay=1e-5)
        self.cbf_optimizer = torch.optim.Adam(
            self.cbf.parameters(), lr=1e-4, weight_decay=1e-5)
        self.n_pos = n_pos
        self.dt = dt
        self.safe_dist = safe_dist
        self.dang_dist = dang_dist


    def train_cbf(self, batch_size=256, opt_iter=50, eps=0.1):

        loss_np = 0.0
        acc_np = np.zeros((5,), dtype=np.float32)
        
        for i in range(opt_iter):
            # state (bs, n_state), obstacle (bs, k_obstacle, n_state)
            # u_nominal (bs, m_control), state_next (bs, n_state)
            state, obstacle, u_nominal, state_next, state_error = self.dataset.sample_data(batch_size)
            state = torch.from_numpy(state)
            obstacle = torch.from_numpy(obstacle)
            state_next = torch.from_numpy(state_next)

            safe_mask, dang_mask, mid_mask = self.get_mask(state, obstacle)
            h = self.cbf(state, obstacle)

            num_safe = torch.sum(safe_mask)
            num_dang = torch.sum(dang_mask)
            num_mid = torch.sum(mid_mask)

            loss_h_safe = torch.sum(nn.ReLU()(eps - h) * safe_mask) / (1e-5 + num_safe)
            loss_h_dang = torch.sum(nn.ReLU()(h + eps) * dang_mask) / (1e-5 + num_dang)

            acc_h_safe = torch.sum((h >= 0).float() * safe_mask) / (1e-5 + num_safe)
            acc_h_dang = torch.sum((h < 0).float() * dang_mask) / (1e-5 + num_dang)

            h_next = self.cbf(state_next, obstacle)
            deriv_cond = (h_next - h) / self.dt + h

            loss_deriv_safe = torch.sum(nn.ReLU()(-deriv_cond) * safe_mask) / (1e-5 + num_safe)
            loss_deriv_dang = torch.sum(nn.ReLU()(-deriv_cond) * dang_mask) / (1e-5 + num_dang)
            loss_deriv_mid = torch.sum(nn.ReLU()(-deriv_cond) * mid_mask) / (1e-5 + num_mid)

            acc_deriv_safe = torch.sum((deriv_cond > 0).float() * safe_mask) / (1e-5 + num_safe)
            acc_deriv_dang = torch.sum((deriv_cond > 0).float() * dang_mask) / (1e-5 + num_dang)
            acc_deriv_mid = torch.sum((deriv_cond > 0).float() * mid_mask) / (1e-5 + num_mid)

            loss = loss_h_safe + loss_h_dang + loss_deriv_safe + loss_deriv_dang + loss_deriv_mid

            self.cbf_optimizer.zero_grad()
            loss.backward()
            self.cbf_optimizer.step()

            # log statics
            acc_np[0] += acc_h_safe.detach().cpu().numpy()
            acc_np[1] += acc_h_dang.detach().cpu().numpy()

            acc_np[2] += acc_deriv_safe.detach().cpu().numpy()
            acc_np[3] += acc_deriv_dang.detach().cpu().numpy()
            acc_np[4] += acc_deriv_mid.detach().cpu().numpy()

            loss_np += loss.detach().cpu().numpy()

        acc_np = acc_np / opt_iter
        loss_np = loss_np / opt_iter
        return loss_np, acc_np

        
    def train_controller(self, batch_size=256, opt_iter=50, eps=0.1):

        loss_np = 0.0
        acc_np = np.zeros((3,), dtype=np.float32)

        for i in range(opt_iter):
            # state (bs, n_state), obstacle (bs, k_obstacle, n_state)
            # u_nominal (bs, m_control), state_next (bs, n_state)
            state, obstacle, u_nominal, state_next, state_error = self.dataset.sample_data(batch_size)
            state = torch.from_numpy(state)
            obstacle = torch.from_numpy(obstacle)
            u_nominal = torch.from_numpy(u_nominal)
            state_next = torch.from_numpy(state_next)
            state_error = torch.from_numpy(state_error)

            safe_mask, dang_mask, mid_mask = self.get_mask(state, obstacle)

            h = self.cbf(state, obstacle)

            u = self.controller(state, obstacle, u_nominal, state_error)
            dsdt_nominal = self.nominal_dynamics(state, u)
            state_next_nominal = state + dsdt_nominal * self.dt

            state_next_with_grad = state_next_nominal + (state_next - state_next_nominal).detach()

            h_next = self.cbf(state_next_with_grad, obstacle)
            deriv_cond = (h_next - h) / self.dt + h

            num_safe = torch.sum(safe_mask)
            num_dang = torch.sum(dang_mask)
            num_mid = torch.sum(mid_mask)

            loss_deriv_safe = torch.sum(nn.ReLU()(-deriv_cond) * safe_mask) / (1e-5 + num_safe)
            loss_deriv_dang = torch.sum(nn.ReLU()(-deriv_cond) * dang_mask) / (1e-5 + num_dang)
            loss_deriv_mid = torch.sum(nn.ReLU()(-deriv_cond) * mid_mask) / (1e-5 + num_mid)

            acc_deriv_safe = torch.sum((deriv_cond > 0).float() * safe_mask) / (1e-5 + num_safe)
            acc_deriv_dang = torch.sum((deriv_cond > 0).float() * dang_mask) / (1e-5 + num_dang)
            acc_deriv_mid = torch.sum((deriv_cond > 0).float() * mid_mask) / (1e-5 + num_mid)

            loss_action = torch.mean((u - u_nominal)**2)

            loss = loss_deriv_safe + loss_deriv_dang + loss_deriv_mid + loss_action * 0.08

            self.controller_optimizer.zero_grad()
            loss.backward()
            self.controller_optimizer.step()

            # log statics
            acc_np[0] += acc_deriv_safe.detach().cpu().numpy()
            acc_np[1] += acc_deriv_dang.detach().cpu().numpy()
            acc_np[2] += acc_deriv_mid.detach().cpu().numpy()

            loss_np += loss.detach().cpu().numpy()

        acc_np = acc_np / opt_iter
        loss_np = loss_np / opt_iter
        return loss_np, acc_np


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
        state = torch.unsqueeze(state, 2)[:, :self.n_pos]    # (bs, n_pos, 1)
        obstacle = obstacle.permute(0, 2, 1)[:, :self.n_pos] # (bs, n_pos, k_obstacle)
        dist = torch.norm(state - obstacle, dim=1)

        safe_mask = (dist >= self.safe_dist).float()
        dang_mask = (dist <= self.dang_dist).float()
        mid_mask = (1 - safe_mask) * (1 - dang_mask)

        return safe_mask, dang_mask, mid_mask


        
