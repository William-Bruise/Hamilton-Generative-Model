from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, depth: int = 3):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HamiltonianNet(nn.Module):
    """H_theta(q,p,t) -> scalar."""

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.dim = dim
        self.mlp = MLP(in_dim=2 * dim + 1, out_dim=1, hidden=hidden)

    def forward(self, q: torch.Tensor, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(q.size(0), 1)
        elif t.ndim == 1:
            t = t[:, None]
        x = torch.cat([q, p, t], dim=-1)
        return self.mlp(x).squeeze(-1)


class ControlNet(nn.Module):
    """u_phi(q,p,t) -> R^d, added on momentum equation."""

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.mlp = MLP(in_dim=2 * dim + 1, out_dim=dim, hidden=hidden)

    def forward(self, q: torch.Tensor, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(q.size(0), 1)
        elif t.ndim == 1:
            t = t[:, None]
        x = torch.cat([q, p, t], dim=-1)
        return self.mlp(x)


@dataclass
class DynamicsOutput:
    dq: torch.Tensor
    dp: torch.Tensor


class HamiltonianDynamics(nn.Module):
    def __init__(self, h_net: HamiltonianNet, u_net: ControlNet | None = None):
        super().__init__()
        self.h_net = h_net
        self.u_net = u_net

    def forward(self, q: torch.Tensor, p: torch.Tensor, t: torch.Tensor) -> DynamicsOutput:
        q_req = q.requires_grad_(True)
        p_req = p.requires_grad_(True)

        h = self.h_net(q_req, p_req, t).sum()
        grad_q, grad_p = torch.autograd.grad(h, [q_req, p_req], create_graph=True)

        dq = grad_p
        dp = -grad_q
        if self.u_net is not None:
            dp = dp + self.u_net(q_req, p_req, t)

        return DynamicsOutput(dq=dq, dp=dp)


class SymplecticEulerIntegrator(nn.Module):
    """One-step symplectic Euler for controlled Hamiltonian dynamics."""

    def __init__(self, dynamics: HamiltonianDynamics, t0: float = 0.0, t1: float = 1.0, steps: int = 32):
        super().__init__()
        self.dynamics = dynamics
        self.t0 = t0
        self.t1 = t1
        self.steps = steps

    def integrate(self, q0: torch.Tensor, p0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dt = (self.t1 - self.t0) / self.steps
        q, p = q0, p0

        for k in range(self.steps):
            t = torch.full((q.size(0),), self.t0 + k * dt, device=q.device, dtype=q.dtype)
            out1 = self.dynamics(q, p, t)
            p = p + dt * out1.dp

            out2 = self.dynamics(q, p, t)
            q = q + dt * out2.dq

        return q, p


class HamiltonianGenerativeModel(nn.Module):
    def __init__(self, dim: int, hidden: int = 256, steps: int = 32):
        super().__init__()
        self.dim = dim
        self.h_net = HamiltonianNet(dim=dim, hidden=hidden)
        self.u_net = ControlNet(dim=dim, hidden=hidden)
        self.dynamics = HamiltonianDynamics(self.h_net, self.u_net)
        self.integrator = SymplecticEulerIntegrator(self.dynamics, steps=steps)

    def sample_prior(self, n: int, device: torch.device | str) -> Tuple[torch.Tensor, torch.Tensor]:
        q0 = torch.randn(n, self.dim, device=device)
        p0 = torch.randn(n, self.dim, device=device)
        return q0, p0

    def transport(self, q0: torch.Tensor, p0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.integrator.integrate(q0, p0)

    @torch.no_grad()
    def generate_latent(self, n: int, device: torch.device | str) -> torch.Tensor:
        q0, p0 = self.sample_prior(n, device)
        qT, _ = self.transport(q0, p0)
        return qT


def compute_mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    def pdist2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a2 = (a * a).sum(dim=1, keepdim=True)
        b2 = (b * b).sum(dim=1, keepdim=True).T
        return a2 + b2 - 2.0 * (a @ b.T)

    k_xx = torch.exp(-pdist2(x, x) / (2 * sigma * sigma)).mean()
    k_yy = torch.exp(-pdist2(y, y) / (2 * sigma * sigma)).mean()
    k_xy = torch.exp(-pdist2(x, y) / (2 * sigma * sigma)).mean()
    return k_xx + k_yy - 2 * k_xy
