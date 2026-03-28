from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t[None]
        if t.ndim == 2:
            t = t.squeeze(-1)
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1))
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class ResFiLMBlock(nn.Module):
    def __init__(self, width: int, tdim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.lin1 = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.lin2 = nn.Linear(width, width)
        self.act = nn.SiLU()
        self.film = nn.Linear(tdim, 2 * width)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(t_emb).chunk(2, dim=-1)
        h = self.norm1(x)
        h = h * (1 + scale) + shift
        h = self.act(self.lin1(h))
        h = self.act(self.lin2(self.norm2(h)))
        return x + h


class TimeConditionedResNet(nn.Module):
    """A stronger alternative to plain MLP for latent dynamics modeling."""

    def __init__(self, in_dim: int, out_dim: int, width: int = 512, depth: int = 6, tdim: int = 128):
        super().__init__()
        self.t_embed = nn.Sequential(
            SinusoidalTimeEmbedding(tdim),
            nn.Linear(tdim, tdim),
            nn.SiLU(),
            nn.Linear(tdim, tdim),
        )
        self.in_proj = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList([ResFiLMBlock(width, tdim) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(width)
        self.out_proj = nn.Linear(width, out_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(x.size(0))
        if t.ndim == 2:
            t = t.squeeze(-1)
        t_emb = self.t_embed(t)
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h, t_emb)
        return self.out_proj(self.out_norm(h))


class HamiltonianNet(nn.Module):
    """H_theta(q,p,t) -> scalar using time-conditioned residual network."""

    def __init__(self, dim: int, width: int = 512, depth: int = 6):
        super().__init__()
        self.dim = dim
        self.backbone = TimeConditionedResNet(in_dim=2 * dim, out_dim=1, width=width, depth=depth)

    def forward(self, q: torch.Tensor, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([q, p], dim=-1)
        return self.backbone(x, t).squeeze(-1)


class ControlNet(nn.Module):
    """u_phi(q,p,t) -> R^d, added on momentum equation."""

    def __init__(self, dim: int, width: int = 512, depth: int = 6):
        super().__init__()
        self.backbone = TimeConditionedResNet(in_dim=2 * dim, out_dim=dim, width=width, depth=depth)

    def forward(self, q: torch.Tensor, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([q, p], dim=-1)
        return self.backbone(x, t)


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
        # Hamiltonian vector field needs gradients wrt state even at inference time.
        with torch.enable_grad():
            q_req = q.detach().requires_grad_(True)
            p_req = p.detach().requires_grad_(True)

            h = self.h_net(q_req, p_req, t).sum()
            grad_q, grad_p = torch.autograd.grad(h, [q_req, p_req], create_graph=self.training)

            dq = grad_p
            dp = -grad_q
            if self.u_net is not None:
                dp = dp + self.u_net(q_req, p_req, t)

        return DynamicsOutput(dq=dq, dp=dp)


class SymplecticEulerIntegrator(nn.Module):
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
    def __init__(self, dim: int, width: int = 512, depth: int = 6, steps: int = 32):
        super().__init__()
        self.dim = dim
        self.h_net = HamiltonianNet(dim=dim, width=width, depth=depth)
        self.u_net = ControlNet(dim=dim, width=width, depth=depth)
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


def compute_mmd_rbf(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: float | None = None,
    scales: tuple[float, ...] = (0.5, 1.0, 2.0),
    unbiased: bool = True,
) -> torch.Tensor:
    """RBF MMD with optional median-bandwidth heuristic and multi-kernel averaging.

    Notes:
    - `unbiased=True` removes the diagonal terms, preventing artificial floors like ~1/B.
    - If `sigma is None`, median pairwise distance heuristic is used.
    """

    def pdist2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a2 = (a * a).sum(dim=1, keepdim=True)
        b2 = (b * b).sum(dim=1, keepdim=True).T
        return (a2 + b2 - 2.0 * (a @ b.T)).clamp_min(0.0)

    d_xx = pdist2(x, x)
    d_yy = pdist2(y, y)
    d_xy = pdist2(x, y)

    if sigma is None:
        with torch.no_grad():
            mix = torch.cat([d_xx.flatten(), d_yy.flatten(), d_xy.flatten()])
            mix = mix[mix > 0]
            if mix.numel() == 0:
                sigma = 1.0
            else:
                sigma = torch.sqrt(torch.median(mix)).item()
                sigma = max(sigma, 1e-4)

    def kernel_mean(d: torch.Tensor, s: float, drop_diag: bool = False) -> torch.Tensor:
        k = torch.exp(-d / (2 * s * s))
        if drop_diag:
            n = k.shape[0]
            k = k - torch.diag(torch.diag(k))
            return k.sum() / max(n * (n - 1), 1)
        return k.mean()

    k_xx = 0.0
    k_yy = 0.0
    k_xy = 0.0
    for a in scales:
        s = max(float(sigma) * a, 1e-6)
        k_xx = k_xx + kernel_mean(d_xx, s, drop_diag=unbiased)
        k_yy = k_yy + kernel_mean(d_yy, s, drop_diag=unbiased)
        k_xy = k_xy + kernel_mean(d_xy, s, drop_diag=False)

    k_xx = k_xx / len(scales)
    k_yy = k_yy / len(scales)
    k_xy = k_xy / len(scales)
    return k_xx + k_yy - 2 * k_xy
