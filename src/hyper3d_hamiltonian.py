from __future__ import annotations

from dataclasses import dataclass

import math
import torch
import torch.nn as nn


class TimeEmbed(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t[None]
        if t.ndim == 2:
            t = t.squeeze(-1)
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1))
        e = torch.cat([torch.sin(t[:, None] * freqs[None]), torch.cos(t[:, None] * freqs[None])], dim=-1)
        if self.dim % 2 == 1:
            e = torch.cat([e, torch.zeros_like(e[:, :1])], dim=-1)
        return self.mlp(e)


class FiLM3DBlock(nn.Module):
    def __init__(self, ch: int, tdim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv3d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv3d(ch, ch, 3, padding=1)
        self.act = nn.SiLU()
        self.film = nn.Linear(tdim, ch * 2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        s, b = self.film(t_emb).chunk(2, dim=-1)
        s = s[:, :, None, None, None]
        b = b[:, :, None, None, None]

        h = self.norm1(x)
        h = h * (1 + s) + b
        h = self.act(self.conv1(h))
        h = self.act(self.conv2(self.norm2(h)))
        return x + h


class HyperHamiltonianNet3D(nn.Module):
    """Scalar Hamiltonian H(q,p,t) for q,p in R^{C x H x W} using fully-conv 3D net."""

    def __init__(self, width: int = 64, depth: int = 6, tdim: int = 128):
        super().__init__()
        self.t_emb = TimeEmbed(tdim)
        self.in_conv = nn.Conv3d(2, width, 3, padding=1)
        self.blocks = nn.ModuleList([FiLM3DBlock(width, tdim) for _ in range(depth)])
        self.out = nn.Conv3d(width, 1, 1)

    def forward(self, q: torch.Tensor, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([q, p], dim=1)  # B,2,C,H,W
        te = self.t_emb(t)
        h = self.in_conv(x)
        for blk in self.blocks:
            h = blk(h, te)
        energy_density = self.out(h)
        return energy_density.mean(dim=(1, 2, 3, 4))


class HyperControlNet3D(nn.Module):
    """u(q,p,t) with same shape as q/p."""

    def __init__(self, width: int = 64, depth: int = 4, tdim: int = 128):
        super().__init__()
        self.t_emb = TimeEmbed(tdim)
        self.in_conv = nn.Conv3d(2, width, 3, padding=1)
        self.blocks = nn.ModuleList([FiLM3DBlock(width, tdim) for _ in range(depth)])
        self.out = nn.Conv3d(width, 1, 1)

    def forward(self, q: torch.Tensor, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([q, p], dim=1)
        te = self.t_emb(t)
        h = self.in_conv(x)
        for blk in self.blocks:
            h = blk(h, te)
        return self.out(h)


@dataclass
class HyperDynOut:
    dq: torch.Tensor
    dp: torch.Tensor


class HyperHamiltonianDynamics3D(nn.Module):
    def __init__(self, h_net: HyperHamiltonianNet3D, u_net: HyperControlNet3D | None = None):
        super().__init__()
        self.h_net = h_net
        self.u_net = u_net

    def forward(self, q: torch.Tensor, p: torch.Tensor, t: torch.Tensor) -> HyperDynOut:
        with torch.enable_grad():
            q_req = q.detach().requires_grad_(True)
            p_req = p.detach().requires_grad_(True)

            h = self.h_net(q_req, p_req, t).sum()
            grad_q, grad_p = torch.autograd.grad(h, [q_req, p_req], create_graph=self.training)

            dq = grad_p
            dp = -grad_q
            if self.u_net is not None:
                dp = dp + self.u_net(q_req, p_req, t)
        return HyperDynOut(dq=dq, dp=dp)


class HyperSymplecticEuler3D(nn.Module):
    def __init__(self, dynamics: HyperHamiltonianDynamics3D, t0: float = 0.0, t1: float = 1.0, steps: int = 16):
        super().__init__()
        self.dynamics = dynamics
        self.t0 = t0
        self.t1 = t1
        self.steps = steps

    def integrate(self, q0: torch.Tensor, p0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dt = (self.t1 - self.t0) / self.steps
        q, p = q0, p0
        for k in range(self.steps):
            t = torch.full((q.size(0),), self.t0 + k * dt, device=q.device, dtype=q.dtype)
            out1 = self.dynamics(q, p, t)
            p = p + dt * out1.dp

            out2 = self.dynamics(q, p, t)
            q = q + dt * out2.dq
        return q, p


class HyperHamiltonianGenerator3D(nn.Module):
    def __init__(self, width: int = 64, depth: int = 6, steps: int = 16):
        super().__init__()
        self.h_net = HyperHamiltonianNet3D(width=width, depth=depth)
        self.u_net = HyperControlNet3D(width=width, depth=max(2, depth // 2))
        self.dynamics = HyperHamiltonianDynamics3D(self.h_net, self.u_net)
        self.integrator = HyperSymplecticEuler3D(self.dynamics, steps=steps)

    def sample_prior(self, batch: int, c: int, h: int, w: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        q0 = torch.randn(batch, 1, c, h, w, device=device)
        p0 = torch.randn(batch, 1, c, h, w, device=device)
        return q0, p0

    def transport(self, q0: torch.Tensor, p0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.integrator.integrate(q0, p0)

    @torch.no_grad()
    def generate(self, batch: int, c: int, h: int, w: int, device: torch.device | str) -> torch.Tensor:
        q0, p0 = self.sample_prior(batch, c, h, w, device)
        qT, _ = self.transport(q0, p0)
        return qT[:, 0]


def projection_mmd(
    x: torch.Tensor,
    y: torch.Tensor,
    proj_dim: int = 512,
    sigma: float | None = None,
    unbiased: bool = True,
) -> torch.Tensor:
    # x,y: [B,C,H,W]
    xf = x.flatten(1)
    yf = y.flatten(1)

    d = xf.shape[1]
    r = torch.randn(d, proj_dim, device=x.device, dtype=x.dtype) / (d**0.5)
    xp = xf @ r
    yp = yf @ r

    def pdist2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a2 = (a * a).sum(1, keepdim=True)
        b2 = (b * b).sum(1, keepdim=True).T
        return (a2 + b2 - 2 * (a @ b.T)).clamp_min(0.0)

    d_xx = pdist2(xp, xp)
    d_yy = pdist2(yp, yp)
    d_xy = pdist2(xp, yp)

    if sigma is None:
        with torch.no_grad():
            mix = torch.cat([d_xx.flatten(), d_yy.flatten(), d_xy.flatten()])
            mix = mix[mix > 0]
            if mix.numel() == 0:
                sigma = 1.0
            else:
                sigma = torch.sqrt(torch.median(mix)).item()
                sigma = max(sigma, 1e-4)

    def mean_k(dmat: torch.Tensor, drop_diag: bool) -> torch.Tensor:
        k = torch.exp(-dmat / (2 * float(sigma) * float(sigma)))
        if drop_diag:
            n = k.shape[0]
            k = k - torch.diag(torch.diag(k))
            return k.sum() / max(n * (n - 1), 1)
        return k.mean()

    k_xx = mean_k(d_xx, drop_diag=unbiased)
    k_yy = mean_k(d_yy, drop_diag=unbiased)
    k_xy = mean_k(d_xy, drop_diag=False)
    return k_xx + k_yy - 2 * k_xy
