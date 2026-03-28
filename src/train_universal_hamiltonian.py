from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_utils import UniversalImageSpectralDataset, maybe_download_dataset
from hamiltonian_gen_model import HamiltonianGenerativeModel, compute_mmd_rbf


class FlexibleEncoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int = 64, base: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, base),
            nn.SiLU(),
            nn.Conv2d(base, base * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
            nn.Conv2d(base * 2, base * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, base * 4),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base * 4 * 4 * 4, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.pool(h)
        return self.head(h)


class FlexibleDecoder(nn.Module):
    def __init__(self, out_channels: int, latent_dim: int = 64, base: int = 32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.SiLU(),
            nn.Linear(256, base * 4 * 4 * 4),
            nn.SiLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1),
            nn.GroupNorm(4, base),
            nn.SiLU(),
            nn.ConvTranspose2d(base, out_channels, 4, 2, 1),
        )

    def forward(self, z: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        h = self.fc(z).view(-1, self.deconv[0].in_channels, 4, 4)
        x = self.deconv(h)
        if x.shape[-2:] != out_hw:
            x = torch.nn.functional.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        return torch.sigmoid(x)


def parse_resize(resize: str | None) -> Tuple[int, int] | None:
    if not resize:
        return None
    h, w = resize.split("x")
    return int(h), int(w)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = maybe_download_dataset(args.dataset_url, args.data_root, extract=True)
    resize_to = parse_resize(args.resize)
    dataset = UniversalImageSpectralDataset(root=str(data_root), resize_to=resize_to)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    encoder = FlexibleEncoder(in_channels=dataset.channels, latent_dim=args.latent_dim, base=args.base_channels).to(device)
    decoder = FlexibleDecoder(out_channels=dataset.channels, latent_dim=args.latent_dim, base=args.base_channels).to(device)
    flow = HamiltonianGenerativeModel(dim=args.latent_dim, hidden=args.hidden, steps=args.steps).to(device)

    opt_ae = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr_ae)
    opt_flow = optim.AdamW(flow.parameters(), lr=args.lr_flow)

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        decoder.train()
        flow.train()

        rec_meter = 0.0
        mmd_meter = 0.0

        for x in loader:
            x = x.to(device)
            out_hw = (x.shape[-2], x.shape[-1])

            z = encoder(x)
            x_rec = decoder(z, out_hw=out_hw)
            rec_loss = ((x_rec - x) ** 2).mean()

            opt_ae.zero_grad()
            rec_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), max_norm=5.0)
            opt_ae.step()

            with torch.no_grad():
                z_target = encoder(x).detach()

            q0, p0 = flow.sample_prior(x.size(0), device=device)
            qT, _ = flow.transport(q0, p0)
            mmd = compute_mmd_rbf(qT, z_target, sigma=args.mmd_sigma)

            reg_h = sum((p * p).mean() for p in flow.h_net.parameters())
            reg_u = sum((p * p).mean() for p in flow.u_net.parameters())
            flow_loss = mmd + args.lambda_h * reg_h + args.lambda_u * reg_u

            opt_flow.zero_grad()
            flow_loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=5.0)
            opt_flow.step()

            rec_meter += rec_loss.item()
            mmd_meter += mmd.item()

        n = len(loader)
        print(f"Epoch {epoch:03d} | recon={rec_meter / n:.6f} | mmd={mmd_meter / n:.6f}", flush=True)

    ckpt_path = Path(args.out)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "flow": flow.state_dict(),
            "channels": dataset.channels,
            "args": vars(args),
        },
        ckpt_path,
    )
    print(f"Saved checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal-resolution/spectral Hamiltonian generative training")
    parser.add_argument("--dataset-url", type=str, default=None, help="Optional dataset archive URL")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--resize", type=str, default=None, help="Optional fixed size: HxW, e.g. 256x256")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--lr-ae", type=float, default=1e-3)
    parser.add_argument("--lr-flow", type=float, default=2e-4)
    parser.add_argument("--mmd-sigma", type=float, default=1.0)
    parser.add_argument("--lambda-h", type=float, default=1e-6)
    parser.add_argument("--lambda-u", type=float, default=1e-6)
    parser.add_argument("--out", type=str, default="checkpoints/universal_hamiltonian.pt")

    train(parser.parse_args())
