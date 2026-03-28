from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from hamiltonian_gen_model import HamiltonianGenerativeModel, compute_mmd_rbf


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent_dim, 256), nn.SiLU(), nn.Linear(256, 64 * 7 * 7), nn.SiLU())
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 64, 7, 7)
        return self.deconv(h)


def build_loader(batch_size: int) -> DataLoader:
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST(root="./data", train=True, transform=tfm, download=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = build_loader(args.batch_size)

    encoder = Encoder(latent_dim=args.latent_dim).to(device)
    decoder = Decoder(latent_dim=args.latent_dim).to(device)
    model = HamiltonianGenerativeModel(dim=args.latent_dim, hidden=args.hidden, steps=args.steps).to(device)

    opt_ae = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr_ae)
    opt_flow = optim.Adam(model.parameters(), lr=args.lr_flow)

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        decoder.train()
        model.train()

        total_rec = 0.0
        total_mmd = 0.0

        for x, _ in loader:
            x = x.to(device)

            # ---- Stage A: train latent autoencoder ----
            z = encoder(x)
            x_rec = decoder(z)
            rec_loss = ((x_rec - x) ** 2).mean()

            opt_ae.zero_grad()
            rec_loss.backward()
            opt_ae.step()

            # ---- Stage B: train Hamiltonian flow to match encoded latent distribution ----
            with torch.no_grad():
                z_target = encoder(x).detach()

            q0, p0 = model.sample_prior(n=x.size(0), device=device)
            qT, _ = model.transport(q0, p0)

            mmd = compute_mmd_rbf(qT, z_target, sigma=args.mmd_sigma)

            h_reg = 0.0
            for p in model.h_net.parameters():
                h_reg = h_reg + (p * p).mean()

            u_reg = 0.0
            for p in model.u_net.parameters():
                u_reg = u_reg + (p * p).mean()

            loss_flow = mmd + args.lambda_h * h_reg + args.lambda_u * u_reg

            opt_flow.zero_grad()
            loss_flow.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt_flow.step()

            total_rec += rec_loss.item()
            total_mmd += mmd.item()

        n_batches = len(loader)
        print(
            f"Epoch {epoch:03d} | recon={total_rec / n_batches:.6f} | mmd={total_mmd / n_batches:.6f}",
            flush=True,
        )

    torch.save(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "hamiltonian_flow": model.state_dict(),
            "args": vars(args),
        },
        args.out,
    )
    print(f"Saved checkpoint to: {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--lr-ae", type=float, default=1e-3)
    parser.add_argument("--lr-flow", type=float, default=2e-4)
    parser.add_argument("--mmd-sigma", type=float, default=1.0)
    parser.add_argument("--lambda-h", type=float, default=1e-6)
    parser.add_argument("--lambda-u", type=float, default=1e-6)
    parser.add_argument("--out", type=str, default="hamiltonian_mnist.pt")
    train(parser.parse_args())
