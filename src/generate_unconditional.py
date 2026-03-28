from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from hamiltonian_gen_model import HamiltonianGenerativeModel


class FlexibleDecoder(nn.Module):
    def __init__(self, out_channels: int, latent_dim: int = 64, base: int = 64):
        super().__init__()
        self.base = base
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.SiLU(),
            nn.Linear(512, base * 4 * 4 * 4),
            nn.SiLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1),
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.ConvTranspose2d(base, out_channels, 4, 2, 1),
        )

    def forward(self, z: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        h = self.fc(z).view(-1, self.base * 4, 4, 4)
        x = self.deconv(h)
        if x.shape[-2:] != out_hw:
            x = torch.nn.functional.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        return torch.sigmoid(x)


def save_rgb_batch(x: torch.Tensor, out_dir: Path, prefix: str = "sample") -> None:
    x = x.clamp(0, 1).cpu()
    for i in range(x.shape[0]):
        arr = (x[i].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        Image.fromarray(arr).save(out_dir / f"{prefix}_{i:05d}.png")


def save_hyper_batch(x: torch.Tensor, out_dir: Path, prefix: str = "sample") -> None:
    x = x.clamp(0, 1).cpu().numpy()  # N,C,H,W
    for i in range(x.shape[0]):
        np.save(out_dir / f"{prefix}_{i:05d}.npy", x[i])


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("args", {})
    channels = int(ckpt["channels"])

    latent_dim = int(cfg.get("latent_dim", args.latent_dim))
    base_channels = int(cfg.get("base_channels", args.base_channels))
    width = int(cfg.get("width", args.width))
    depth = int(cfg.get("depth", args.depth))
    steps = int(cfg.get("steps", args.steps))

    flow = HamiltonianGenerativeModel(dim=latent_dim, width=width, depth=depth, steps=steps).to(device)
    decoder = FlexibleDecoder(out_channels=channels, latent_dim=latent_dim, base=base_channels).to(device)

    flow.load_state_dict(ckpt["flow"])
    decoder.load_state_dict(ckpt["decoder"])

    flow.eval()
    decoder.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        z = flow.generate_latent(n=args.num_samples, device=device)
        x = decoder(z, out_hw=(args.height, args.width_out))

    if args.mode == "rgb":
        save_rgb_batch(x, out_dir)
        print(f"Saved {args.num_samples} RGB samples to {out_dir}")
    else:
        save_hyper_batch(x, out_dir)
        print(f"Saved {args.num_samples} hyperspectral samples (.npy) to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unconditional generation from trained Hamiltonian model checkpoint")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["rgb", "hyper"], required=True)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width-out", type=int, default=256)

    # fallback hyperparams if not in ckpt args
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--steps", type=int, default=32)

    main(parser.parse_args())
