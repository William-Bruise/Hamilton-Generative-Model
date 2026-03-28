from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from hyper3d_hamiltonian import HyperHamiltonianGenerator3D


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("args", {})

    model = HyperHamiltonianGenerator3D(
        width=int(cfg.get("width", args.width)),
        depth=int(cfg.get("depth", args.depth)),
        steps=int(cfg.get("steps", args.steps)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        x = model.generate(batch=args.num_samples, c=args.channels, h=args.height, w=args.width_out, device=device)

    x = x.clamp(0, 1).cpu().numpy()  # B,C,H,W
    for i in range(x.shape[0]):
        np.save(out_dir / f"sample_{i:05d}.npy", x[i])

    print(f"Saved {x.shape[0]} samples to {out_dir}, each with shape [C,H,W]=[{args.channels},{args.height},{args.width_out}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate hyperspectral cubes at arbitrary spatial/spectral resolutions")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--channels", type=int, required=True, help="spectral resolution, e.g. 191 or 31")
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width-out", type=int, required=True)

    # fallback model cfg
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--steps", type=int, default=16)
    main(parser.parse_args())
