from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from hamiltonian_gen_model import HamiltonianGenerativeModel, compute_mmd_rbf
from train_universal_hamiltonian import FlexibleDecoder, FlexibleEncoder, build_dataset


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    encoder = FlexibleEncoder(in_channels=dataset.channels, latent_dim=args.latent_dim, base=args.base_channels).to(device)
    decoder = FlexibleDecoder(out_channels=dataset.channels, latent_dim=args.latent_dim, base=args.base_channels).to(device)

    # Pure Hamiltonian: u(q,p,t) = 0
    flow = HamiltonianGenerativeModel(
        dim=args.latent_dim,
        width=args.width,
        depth=args.depth,
        steps=args.steps,
        use_control=False,
    ).to(device)

    opt_ae = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr_ae)
    opt_flow = optim.AdamW(flow.parameters(), lr=args.lr_flow)

    for epoch in range(1, args.epochs + 1):
        rec_meter = 0.0
        mmd_meter = 0.0

        for x in loader:
            x = x.to(device)
            out_hw = (x.shape[-2], x.shape[-1])

            # Autoencoder stage (for latent space construction)
            z = encoder(x)
            x_rec = decoder(z, out_hw=out_hw)
            rec_loss = ((x_rec - x) ** 2).mean()
            opt_ae.zero_grad()
            rec_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), max_norm=5.0)
            opt_ae.step()

            # Pure Hamiltonian transport stage
            with torch.no_grad():
                z_target = encoder(x).detach()

            q0, p0 = flow.sample_prior(x.size(0), device=device)
            qT, _ = flow.transport(q0, p0)
            sigma = None if args.mmd_sigma <= 0 else args.mmd_sigma
            mmd = compute_mmd_rbf(qT, z_target, sigma=sigma)

            reg_h = sum((p * p).mean() for p in flow.h_net.parameters())
            flow_loss = mmd + args.lambda_h * reg_h

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
            "dataset_type": args.dataset_type,
            "pure_hamiltonian": True,
            "args": vars(args),
        },
        ckpt_path,
    )
    print(f"Saved checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal training with pure Hamiltonian dynamics (u=0)")
    parser.add_argument("--dataset-type", type=str, default="div2k", choices=["generic", "cifar10", "div2k", "pavia_u", "icvl_31"])
    parser.add_argument("--dataset-url", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--resize", type=str, default="256x256")
    parser.add_argument("--preprocess", type=str, default="resize", choices=["resize", "crop"])
    parser.add_argument("--hyper-patch", type=int, default=64)
    parser.add_argument("--hyper-stride", type=int, default=32)
    parser.add_argument("--hyper-max-patches", type=int, default=2048)
    parser.add_argument("--icvl-max-files", type=int, default=120)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--lr-ae", type=float, default=1e-3)
    parser.add_argument("--lr-flow", type=float, default=2e-4)
    parser.add_argument("--mmd-sigma", type=float, default=0.0, help="<=0 uses median-bandwidth heuristic")
    parser.add_argument("--lambda-h", type=float, default=1e-6)
    parser.add_argument("--out", type=str, default="checkpoints/div2k_pure_hamiltonian.pt")

    train(parser.parse_args())
