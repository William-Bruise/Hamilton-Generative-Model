from __future__ import annotations

import argparse
import tarfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset

from hyper3d_hamiltonian import HyperHamiltonianGenerator3D, projection_mmd

HYSPECNET_HF_BASE = "https://huggingface.co/datasets/torchgeo/hyspecnet/resolve/main"


def _download(url: str, path: Path) -> None:
    from urllib.request import urlopen

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with urlopen(url) as resp:
        path.write_bytes(resp.read())


def download_hyspecnet_shards(data_root: str, shards: int = 1) -> Path:
    root = Path(data_root) / "hyspecnet11k"
    root.mkdir(parents=True, exist_ok=True)

    for i in range(1, shards + 1):
        name = f"hyspecnet-11k-{i:02d}.tar.gz"
        url = f"{HYSPECNET_HF_BASE}/{name}"
        tar_path = root / name
        _download(url, tar_path)

        extract_dir = root / name.replace(".tar.gz", "")
        if not extract_dir.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar_path, "r:gz") as tf:
                tf.extractall(extract_dir)

    return root


def _cube_from_file(path: Path) -> Optional[np.ndarray]:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() == ".npz":
        d = np.load(path)
        arr = d[list(d.keys())[0]]
    elif path.suffix.lower() == ".mat":
        d = loadmat(path)
        arr = None
        for k, v in d.items():
            if k.startswith("__"):
                continue
            if isinstance(v, np.ndarray) and v.ndim == 3:
                arr = v
                break
        if arr is None:
            return None
    else:
        return None

    if arr.ndim != 3:
        return None

    # HWC/CHW -> CHW
    if arr.shape[0] <= 512 and arr.shape[1] > 16 and arr.shape[2] > 16:
        # ambiguous; if first dim is spectral keep as CHW
        if arr.shape[0] < arr.shape[-1]:
            chw = arr
        else:
            chw = arr.transpose(2, 0, 1)
    else:
        chw = arr.transpose(2, 0, 1)

    chw = chw.astype(np.float32)
    m = float(chw.max())
    if m > 1.0:
        chw = chw / m
    return chw


class HySpecNetCubeDataset(Dataset):
    def __init__(self, root: str, train_h: int = 128, train_w: int = 128, train_c: int = 224):
        self.root = Path(root)
        self.files = [p for p in self.root.rglob("*") if p.suffix.lower() in {".npy", ".npz", ".mat"}]
        if not self.files:
            raise RuntimeError(f"No cube files found under: {self.root}")
        self.train_h = train_h
        self.train_w = train_w
        self.train_c = train_c

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        arr = _cube_from_file(self.files[idx])
        if arr is None:
            raise RuntimeError(f"Invalid cube format: {self.files[idx]}")
        x = torch.from_numpy(arr)[None, ...]  # 1,C,H,W
        x = torch.nn.functional.interpolate(x, size=(self.train_c, self.train_h, self.train_w), mode="trilinear", align_corners=False)
        return x[0]


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = download_hyspecnet_shards(args.data_root, shards=args.num_shards) if args.auto_download else Path(args.data_root)
    ds = HySpecNetCubeDataset(root=str(root), train_h=args.train_h, train_w=args.train_w, train_c=args.train_c)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    model = HyperHamiltonianGenerator3D(width=args.width, depth=args.depth, steps=args.steps).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        meter = 0.0
        for x in dl:
            x = x.to(device)  # B,C,H,W
            b, c, h, w = x.shape

            q0, p0 = model.sample_prior(b, c, h, w, device=device)
            qT, _ = model.transport(q0, p0)
            pred = qT[:, 0]

            sigma = None if args.mmd_sigma <= 0 else args.mmd_sigma
            mmd = projection_mmd(pred, x, proj_dim=args.proj_dim, sigma=sigma)
            reg = sum((p * p).mean() for p in model.parameters())
            loss = mmd + args.weight_decay_reg * reg

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            meter += loss.item()

        print(f"Epoch {epoch:03d} | loss={meter / len(dl):.6f}", flush=True)

    ckpt = {
        "model": model.state_dict(),
        "args": vars(args),
        "train_shape": [args.train_c, args.train_h, args.train_w],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)
    print(f"Saved checkpoint: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D Hamiltonian generator on HySpecNet-11k")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--auto-download", action="store_true", help="Download HySpecNet-11k shards from HuggingFace")
    parser.add_argument("--num-shards", type=int, default=1, help="How many 6-7GB shards to download (1-10)")
    parser.add_argument("--train-h", type=int, default=128)
    parser.add_argument("--train-w", type=int, default=128)
    parser.add_argument("--train-c", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--proj-dim", type=int, default=512)
    parser.add_argument("--mmd-sigma", type=float, default=0.0, help="<=0 uses median-bandwidth heuristic")
    parser.add_argument("--weight-decay-reg", type=float, default=1e-8)
    parser.add_argument("--out", type=str, default="checkpoints/hyspecnet3d_hamiltonian.pt")
    train(parser.parse_args())
