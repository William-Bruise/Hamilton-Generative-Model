from __future__ import annotations

import io
import os
import tarfile
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ARRAY_EXTS = {".npy", ".npz"}


def download_file(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as resp:
        data = resp.read()
    out_path.write_bytes(data)
    return out_path


def extract_archive(archive_path: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(archive_path.suffixes[-2:])

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
    elif archive_path.suffix in {".tar", ".gz", ".tgz"} or suffixes in {".tar.gz", ".tar.bz2"}:
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    return dest_dir


def maybe_download_dataset(url: Optional[str], data_root: str, extract: bool = True) -> Path:
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    if not url:
        return root

    parsed = urlparse(url)
    filename = Path(parsed.path).name or "dataset.bin"
    archive_path = root / filename

    if not archive_path.exists():
        download_file(url, archive_path)

    if extract and (archive_path.suffix in {".zip", ".tar", ".gz", ".tgz"} or "".join(archive_path.suffixes[-2:]) in {".tar.gz", ".tar.bz2"}):
        extracted_dir = root / "extracted"
        if not extracted_dir.exists() or not any(extracted_dir.iterdir()):
            extract_archive(archive_path, extracted_dir)
        return extracted_dir

    return root


def _discover_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in IMAGE_EXTS or p.suffix.lower() in ARRAY_EXTS:
            files.append(p)
    return sorted(files)


def _to_chw_tensor(arr: np.ndarray) -> torch.Tensor:
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3:
        # handle HWC / CHW
        if arr.shape[0] <= 8 and arr.shape[1] > 8 and arr.shape[2] > 8:
            pass
        else:
            arr = arr.transpose(2, 0, 1)
    else:
        raise ValueError(f"Unsupported array rank: {arr.ndim}")

    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return torch.from_numpy(arr)


class UniversalImageSpectralDataset(Dataset):
    """Load standard images and hyperspectral arrays (.npy/.npz) with arbitrary channels/resolution."""

    def __init__(self, root: str, resize_to: Optional[Tuple[int, int]] = None):
        self.root = Path(root)
        self.files = _discover_files(self.root)
        if not self.files:
            raise RuntimeError(f"No supported files found in {self.root}")

        self.resize_to = resize_to
        self._channels = self._infer_channels(self.files[0])

    @staticmethod
    def _infer_channels(path: Path) -> int:
        if path.suffix.lower() in IMAGE_EXTS:
            with Image.open(path) as img:
                if img.mode == "L":
                    return 1
                if img.mode in {"RGB", "YCbCr", "HSV"}:
                    return 3
                if img.mode == "RGBA":
                    return 4
                arr = np.array(img)
                return 1 if arr.ndim == 2 else int(arr.shape[-1])

        obj = np.load(path)
        if isinstance(obj, np.lib.npyio.NpzFile):
            key = list(obj.keys())[0]
            arr = obj[key]
        else:
            arr = obj
        if arr.ndim == 2:
            return 1
        if arr.ndim == 3:
            return int(arr.shape[0] if arr.shape[0] <= 8 and arr.shape[1] > 8 else arr.shape[-1])
        raise ValueError(f"Unsupported sample shape in {path}: {arr.shape}")

    @property
    def channels(self) -> int:
        return self._channels

    def __len__(self) -> int:
        return len(self.files)

    def _load_sample(self, path: Path) -> torch.Tensor:
        if path.suffix.lower() in IMAGE_EXTS:
            with Image.open(path) as img:
                arr = np.array(img)
            x = _to_chw_tensor(arr)
        else:
            obj = np.load(path)
            if isinstance(obj, np.lib.npyio.NpzFile):
                key = list(obj.keys())[0]
                arr = obj[key]
            else:
                arr = obj
            x = _to_chw_tensor(arr)

        if x.shape[0] != self._channels:
            raise RuntimeError(
                f"Inconsistent channel count. Expected {self._channels}, got {x.shape[0]} in {path}. "
                "Please keep channel count consistent in one training run."
            )

        if self.resize_to is not None:
            x = torch.nn.functional.interpolate(
                x[None, ...],
                size=self.resize_to,
                mode="bilinear",
                align_corners=False,
            )[0]

        return x

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._load_sample(self.files[index])
