from __future__ import annotations

import re
import tarfile
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision import datasets, transforms

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ARRAY_EXTS = {".npy", ".npz"}

COLOR_DATASETS = {
    "div2k": {
        "url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
        "extract_subdir": "DIV2K_train_HR",
    }
}

HYPERSPECTRAL_DATASETS = {
    "icvl_31": {
        "index_url": "https://cndaqiang.github.io/ICVL-Natural-Hyperspectral-Image-Database/",
        "cube_key": None,  # auto infer variable in each .mat
    },
    "pavia_u": {
        "cube_url": "https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
        "cube_key": "paviaU",
    },
}


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


def auto_download_cifar10(data_root: str, train: bool = True):
    tfm = transforms.Compose([transforms.ToTensor()])
    return datasets.CIFAR10(root=data_root, train=train, transform=tfm, download=True)


def auto_download_div2k(data_root: str) -> Path:
    cfg = COLOR_DATASETS["div2k"]
    root = Path(data_root) / "div2k"
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / Path(urlparse(cfg["url"]).path).name

    if not zip_path.exists():
        download_file(cfg["url"], zip_path)

    extract_root = root / "extracted"
    extract_subdir = extract_root / cfg["extract_subdir"]
    if not extract_subdir.exists():
        extract_archive(zip_path, extract_root)

    return extract_subdir


def auto_download_hyperspectral_pavia(data_root: str) -> tuple[Path, str]:
    cfg = HYPERSPECTRAL_DATASETS["pavia_u"]
    root = Path(data_root) / "pavia_u"
    root.mkdir(parents=True, exist_ok=True)

    cube_path = root / Path(urlparse(cfg["cube_url"]).path).name
    if not cube_path.exists():
        download_file(cfg["cube_url"], cube_path)
    return cube_path, str(cfg["cube_key"])


def auto_download_hyperspectral_icvl(data_root: str, max_files: int = 120) -> tuple[list[Path], Optional[str]]:
    cfg = HYPERSPECTRAL_DATASETS["icvl_31"]
    root = Path(data_root) / "icvl_31"
    root.mkdir(parents=True, exist_ok=True)

    with urlopen(cfg["index_url"]) as resp:
        html = resp.read().decode("utf-8", errors="ignore")

    rel_links = sorted(set(re.findall(r'href=["\']([^"\']+\.mat)["\']', html, flags=re.IGNORECASE)))
    if not rel_links:
        raise RuntimeError("No .mat links found from ICVL index page. You can fallback to --dataset-type pavia_u.")

    links = [urljoin(cfg["index_url"], rel) for rel in rel_links][:max_files]

    mat_paths: list[Path] = []
    for link in links:
        fp = root / Path(urlparse(link).path).name
        if not fp.exists():
            download_file(link, fp)
        mat_paths.append(fp)

    return mat_paths, cfg["cube_key"]


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
        if arr.shape[0] <= 64 and arr.shape[1] > 64 and arr.shape[2] > 64:
            pass
        else:
            arr = arr.transpose(2, 0, 1)
    else:
        raise ValueError(f"Unsupported array rank: {arr.ndim}")

    arr = arr.astype(np.float32)
    maxv = float(arr.max()) if arr.size > 0 else 1.0
    if maxv > 1.0:
        arr = arr / maxv
    return torch.from_numpy(arr)


def _infer_cube_key(mat_dict: dict, preferred: str | None = None) -> str:
    if preferred and preferred in mat_dict:
        return preferred
    for k, v in mat_dict.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.ndim == 3:
            return k
    raise RuntimeError("No 3D cube variable found in MAT file.")


class UniversalImageSpectralDataset(Dataset):
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
            return int(arr.shape[0] if arr.shape[0] <= 64 and arr.shape[1] > 64 else arr.shape[-1])
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


class CIFAR10TensorDataset(Dataset):
    def __init__(self, data_root: str, train: bool = True):
        self.ds = auto_download_cifar10(data_root=data_root, train=train)
        self.channels = 3

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> torch.Tensor:
        x, _ = self.ds[index]
        return x


class FolderImageDataset(Dataset):
    """Image-only folder dataset (used for HR RGB benchmarks like DIV2K)."""

    def __init__(self, root: str, resize_to: Optional[Tuple[int, int]] = None):
        self.root = Path(root)
        self.files = sorted([p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
        if not self.files:
            raise RuntimeError(f"No image files found in {self.root}")
        self.channels = 3
        self.resize_to = resize_to

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with Image.open(self.files[idx]) as img:
            img = img.convert("RGB")
            arr = np.array(img)
        x = _to_chw_tensor(arr)
        if self.resize_to is not None:
            x = torch.nn.functional.interpolate(x[None], size=self.resize_to, mode="bilinear", align_corners=False)[0]
        return x


class HyperspectralMatDataset(Dataset):
    """Dataset from one or multiple hyperspectral cubes, sampled as strided patches."""

    def __init__(
        self,
        cube_paths: list[str],
        cube_key: str | None = None,
        patch_size: int = 64,
        stride: int = 32,
        max_patches_per_cube: int | None = 2048,
    ):
        self.patch_size = patch_size
        self.samples: list[torch.Tensor] = []

        ref_channels: int | None = None

        for cp in cube_paths:
            mat = loadmat(cp)
            key = _infer_cube_key(mat, cube_key)
            cube = mat[key].astype(np.float32)
            cube = cube / max(float(cube.max()), 1e-6)
            cube_t = torch.from_numpy(cube.transpose(2, 0, 1))

            if ref_channels is None:
                ref_channels = cube_t.shape[0]
            if cube_t.shape[0] != ref_channels:
                continue

            _, h, w = cube_t.shape
            coords = []
            for y in range(0, max(1, h - patch_size + 1), stride):
                for x in range(0, max(1, w - patch_size + 1), stride):
                    coords.append((y, x))
            if max_patches_per_cube is not None:
                coords = coords[:max_patches_per_cube]

            for y, x in coords:
                y2 = min(y + patch_size, h)
                x2 = min(x + patch_size, w)
                patch = cube_t[:, y:y2, x:x2]
                if patch.shape[-2:] != (patch_size, patch_size):
                    patch = torch.nn.functional.interpolate(
                        patch[None, ...], size=(patch_size, patch_size), mode="bilinear", align_corners=False
                    )[0]
                self.samples.append(patch)

        if not self.samples:
            raise RuntimeError("No hyperspectral patches built. Check dataset download and cube key.")

        self.channels = ref_channels if ref_channels is not None else self.samples[0].shape[0]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.samples[index]
