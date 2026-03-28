# Hamiltonian Generative Model（理论力学视角）

本项目包含两条路线：

1. **Latent 路线（通用图像/高光谱）**：`src/train_universal_hamiltonian.py`
2. **HySpecNet-11k 3D 路线（重点）**：`src/train_hyspecnet3d_hamiltonian.py`

你提出的关键需求：
- 训练集是 `128x128x224`；
- 生成时可改成 `256x256x191` / `512x512x31`（任意空间与光谱分辨率）；

已通过 **3D 全卷积哈密顿网络**实现：网络在 `(C,H,W)` 三维网格上做动力学演化，输入噪声尺寸就是输出尺寸。

---

## 1. HySpecNet-11k 3D 训练

```bash
bash scripts/train_hyperspectral_hyspecnet11k_3d.sh
```

该脚本会调用：
- `--auto-download --num-shards 1`（从 HuggingFace 下载 HySpecNet-11k 的分片）
- `--train-h 128 --train-w 128 --train-c 224`

> 你可把 `--num-shards` 调大到 10 以覆盖更多数据。

---

## 2. 任意分辨率无条件生成（只需改噪声尺寸）

### 2.1 生成 `256x256x191`

```bash
bash scripts/generate_hyper_anyshape_256x256x191.sh
```

### 2.2 生成 `512x512x31`

```bash
bash scripts/generate_hyper_anyshape_512x512x31.sh
```

输出文件均为：`*.npy`，shape 为 `[C,H,W]`。

---

## 3. 核心文件

- `src/hyper3d_hamiltonian.py`：3D 哈密顿生成器（任意 C/H/W）
- `src/train_hyspecnet3d_hamiltonian.py`：HySpecNet-11k 训练入口
- `src/generate_hyper_anyshape.py`：任意谱-空分辨率无条件生成

---

## 4. 依赖

```bash
pip install torch torchvision pillow scipy
```
