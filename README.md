# Hamiltonian Generative Model（理论力学视角）

本项目包含两条路线：

1. **Latent 路线（通用图像/高光谱）**：`src/train_universal_hamiltonian.py`
2. **HySpecNet-11k 3D 路线（重点）**：`src/train_hyspecnet3d_hamiltonian.py`

你关心的目标是：
- 训练集可固定为 `128x128x224`；
- 生成时可以是**任意**空间分辨率与光谱分辨率（不仅是示例里的 `256x256x191/512x512x31`）。

已通过 **3D 全卷积哈密顿网络**实现：网络在 `(C,H,W)` 三维网格上做动力学演化，输入噪声尺寸就是输出尺寸。

---

## 1. HySpecNet-11k 3D 训练

```bash
bash scripts/train_hyperspectral_hyspecnet11k_3d.sh
```

该脚本会调用：
- `--auto-download --num-shards 1`（从 HuggingFace 下载 HySpecNet-11k 分片）
- `--train-h 128 --train-w 128 --train-c 224`

> 你可把 `--num-shards` 调大到 10 覆盖更多数据。

---

## 2. 任意分辨率无条件生成（通用脚本）

```bash
bash scripts/generate_hyper_anyshape.sh 191x256x256 outputs/hyper_191x256x256 8
bash scripts/generate_hyper_anyshape.sh 31x512x512 outputs/hyper_31x512x512 4
bash scripts/generate_hyper_anyshape.sh 224x1024x1024 outputs/hyper_224x1024x1024 1
```

含义：
- `191x256x256` 是 **C x H x W** 噪声尺寸；
- 输出样本也是同样的 `C x H x W`。

> 也就是说你输入任意 `CxHxW`，模型就按这个光谱/空间分辨率生成。

---

## 3. 示例脚本（保留）

```bash
bash scripts/generate_hyper_anyshape_256x256x191.sh
bash scripts/generate_hyper_anyshape_512x512x31.sh
```

---

## 4. 核心文件

- `src/hyper3d_hamiltonian.py`：3D 哈密顿生成器（任意 C/H/W）
- `src/train_hyspecnet3d_hamiltonian.py`：HySpecNet-11k 训练入口
- `src/generate_hyper_anyshape.py`：任意谱-空分辨率无条件生成（支持 `--noise-shape CxHxW`）

---

## 5. 依赖

```bash
pip install torch torchvision pillow scipy
```


## 6. 训练日志里 MMD 卡在常数怎么办？

如果你看到类似 `mmd=0.0625` 长时间不变，这通常是 RBF 带宽不合适导致核退化。
当前代码默认 `--mmd-sigma 0`，会自动启用 median-bandwidth heuristic（自适应带宽）并使用无偏估计，避免批大小相关的常数下限。


## 7. Linux/screen 下完整保存训练日志与报错日志

训练脚本已默认把 stdout/stderr 分开保存到 `logs/`：
- `*.train.log`：训练过程日志（epoch/loss）
- `*.error.log`：报错堆栈（traceback）

例如：
```bash
bash scripts/train_hyperspectral_hyspecnet11k_3d.sh
```

实时看日志：
```bash
tail -f logs/*_hyspecnet3d.train.log
tail -f logs/*_hyspecnet3d.error.log
```

如果你想换日志目录：
```bash
LOG_DIR=/path/to/your_logs bash scripts/train_hyperspectral_hyspecnet11k_3d.sh
```
