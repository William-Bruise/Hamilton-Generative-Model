# Hamiltonian Generative Model（理论力学视角）

本项目把“高斯先验 -> 图像分布”建模为受控哈密顿动力系统：

\[
\dot q = \nabla_p H_\theta(q,p,t),\qquad
\dot p = -\nabla_q H_\theta(q,p,t) + u_\phi(q,p,t).
\]

---

## 1. 高分辨率数据集支持（已内置自动下载）

- **彩色图像（>=256）**：`DIV2K`（官方 HR 训练集）
- **高光谱图像（>=256）**：`ICVL-31`（多场景高光谱 `.mat`）
- 兼容旧选项：`cifar10`、`pavia_u`、`generic`

---

## 2. 训练脚本（bash）

### 2.1 训练高分辨率彩色图像

```bash
bash scripts/train_color_div2k.sh
```

### 2.2 训练高分辨率高光谱图像

```bash
bash scripts/train_hyperspectral_icvl.sh
```

---

## 3. 无条件生成脚本（bash）

### 3.1 无条件生成彩色图像（PNG）

```bash
bash scripts/generate_color_unconditional.sh
```

输出目录：`outputs/color_uncond/*.png`

### 3.2 无条件生成高光谱图像（NPY）

```bash
bash scripts/generate_hyperspectral_unconditional.sh
```

输出目录：`outputs/hyper_uncond/*.npy`（shape: `C,H,W`）

---

## 4. 也可直接用 Python 命令

```bash
python src/train_universal_hamiltonian.py --dataset-type div2k --data-root ./data --resize 256x256
python src/train_universal_hamiltonian.py --dataset-type icvl_31 --data-root ./data --hyper-patch 64 --hyper-stride 32

python src/generate_unconditional.py --ckpt checkpoints/div2k_hamiltonian.pt --out-dir outputs/color_uncond --mode rgb --num-samples 16 --height 256 --width-out 256
python src/generate_unconditional.py --ckpt checkpoints/icvl_hamiltonian.pt --out-dir outputs/hyper_uncond --mode hyper --num-samples 16 --height 64 --width-out 64
```

---

## 5. 依赖

```bash
pip install torch torchvision pillow scipy
```
