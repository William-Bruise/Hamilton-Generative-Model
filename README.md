# Hamiltonian Generative Model（理论力学视角）

本项目把“高斯先验 -> 图像分布”建模为受控哈密顿动力系统：

\[
\dot q = \nabla_p H_\theta(q,p,t),\qquad
\dot p = -\nabla_q H_\theta(q,p,t) + u_\phi(q,p,t).
\]

---

## 1. 高分辨率数据集支持（已内置自动下载）

你提出 CIFAR10 分辨率太低、PaviaU 样本太少——当前代码已新增更合适的默认选项：

- **彩色图像（>=256）**：`DIV2K`（官方 HR 训练集，原始分辨率远高于 256）
- **高光谱图像（>=256）**：`ICVL-31`（常见自然场景高光谱数据，典型空间分辨率 > 256）
- 兼容旧选项：`cifar10`、`pavia_u`、`generic`

---

## 2. 训练命令

### 2.1 训练高分辨率彩色图像（DIV2K）

```bash
python src/train_universal_hamiltonian.py \
  --dataset-type div2k \
  --data-root ./data \
  --resize 256x256 \
  --epochs 20
```

### 2.2 训练高分辨率高光谱图像（ICVL-31）

```bash
python src/train_universal_hamiltonian.py \
  --dataset-type icvl_31 \
  --data-root ./data \
  --hyper-patch 64 \
  --hyper-stride 32 \
  --icvl-max-files 120 \
  --epochs 20
```

### 2.3 兼容原来的 PaviaU（单场景）

```bash
python src/train_universal_hamiltonian.py \
  --dataset-type pavia_u \
  --data-root ./data \
  --hyper-patch 64 \
  --hyper-stride 32
```

---

## 3. 网络结构（已非简单 MLP）

`HamiltonianNet` 与 `ControlNet` 使用 **Time-Conditioned ResNet + FiLM**：
- 正弦时间嵌入（Sinusoidal time embedding）
- 残差块（LayerNorm + FiLM 调制）
- 更适合复杂非线性动力学建模

---

## 4. 依赖

```bash
pip install torch torchvision pillow scipy
```
