# Hamiltonian Generative Model（理论力学视角）

这个仓库给出一个**从高斯分布到图像分布**的哈密顿生成模型原型：

- 起点：$(q_0,p_0) \sim \mathcal N(0, I)$（可看作“相空间初态”）
- 终点：$q_T \sim p_{\text{data}}$（图像或图像潜变量分布）
- 演化：通过可学习哈密顿量 $H_\theta(q,p,t)$ 与控制项 $u_\phi(q,p,t)$，在相空间中完成动力学运输。

---

## 1. 可行性与原理

1. **概率分布运输本质上是动力系统问题**：生成模型可看作将简单先验映射到复杂数据分布。
2. **哈密顿流具备结构保持性**：有利于长时间稳定积分。
3. **潜空间训练降低图像高维难度**：先编码，再在 latent 中做哈密顿流。
4. **控制项提升表达力**：纯哈密顿流体积保持，加入 $u_\phi$ 后可逼近更一般密度变换。

关键挑战：
- 高维数据训练稳定性；
- 终端分布匹配（MMD/Wasserstein/对抗）；
- 数值积分误差控制。

---

## 2. 数学模型

令状态 $x=(q,p)\in\mathbb R^{2d}$，其中 $q$ 是“位置”（用于生成样本），$p$ 是“动量”。

### 2.1 动力学方程

\[
\dot q = \nabla_p H_\theta(q,p,t), \qquad
\dot p = -\nabla_q H_\theta(q,p,t) + u_\phi(q,p,t).
\]

### 2.2 概率密度演化

\[
\partial_t \rho_t + \nabla\cdot(\rho_t f_{\theta,\phi}) = 0,
\quad
f_{\theta,\phi}(x,t)=
\begin{bmatrix}
\nabla_p H_\theta \\
-\nabla_q H_\theta + u_\phi
\end{bmatrix}.
\]

若 $u_\phi=0$，则 $\nabla\cdot f_{\theta,0}=0$（李乌维尔定理，体积保持）。

### 2.3 训练目标

\[
x_0\sim\mathcal N(0,I),\quad x_T=\Phi_{0\to T}^{\theta,\phi}(x_0),
\]
\[
\mathcal L(\theta,\phi)=D\big((q_T)_\#\mathcal N(0,I), p_{\text{data}}\big)
+\lambda_H\mathcal R_H+\lambda_u\mathcal R_u.
\]

---

## 3. 代码结构

- `src/hamiltonian_gen_model.py`：核心动力学、辛积分器、MMD。
- `src/data_utils.py`：
  - 数据集下载（URL）
  - zip/tar 自动解压
  - 通用图像/光谱数组读取（支持 jpg/png/tif + npy/npz）
- `src/train_universal_hamiltonian.py`：
  - 支持任意空间分辨率（通过全卷积 + Adaptive Pool + 输出插值）
  - 支持任意光谱通道数（按数据自动推断 `in_channels`）

---

## 4. 训练（支持任意空间和光谱分辨率）

### 4.1 本地数据集

```bash
python src/train_universal_hamiltonian.py \
  --data-root ./your_dataset_root \
  --epochs 20 \
  --batch-size 16
```

### 4.2 自动下载数据集（URL）

```bash
python src/train_universal_hamiltonian.py \
  --dataset-url "https://your-domain.com/dataset.zip" \
  --data-root ./data \
  --epochs 20
```

### 4.3 固定输入大小（可选）

```bash
python src/train_universal_hamiltonian.py \
  --data-root ./your_dataset_root \
  --resize 256x256
```

---

## 5. 关于“任意空间/光谱分辨率”的边界

- **空间分辨率**：训练脚本可直接处理不同 H×W（同一 batch 内建议统一尺寸，或用 `--resize`）。
- **光谱分辨率**：可处理任意通道数 C（如 RGB=3，多光谱=8，高光谱=31+），但**一次训练中应保持通道数一致**。
- 对超高分辨率/超高光谱数据，建议加 patch 训练与混合精度以降低显存压力。
