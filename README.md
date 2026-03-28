# Hamiltonian Generative Model（理论力学视角）

这个仓库给出一个**从高斯分布到图像分布**的哈密顿生成模型原型：

- 起点：$(q_0,p_0) \sim \mathcal N(0, I)$（可看作“相空间初态”）
- 终点：$q_T \sim p_{\text{data}}$（图像或图像潜变量分布）
- 演化：通过可学习哈密顿量 $H_\theta(q,p,t)$ 与控制项 $u_\phi(q,p,t)$，在相空间中完成动力学运输。

---

## 1. 可行性分析

## 1.1 为什么“哈密顿方法 + 生成模型”可行

1. **概率分布运输本质上是动力系统问题**  
   生成模型可以理解为把简单先验分布通过可逆/近可逆流映射到复杂数据分布。哈密顿系统天然提供了连续时间流。

2. **哈密顿流具备结构保持性**  
   纯哈密顿系统满足辛结构保持（体积保持），长时间积分更稳定，适合构造深层连续流。

3. **图像高维性可通过潜空间缓解**  
   直接在像素空间做哈密顿流代价高；将图像编码到潜变量后，在潜空间做哈密顿演化更可行。

4. **可加入控制项突破“纯保体积”限制**  
   纯哈密顿流保辛体积，表达能力受限；引入可学习控制/耗散项后，可覆盖更广泛的密度变换，同时保留力学可解释性。

## 1.2 关键挑战

- 高维图像分布的训练稳定性（需配合编码器、正则、梯度裁剪）
- 终端分布匹配（MMD、对抗损失、切片Wasserstein等）
- 数值积分误差与计算成本（选择辛积分器、步数调度）

---

## 2. 数学模型

令状态 $x=(q,p)\in\mathbb R^{2d}$，其中 $q$ 是“位置”（用于生成样本），$p$ 是“动量”。

### 2.1 动力学方程

\[
\dot q = \nabla_p H_\theta(q,p,t), \qquad
\dot p = -\nabla_q H_\theta(q,p,t) + u_\phi(q,p,t).
\]

- 当 $u_\phi\equiv 0$ 时是标准哈密顿系统；
- $u_\phi$ 是可学习“控制力/耗散力”，用于提高密度拟合能力。

### 2.2 连续分布演化

概率密度 $\rho_t(x)$ 满足连续性方程：

\[
\partial_t \rho_t + \nabla\cdot(\rho_t f_{\theta,\phi}) = 0,
\]

其中
\[
f_{\theta,\phi}(x,t)=
\begin{bmatrix}
\nabla_p H_\theta \\
-\nabla_q H_\theta + u_\phi
\end{bmatrix}.
\]

若 $u_\phi=0$，则 $\nabla\cdot f_{\theta,0}=0$（李乌维尔定理），流体积保持。

### 2.3 训练目标（终端分布匹配 + 正则）

我们采样
\[
x_0=(q_0,p_0)\sim\mathcal N(0,I),\quad x_T=\Phi_{0\to T}^{\theta,\phi}(x_0),
\]
并最小化
\[
\mathcal L(\theta,\phi)=D\big((q_T)_\#\mathcal N(0,I),\ p_{\text{data}}\big)
+\lambda_H\mathcal R_H+\lambda_u\mathcal R_u.
\]

- $D$ 可以取 MMD/Wasserstein/对抗损失；
- $\mathcal R_H$ 约束哈密顿量平滑与能量规模；
- $\mathcal R_u$ 约束控制项强度（避免数值发散）。

### 2.4 图像建模

给定编码器 $E$ 与解码器 $G$：

\[
z=E(y),\quad z\sim p_z,\quad q_T\approx z,\quad \hat y=G(q_T).
\]

即在潜空间拟合 $p_z$，生成时先采样高斯再积分得到 $q_T$，最后解码为图像。

---

## 3. 代码结构

- `src/hamiltonian_gen_model.py`：核心动力学、辛积分器、模型定义。
- `src/train_mnist_hamiltonian.py`：MNIST 上的最小训练原型（含编码器/解码器、MMD 训练）。

> 说明：该实现强调“理论-代码对齐”的研究原型性质，便于你继续扩展到更强的图像模型（如 VAE latent、DiT latent、Rectified Flow 路径监督等）。

---

## 4. 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision
python src/train_mnist_hamiltonian.py --epochs 1 --batch-size 64
```

---

## 5. 可扩展方向

1. 把 MMD 换成 Sinkhorn-Wasserstein 或对抗判别器。
2. 将 `u_\phi` 改成显式受控最优控制项，加入控制能量积分惩罚。
3. 使用更高阶辛积分器（Leapfrog/Stormer-Verlet 变体）。
4. 在 latent diffusion 的 VAE latent 空间上做哈密顿流，兼顾质量与可解释性。
