# Hamiltonian 生成模型一页图：方程 ↔ 代码类 ↔ 训练日志指标

```mermaid
flowchart TB
    subgraph EQ[方程层（数学）]
      E1[状态: x=(q,p)]
      E2[动力学:\nq_dot = ∂H/∂p\np_dot = -∂H/∂q + u]
      E3[流映射: x_T = Φ_{0→T}(x_0)]
      E4[先验: x_0 ~ N(0, I)]
      E5[目标: q_T ~ p_data]
      E6[损失: L = D(q_T, data) + 正则]
      E7[D = MMD + λ_swd · SWD]
    end

    subgraph CODE[代码层（实现）]
      C1[src/hamiltonian_gen_model.py\nHamiltonianNet / ControlNet]
      C2[src/hamiltonian_gen_model.py\nHamiltonianDynamics.forward\n(autograd 求 ∂H/∂q, ∂H/∂p)]
      C3[src/hamiltonian_gen_model.py\nSymplecticEulerIntegrator.integrate]
      C4[src/hamiltonian_gen_model.py\ncompute_mmd_rbf / sliced_wasserstein_distance]
      C5[src/train_universal_hamiltonian.py\nflow_loss = mmd + λ_swd*swd + λh*reg_h + λu*reg_u]
      C6[src/hyper3d_hamiltonian.py\nHyperHamiltonianGenerator3D\n(3D 全卷积, 任意 C/H/W)]
      C7[src/train_hyspecnet3d_hamiltonian.py\nloss = projection_mmd + reg]
    end

    subgraph LOG[训练日志层（可观测指标）]
      L1[recon\n(AE 重建误差)]
      L2[mmd\n(分布匹配误差)]
      L3[swd\n(切片Wasserstein)]
      L4[flow_loss\n(训练哈密顿流总目标)]
      L5[3D loss\n(投影MMD+正则)]
      L6[异常信号:\nmmd长期常数/不下降]
    end

    E2 --> C2
    E3 --> C3
    E4 --> C3
    E5 --> C5
    E6 --> C5
    E7 --> C4

    C1 --> C2
    C2 --> C3
    C4 --> C5
    C6 --> C7

    C5 --> L1
    C5 --> L2
    C5 --> L3
    C5 --> L4
    C7 --> L5

    L2 --> L6
    L3 --> L6

    style EQ fill:#f7fbff,stroke:#4a90e2,stroke-width:1px
    style CODE fill:#f6fff8,stroke:#27ae60,stroke-width:1px
    style LOG fill:#fffaf2,stroke:#f39c12,stroke-width:1px
```

---

## 如何读这张图（简版）

1. **方程层**定义了物理结构：哈密顿动力系统 + 分布匹配目标。  
2. **代码层**把方程逐步落地：网络参数化 `H,u` → autograd 求导 → 辛积分推进。  
3. **日志层**是你训练时能看到的量：`recon/mmd/swd/loss`。  
4. 当 `mmd` 长期卡常数时，要看带宽、SWD项、batch、特征尺度是否合适。

---

## 对应文件索引

- 2D/latent 主体：`src/hamiltonian_gen_model.py`
- 通用训练：`src/train_universal_hamiltonian.py`
- 3D hyperspectral 主体：`src/hyper3d_hamiltonian.py`
- 3D 训练：`src/train_hyspecnet3d_hamiltonian.py`
