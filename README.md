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


## 8. 常见报错说明

1. **下载 DIV2K 时卡住/中断（KeyboardInterrupt）**  
   现在下载逻辑已改为分块流式写盘（不会一次性把大文件读入内存）。如果中断，重新运行会继续从头安全下载。

2. **生成时报错 `does not require grad`**  
   哈密顿动力学在推理时也需要对状态求梯度。现在代码已在动力学模块内部强制启用状态梯度，训练和无条件生成都可正常运行。


## 9. 256x256 的训练预处理策略（resize / crop）

`train_universal_hamiltonian.py` 现在支持两种方式：
- `--preprocess resize --resize 256x256`：先缩放到 256x256；
- `--preprocess crop --resize 256x256`：从图像随机裁 256x256（如果原图太小会先上采样再裁剪）。

例如：
```bash
PREPROCESS=resize bash scripts/train_color_div2k.sh
PREPROCESS=crop   bash scripts/train_color_div2k.sh
```

## 10. 生成质量排查建议

如果你看到 MMD 长时间卡在常数（比如 `0.0625`），当前代码会：
- 使用自适应带宽 MMD（`--mmd-sigma 0`）；
- 额外引入 Sliced Wasserstein 距离项（`--lambda-swd`）。

这比单一 MMD 在高维特征上更稳定。


## 附录：方程 ↔ 代码类 ↔ 训练日志指标 一页图

见：`docs/equation_code_metrics_diagram.md`


## 11. 纯哈密顿版本（不引入控制项/感知/GAN）

如果你要严格的“纯哈密顿”训练（`u=0`，只学习哈密顿函数 `H`），使用：

```bash
bash scripts/train_color_div2k_pure_hamiltonian.sh
```

对应脚本：`src/train_universal_pure_hamiltonian.py`，核心目标：
- `recon`（仅用于构建 latent 空间）
- `mmd + lambda_h * ||theta_H||^2`

不包含：
- 控制项 `u(q,p,t)`
- 感知损失 / 对抗损失
- SWD 项

### 和主流生成模型的区别（简要）

- Diffusion/Score 模型：依赖噪声预测与逐步去噪，不显式使用辛结构。
- GAN：对抗博弈得到高视觉质量，但理论稳定性依赖训练技巧。
- 本纯哈密顿版本：
  - 生成路径由哈密顿方程给出（可解释动力学）
  - 数值积分采用辛方法（结构保持）
  - 目标以分布匹配（MMD）与哈密顿参数正则为主
