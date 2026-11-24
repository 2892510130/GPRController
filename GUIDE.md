# GPRController 使用指南

> 本指南基于 `vibe_code` 分支（P0–P2 修复之后），覆盖环境安装、数据采集、GP 训练、闭环运行、配置说明、输出文件、代码架构，以及 `test/` 下每个测试文件的用途。
>
> 所有命令默认在仓库根目录执行（配置里的路径都是相对路径）。

---

## 目录

1. [这个库是做什么的](#1-这个库是做什么的)
2. [环境安装](#2-环境安装)
3. [目录结构](#3-目录结构)
4. [5 分钟快速开始](#4-5-分钟快速开始)
5. [完整工作流：采数据 → 训练 → 跑闭环](#5-完整工作流采数据--训练--跑闭环)
6. [关键概念](#6-关键概念)
7. [配置参考](#7-配置参考)
8. [输出文件与指标](#8-输出文件与指标)
9. [代码架构与调用链](#9-代码架构与调用链)
10. [测试说明：test/ 每个文件测什么](#10-测试说明test-每个文件测什么)
11. [性能](#11-性能)
12. [常见问题（FAQ）](#12-常见问题faq)
13. [已知限制与后续方向](#13-已知限制与后续方向)

---

## 1. 这个库是做什么的

一个**用高斯过程回归（GPR）补偿模型误差的线性 MPC 轨迹跟踪控制器**，配套：

- 赛道/轨迹生成（Bezier、RacePath、五次多项式速度规划、非线性速度优化）；
- 车辆运动学模型（Cartesian / Frenet，支持延迟模型）；
- 线性 MPC（CasADi + OSQP/QPOASES），支持热启动、增量输入约束；
- GP 模型：零阶（只补偿均值）和一阶（同时补偿均值梯度，进入线性化矩阵 A/B/C）；
- 稀疏 GP：DTC / FITC，支持在线更新（`update`、快速添加诱导点）；
- 闭环仿真器（可 headless，也可动画/分析图）；
- 训练脚本（GPy）、模型保存/加载、日志与指标工具。

典型用途：在存在执行延迟、轮速不一致、未建模动力学时，用 GP 学习“MPC 名义预测与实际状态的残差”，提升跟踪精度。

---

## 2. 环境安装

要求 **Python >= 3.10**（代码使用了 `list[int]`、`X | None` 等语法）。

```bash
# 方式 A：只装依赖
python -m pip install -r requirements.txt

# 方式 B：以可编辑包安装（推荐）
python -m pip install -e .
```

依赖（见 `requirements.txt` / `pyproject.toml`）：

```text
numpy>=1.26.4
casadi>=3.6.7
GPy>=1.13.2
matplotlib>=3.9.1
PyYAML>=6.0.1
scipy>=1.12.0
colorama>=0.4.6
```

开发/测试：

```bash
python -m pip install -e ".[dev]"   # 会安装 pytest
python -c "import numpy, scipy, casadi, GPy, yaml, matplotlib; print('deps ok')"
```

> 本项目在 Windows + conda 环境验证过（Python 3.12、numpy 1.26.4、casadi 3.6.7、GPy 1.13.2）。
> 运行时出现 GPy/paramz 内部的 `DeprecationWarning` 属正常现象，不影响结果。

---

## 3. 目录结构

```text
GPRController/
├── config/
│   ├── mpc/linear_mpc_frenet.yaml     # MPC + GP + 仿真的主配置
│   ├── train/train.yaml               # GP 训练配置
│   └── track/*.yaml                   # 预生成赛道/轨迹（x, y, yaw, curvature, 时间, s, v, a, ω, a_ω）
├── controller/
│   └── linear_mpc.py                  # LinearMPC：线性化、QP 组装、求解、在线 GP 更新
├── model/
│   ├── cartesian_model.py             # 运动学 Cartesian / Cartesian Delay
│   ├── frenet_model.py                # 运动学 Frenet
│   ├── cartesian_gp_model.py          # Cartesian + GP（含 augmented）
│   ├── frenet_gp_model.py             # Frenet + GP（含 augmented）
│   ├── gp_model.py                    # 稀疏 GP（DTC/FITC）、mu_su/Cholesky、保存加载
│   ├── discrete.py                    # Euler / CrankNicolson / RK4 离散化
│   ├── noise_model.py                 # 过程噪声（协方差 Q·dt）
│   └── model_interface.py             # 模型接口
├── trajectory/
│   ├── traj.py                        # 读取预生成轨迹
│   ├── bezier.py                      # Bezier 路径 + 速度规划
│   ├── race_path.py                   # 赛道 yaml 重采样
│   ├── resample.py                    # 线性/样条重采样
│   ├── traj_optimizer.py              # 非线性速度优化（IPOPT）
│   └── quintic_polynomials.py         # 五次多项式
├── play/
│   ├── mpc_closed_loop.py             # SimulationRunner + MPCPlotter + test_mpc_closed_loop
│   ├── mpc.py                         # 旧入口的薄 wrapper（动画模式）
│   └── plot_result.py                 # 日志解析/画图（plot_delta_state 等）
├── train/
│   └── train_gp.py                    # GPy 训练、指标、写出训练/诱导数据
├── utils/
│   ├── config.py                      # 配置 dataclass + yaml 读取（严格校验）
│   ├── filter.py                      # 二阶 Butterworth 低通
│   ├── log.py                         # logging 封装
│   ├── criterion.py                   # MSE/RMSE/SMSE/MAE/LPD/MSLL
│   ├── angle.py / math.py / matrix_methods.py
├── test/                              # pytest 测试（详见第 10 节）
├── log/                               # 运行产物（通常被 .gitignore 忽略）
├── run.py                             # 示例：10 次“训练/加载 + 闭环 + 记录指标”
├── requirements.txt
├── pyproject.toml
└── GUIDE.md                           # 本文件
```

---

## 4. 5 分钟快速开始

### 4.1 跑默认闭环（headless）

```bash
python -c "from utils import Config; from play import test_mpc_closed_loop; cfg = Config.from_yaml('config/mpc/linear_mpc_frenet.yaml'); cfg.mpc.record_data = False; test_mpc_closed_loop(visualize=0, config=cfg)"
```

- `visualize=0`：不弹窗，跑完后打印性能摘要；
- `visualize=1`：额外弹跟踪/控制分析图；
- `visualize=2`：实时动画 + 分析图（等价于旧入口 `from play import test_mpc; test_mpc()`）。

> ⚠️ 默认配置里 `record_data: true`，且 `record_data_path` 指向
> `log/gpr_log/linear_mpc_frenet.txt`。控制器构造时会以 `'w'` 模式打开该文件，
> **直接运行默认配置会覆盖用于训练的原始数据**。只想看闭环效果时请在代码里把
> `cfg.mpc.record_data = False`；要采数据时请先把路径改成新文件或备份旧文件。

`test_mpc_closed_loop` 默认使用 `config/track/full_track_opt.yaml`，闭环逻辑在 `play/mpc_closed_loop.py`。

### 4.2 在 Python 脚本中运行（推荐）

```python
import numpy as np

from utils import Config
from trajectory import Trajectory
from controller import MPCList

cfg = Config.from_yaml("config/mpc/linear_mpc_frenet.yaml")
cfg.mpc.record_data = False   # 避免覆盖 log/gpr_log 下的训练数据
track = Trajectory("config/track/full_track_opt.yaml")

mpc = MPCList[cfg.mpc.mpc_name](cfg, track.path, track.speed_profile)

# 控制器始终接收完整 5 维 Cartesian 量测 [x, y, yaw, v, w]
state = np.array([
    [track.path[0, 0]], [track.path[0, 1]], [track.path[0, 2]],
    [track.speed_profile[0, 2]], [track.speed_profile[0, 4]],
])
mpc.initialize(state, cfg.mpc.control_dt)
u = mpc.solve()[:mpc.model.nu].squeeze()   # 本步要施加的 [δv, δw]
mpc.logger.close_log()
```

> `mpc.solve()` 返回的是整个预测时域的控制序列 `(nu * mpcWindow, 1)`，闭环通常只取前 `nu` 个。
> `initialize()` 每一步都要调用一次。

### 4.3 选择配置

`config/mpc/` 下的 yaml 就是完整配置。`test_mpc_closed_loop(config=None)` 会列出文件让你选；也可以像上面那样显式传入。

换赛道时修改 `play/mpc_closed_loop.py` 里写死的 `Trajectory("config/track/full_track_opt.yaml")`，或在脚本里自己构造 `SimulationRunner`：

```python
from play.mpc_closed_loop import SimulationRunner

runner = SimulationRunner(mpc, cfg)      # cfg / mpc 来自 4.2
state = runner.get_initial_state()
for _ in range(200):
    if runner.simulation_finished:
        break
    state = runner.run_step(state)
print(runner.termination_reason)
runner.mpc.logger.close_log()
```

`SimulationRunner` 负责仿真模型、噪声、低通滤波、控制延迟队列和终止条件；`run_step` 内部会调用 `mpc.initialize` + `mpc.solve` + 模型积分。

---

## 5. 完整工作流：采数据 → 训练 → 跑闭环

### 5.1 采集 GP 训练数据

1. 在 MPC 配置的 `mpc` 段设置：

```yaml
record_data:      true
record_data_path: "log/gpr_log/my_run.txt"
```

2. 跑闭环（任意方式）。控制器每步会写入一行 6 列数据：

```text
δv_nominal  δw_nominal  v  w  u_v  u_w
```

其中 `δv_nominal = (v_k - v_pred,k)/dt`、`δw_nominal = (w_k - w_pred,k)/dt` 是“实际状态 − 名义模型一步预测”的残差；`v/w` 是当前状态，`u_v/u_w` 是当前施加的控制。

> ⚠️ 每次构造 `LinearMPC` 都会用 `'w'` 模式打开 `record_data_path`，即**覆盖**旧文件。想保留历史数据就先备份，或让每次 run 使用不同的 `record_data_path`。

### 5.2 训练稀疏 GP

编辑 `config/train/train.yaml`：

```yaml
train_file_path:   "log/gpr_log/my_run.txt"      # 原始 6 列数据
fixed_data_path:   "log/processed_data/my_gp.txt" # 训练特征+标签，供 MPC 读取
train_data_number: 4000
test_data_number:  1000
input_window:      3
gap_window:        0
num_inducing:      40
method:            "FITC"   # FITC / DTC / Exact
optimize:          true
```

运行：

```bash
python -c "from train import test_train_gp; test_train_gp()"
```

它会对两个输出（δv 和 δw）各训练一个 GP，并（当 `write_data: true` 时）：

- 把“特征 + 标签”写入 `fixed_data_path`（先 `w` 清空）；
- 追加 `2 * num_inducing` 行诱导输入（第 1 个 GP 的在前，第 2 个的在后）；
- 打印每个 GP 的最优超参数，格式为：

```text
noise_variance, rbf_variance, lengthscale_1, ..., lengthscale_D
```

把这些数字按顺序填入 MPC 配置的 `gaussian.params`，每个 GP 一行：

```yaml
params:
  - [noise_var_v, rbf_var_v, l1_v, ..., l12_v]   # δv 的 GP
  - [noise_var_w, rbf_var_w, l1_w, ..., l12_w]   # δw 的 GP
```

> `Exact` 模式只适合离线评估；它不会写诱导输入，因此不能直接给稀疏 MPC 使用。
> 训练会覆盖 `fixed_data_path`，请确认该文件不是你想保留的旧数据。
>
> `fixed_data_path` 中“特征+标签”的行数为
> `min(train_data_number, 原始日志行数 - input_window)`；把这个值同步到 MPC 配置的
> `gaussian.total_data_number`，否则第 5.3 节的行数不变量会被破坏。
>
> 当前 `GaussianModel` 按 `gaussian.input_window * 4` 维构造核，而训练特征维度是
> `(input_window - gap_window) * 4`。因此请保持 `gap_window: 0`，并让训练与 MPC 的
> `input_window` 一致；`normalize_y` 也必须在训练配置和 MPC 配置中保持一致。

### 5.3 让 MPC 使用训练结果

在 `config/mpc/linear_mpc_frenet.yaml` 中：

```yaml
gaussian:
  path:                 "log/processed_data/my_gp.txt"
  total_data_number:    396      # 必须等于 fixed_data 文件中“特征+标签”的行数
  inducing_data_number: 40       # 必须等于训练时的 num_inducing
  input_window:         3        # 必须等于训练时的 input_window
  sparse_method:        3        # 0/2=DTC, 1/3=FITC
  read_model:           false    # 首次让控制器现场 fit；训练完成后可改为 true 加载缓存
  params: [...]                  # 粘贴 5.2 打印的超参数
```

注意不变量：

```text
processed 文件总行数 = total_data_number + 2 * inducing_data_number
```

如果对不上，GP 会读到空的/错误的诱导数据（会直接报错或得到异常模型）。

> 仓库自带的 `config/mpc/linear_mpc_frenet.yaml` 中 `params` 的 ARD lengthscale 目前是
> `1.0` 占位值；要发挥一阶 GP 的效果，请用训练打印出的真实超参数替换它们。

### 5.4 缓存模型并复用（可选）

`test_mpc_closed_loop` 结束时会调用 `mpc.save_model()`（仅 Gaussian 模型），把两个 GP 存到 `gaussian.save_path + `_0/_1` + `.npy`（实际是带元数据的 npz 容器，只是沿用 `.npy` 后缀）。

下一次把 `gaussian.read_model: true`，构造 `LinearMPC` 时会直接加载，不再现场拟合：

```yaml
gaussian:
  read_model: true
  save_path:  "log/save"
  load_path:  "log/save"
```

模型文件保存了 GP 类型（DTC/FITC）、核超参、`normalize_y`、特征维度、`cholesky_valid` 等元信息。加载时会校验：

- 文件与当前 `sparse_method` 是否同一类 GP；
- 文件中的核超参是否与当前 `gaussian.params` 一致；
- `normalize_y`、`input_window` 是否一致。

不一致会直接抛 `ValueError`，避免“加载了但静默用错模型”。

`run.py` 就是“第一次 fit + 保存，后续 9 次加载 + 跑闭环 + 记录指标”的示例（注意它会重复覆盖 `record_data_path`）。

---

## 6. 关键概念

### 6.1 模型族

`model/__init__.py` 的 `ModelLists` 提供：

| 配置名 | 状态维数 | 说明 |
|---|---|---|
| `Kinematic Cartesian` | 5 | `[x, y, yaw, v, w]` |
| `Kinematic Cartesian Delay` | 5 | 一阶延迟模型：`v_dot=(u_v-v)/τ_v`，`w_dot=(u_w-w)/τ_w` |
| `Kinematic Cartesian Gaussian` | 5 | Cartesian + GP 残差（零阶或一阶） |
| `Kinematic Augmented Cartesian Gaussian` | `5 + 4(W-1)` | 含 `W-1` 步状态/控制历史的 augmented 状态（`W=input_window>=2`）；`W=3` 时为 13 |
| `Kinematic Frenet` | 4 | `[e_y, e_phi, v, w]` 误差坐标 |
| `Kinematic Frenet Gaussian` | 4 | Frenet + GP 残差 |
| `Kinematic Augmented Frenet Gaussian` | `4 + 4(W-1)` | 同上，`W=3` 时为 12 |

> augmented 状态维度为 `base_nx + 4*(W-1)`，历史状态/控制 buffer 按 `W` 通用组装；`W>=2` 均可使用（回归测试覆盖 `W=2` 与 `W=4`）。`input_window=1` 没有历史可加，会明确报错。
> 模型层的 A/B/C 与 `LinearMPC` 的 `augmented_x0/u0` 现在保持一致；窗口越大，`Q/Q_final` 长度与训练特征维度（`4*W`）也要同步增大。
> 闭环仿真只支持 Cartesian 的 `sim_model`；Frenet 仿真模型缺少 `kappa` 参数，不支持。

### 6.2 GP 模式（`gpr_method`）

| 值 | 含义 |
|---|---|
| `-1` | 不把 GP 均值加入模型（名义 MPC；仍可保留 GP 对象用于记录） |
| `0` | 零阶：`C` 中加入 GP 均值，不把梯度写进 `A/B/C` |
| `1` | 一阶：均值 + 对特征的梯度进入 `A/B/C`（推荐） |

### 6.3 稀疏方法（`sparse_method`）

| 值 | 实现 | 说明 |
|---|---|---|
| `0` / `2` | DTC | `2` 是历史上“支持在线更新/存盘”的变体，数学与 `0` 相同 |
| `1` / `3` | FITC | `3` 是历史上“支持在线更新/存盘”的变体，数学与 `1` 相同 |

P1 之后，保存/加载对所有 Gaussian 稀疏方法都可用。

### 6.4 mu_su 与 Cholesky：quick 更新的语义

这是本库最重要的一个设计点：

- **`mu_su`（主路径）**：维护诱导点后验均值 `mu` 和协方差 `Su`，预测只依赖 `Luu/mu/Su`。`predict_noiseless()` 和 `predict(..., method='mu_su')` 都走这里。
- **Cholesky（辅助路径）**：额外维护 `Kuf/W/diag/W_d_inv_y/L`，用于 `predict_cholesky()`；训练集增删（`add_new_data` / `add_new_inducing` / `remove_data`）也依赖这套缓存。计算更贵。
- **`add_new_inducing_quick()`**：为在线算法设计，只更新 `U/M/Luu/mu/Su`，**故意不更新 Cholesky 相关缓存**，并置 `cholesky_valid=False`。
  - 此后 `predict_cholesky()` / `add_new_data()` / `add_new_inducing()` / `remove_data()` 会抛 `RuntimeError`，提示你使用 `predict_noiseless()` 或先调用 `sync_cholesky()`。
  - `sync_cholesky()` 会按当前 `U/X/Y` 重建 Cholesky 状态，代价是 `O(N·M)`；一旦需要导出/校验/切换预测器就调用它。
  - 注意：`sync_cholesky()` 不会覆盖 `mu/Su`（保留 quick 的增量近似），因此两种预测器的结果不要求逐位一致。

配置开关：

```yaml
gaussian:
  add_mu_su:    false   # true：在线增加诱导点（quick）
  update_mu_su: true    # true：在线更新后验（单点 update）
  threshold:    0.95    # 目前未使用（保留字段）
```

### 6.5 方差感知（`variance_aware`）

```yaml
mpc:
  variance_aware: true
  aware_len: 40
  lambda_var: [1, 1]
gaussian:
  gpr_method: 1
```

把 GP 预测方差对状态的梯度加入 QP 梯度。限制：

- 只支持 augmented Gaussian 模型（`Kinematic Augmented Frenet/Cartesian Gaussian`）；
- 必须 `gpr_method >= 1`；
- `lambda_var` 必须恰好两个权重；
- 每次 `setup_matrices` 会清零 `A_sigma/B_sigma`，`aware_len` 之外的部分不再残留旧值。

不满足以上条件会在构造 `LinearMPC` 时抛出明确的 `ValueError`。

---

## 7. 配置参考

### 7.1 `mpc` 段（核心）

| 字段 | 说明 |
|---|---|
| `mpc_name` | 目前只有 `"LinearMPC"` |
| `model` | `ModelLists` 中的模型名（见 6.1） |
| `discrete_method` | `Euler` / `CrankNicolson` / `RK4` |
| `solver` | `osqp` / `qpoases` |
| `mpcWindow` | 预测时域步数 |
| `mpc_dt` / `control_dt` | 控制器离散步长 / 实际控制周期；`mpc.control_dt` 必须与 `sim.control_dt` 一致（`Config` 会检查），建议三者保持一致 |
| `Q`, `Q_final`, `R` | 状态/终端/控制权重，长度必须等于模型 `nx`/`nu` |
| `max_input` | 控制幅值约束 `[δv_max, δw_max]` |
| `max_delta_input` | 控制增量约束 |
| `add_delta_constraints` | 是否把增量约束加进 QP 的 `a/lba/uba` |
| `linear_method` | `0`：在参考轨迹处线性化；`1`：在上一次解（热启动）处线性化 |
| `print_time` | 是否打印 solver 计时 |
| `propagete_cov` | 是否传播状态协方差（用于可视化/`x_var_list`） |
| `linearized` | 保留字段（当前未使用） |
| `variance_aware` / `aware_len` / `lambda_var` | 见 6.5 |
| `record_data` / `record_data_path` | 是否记录 6 列原始 GP 数据及其路径 |
| `log_path` | 控制器日志路径（默认 `log/linear_mpc_log.txt`） |

### 7.2 `gaussian` 段

| 字段 | 说明 |
|---|---|
| `path` | 处理后的训练数据（12 列特征 + 2 列标签，或含诱导行的混合文件） |
| `save_path` / `load_path` | GP 模型缓存的路径前缀（实际文件为 `<prefix>_0.npy`、`<prefix>_1.npy`） |
| `total_data_number` | 训练样本数（不含诱导行） |
| `inducing_data_number` | 每个 GP 的诱导点数（文件里共 2 组） |
| `input_window` | 历史窗口长度；特征维数 = `4 * input_window`；augmented 模型要求 >= 2 |
| `gpr_method` | `-1` / `0` / `1`（见 6.2） |
| `sparse_method` | `0/2` DTC、`1/3` FITC（见 6.3） |
| `normalize_y` | 是否标准化 GP 输出 |
| `lagacy` | 是否使用旧的 GPR 输入构造方式（一般 `false`） |
| `enable_test` | 构造模型时是否打印训练集预测误差 |
| `update_mu_su` / `add_mu_su` | 在线后验更新 / 在线加诱导点（见 6.4） |
| `threshold` | 保留字段（当前未使用） |
| `read_model` | `true`：从 `load_path` 加载；`false`：现场 `fit` |
| `params` | `[noise_variance, rbf_variance, lengthscale_1..D]`，每个 GP 一组 |

### 7.3 `sim` 段

| 字段 | 说明 |
|---|---|
| `sim_model` | 仿真模型，必须是 Cartesian（`Kinematic Cartesian` / `Kinematic Cartesian Delay`） |
| `SIM_TIMES` | 最大仿真步数 |
| `noise_scale` | 过程噪声方差缩放（`Q` 会乘这个值） |
| `delay_window` | 控制指令队列长度（延迟步数） |
| `start_index` | 从赛道第几个点开始（P2 起被 `mpc_closed_loop` 支持；仅 Cartesian sim） |
| `tau_scale` | Delay 模型时间常数 `τ = tau_scale * control_dt` |
| `filter_feq_v` / `filter_feq_w` | 低通截止频率（Hz，必须 < 1/(2·dt)） |
| `add_noise` / `add_filter` / `add_clamp` | 噪声 / 低通 / 非负速度钳位 |
| `sim_noise` | 连续时间噪声协方差率 `Q`（每步协方差为 `Q·dt`） |
| `bad_wheel` / `wheel_left` / `wheel_right` / `L` | 轮速不一致模型参数 |
| `max_lateral_error` | 横向误差超过此值终止 |
| `terminal_tolerance` | 到达终点的距离容差 |

> `MPCConfig.log_path` 和 `MPCConfig.add_delta_constraints` 是可选字段（有默认值）；`Config.from_yaml` 只接受 `mpc/gaussian/traj/sim` 这四个顶层 section，未知 section 会直接报错。

---

## 8. 输出文件与指标

| 文件 | 产生者 | 内容 |
|---|---|---|
| `log/linear_mpc_log.txt`（可配置为 `mpc.log_path`） | `LinearMPC.log_data` | 每步状态、残差、GPR 输入、矩阵 C、控制；`plot_delta_state` 从这里读 |
| `log/gpr_log/<name>.txt` | `record_data` | 6 列原始数据：`δv δw v w u_v u_w` |
| `log/processed_data/<name>.txt` | `train_gp` | 前 N 行 14 列（12 特征 + 2 标签），后 2M 行诱导输入 |
| `log/save_0.npy` / `log/save_1.npy` | `LinearMPC.save_model` | GP 模型（npz 容器，含元数据；兼容旧扁平 `.npy`） |
| `log/update.txt` | `run.py` | 每次迭代的 `inc_dv/inc_dw` 指标 |

指标：

- `plot_delta_state(lines)`（`play/plot_result.py`）返回相对名义模型的 MAE 改善百分比（dv, dw）。
- `print_performance_summary` 打印最终/最大/平均横向与航向误差。
- `test_gaussian_model`（`model/gp_model.py`）打印训练点平均误差（需要 `enable_test: true`）。

---

## 9. 代码架构与调用链

```text
Config.from_yaml(...)
  ├─ GaussianConfig（数据路径、稀疏方法、GP 超参）
  ├─ MPCConfig（时域、权重、约束、solver）
  └─ SimulationConfig（sim 模型、噪声、滤波、终止条件）

Trajectory(yaml) → path (N,4) + speed_profile (N,6)

MPCList[mpc_name](config, path, speed_profile)
  └─ LinearMPC
       ├─ model = ModelLists[config.mpc.model](...)
       ├─ initialize(state, dt)
       │    ├─ compute_error：最近点/横向/航向误差
       │    ├─ setup_ref / setup_matrices：逐 horizon 线性化，组装 Aex/Bex/Cex
       │    ├─ setup_bound / setup_qp_problem
       │    └─ prepare（首次 + linear_method=1 时）
       └─ solve()
            ├─ 热启动 QP（CasADi conic，缓存的 solver）
            ├─ 校验 solver 成功 + 有限性
            ├─ 计算 GP 残差 δv/δw，记录数据
            ├─ 在线更新 GP（update / add_mu_su）
            └─ 预测轨迹、保存 last_command、写日志

SimulationRunner(mpc, config)
  ├─ sim_model（Cartesian/Delay）、噪声模型
  ├─ LPFilter、控制延迟队列
  └─ run_step(state)
       ├─ mpc.initialize / solve
       ├─ 延迟队列 → 滤波 → 钳位 → 轮速模型
       ├─ sim_model.integrate + 过程噪声
       └─ 终止条件检查
```

GP 训练链：

```text
raw 6 列数据
  └─ get_pre_processed_data（滑窗，newest-first：v,w,u_v,u_w）
       └─ GPy 训练（FITC/DTC/Exact，可选超参优化）
            ├─ 打印超参数 → 粘贴到 mpc.gaussian.params
            └─ 写特征/标签 + 诱导输入 → mpc.gaussian.path
```

---

## 10. 测试说明：`test/` 每个文件测什么

运行：

```bash
python -m pytest -q                 # 全量（约 3 秒，58 个测试）
python -m pytest test/test_gp_persistence.py -v
python -m pytest -k variance        # 按名称过滤
```

> 如果环境不允许写 `.pytest_cache`（某些沙箱/受限目录），可加 `-p no:cacheprovider`。
> 部分测试会在仓库的 `build/` 下创建临时模型文件（已在 `.gitignore` 中），跑完可删除。

| 测试文件 | 覆盖内容 |
|---|---|
| `test_filter.py` | `LPFilter` 与 `scipy.signal.butter` 的响应逐点一致；阶跃稳态为 1；非法截止频率（0、负、≥ Nyquist）与非法 `dt` 被拒绝 |
| `test_config.py` | 只提供 `mpc` 或只提供 `sim` 的配置可构造；`control_dt` 一致时通过、不一致时报错；`from_yaml` 支持部分 section、拒绝未知 section、拒绝非 mapping section |
| `test_compute_error.py` | `compute_error` 返回 `float` 而不是 `(1,)` 数组；横向误差符号；局部窗口搜索；偏离较远时回退全局搜索 |
| `test_cartesian_model.py` | Cartesian 运动学：CasADi 积分 vs 线性化预测；远/近参考点对比（近点误差更小） |
| `test_frenet_model.py` | Frenet 误差动力学：CasADi 积分 vs 线性化预测；远/近参考点对比 |
| `test_cartesian_gp_model.py` | Cartesian GP 模型线性化（含 `m_curvature` 参数兼容）；augmented 版本；base 与 augmented 前 5 维一致；`set_predict_traj` 返回 `(N,5)` |
| `test_frenet_gp_model.py` | Frenet GP 模型线性化；augmented 版本；base 与 augmented 前 4 维一致；`set_predict_traj` 返回 `(N,5)` |
| `test_gp_model.py` | FITC（sparse 3）与 DTC（sparse 2）在真实数据上 fit；预测形状/有限性检查 |
| `test_gp_kernel.py` | 快速 numpy RBF 核/Kdiag 与 GPy 对齐；解析 `dk_dx` 与 GPy `dK_dX2` 对齐；`dmu_dx` 与有限差分对齐；批量输入被拒绝 |
| `test_gp_persistence.py` | FITC（normalize 开/关）与 DTC 保存/加载往返；quick 模式：cholesky 被拒绝 + `sync_cholesky` 恢复；quick 状态保存/加载；完整 `add_new_inducing` 后保存/加载；核超参/normalize/sparse 类型不匹配被拒绝；旧扁平 `.npy` 兼容 |
| `test_noise_model.py` | 过程噪声经验协方差 ≈ `Q·dt`；零 Q 返回零噪声；非法 dt/非方阵/非半正定 Q 被拒绝 |
| `test_controller_models.py` | 7 种模型族的 `initialize + solve`（含 `Kinematic Cartesian Delay`）；variance-aware 两种 augmented 模型；variance-aware 非法组合报错；augmented `input_window=1` 报错；solver 失败/非有限检查；solver 缓存复用；无 gaussian section 的普通模型可运行 |
| `test_augmented_input_window.py` | augmented 模型在 `input_window=2/4` 下（Frenet + Cartesian）可构造、求解并返回正确形状；`change_vector_to_model2` 对不一致 buffer 明确报错 |
| `test_simulation_runner.py` | `sim.start_index` 把车辆放到赛道对应点，且能正常跑一步 |

测试不访问网络；GP 相关测试使用 `log/processed_data/linear_mpc_frenet.txt` 和 `config/track/full_track_opt.yaml`。

---

## 11. 性能

在本机（Windows、Python 3.12、单线程）默认配置、缩短闭环下实测：

- **约 31 ms/步**（稳态，`mpcWindow=40`），满足 `control_dt=0.05 s` 的实时预算；
- P1 时约 158 ms/步，P2 的解析 RBF 核/梯度和 solver 缓存带来约 5× 提升；
- 主要剩余开销在 `setup_matrices` 中每个 horizon 步的 GP 均值/梯度评估。

简单 profiling 方法：

```python
import cProfile, pstats
from utils import Config
from trajectory import Trajectory
from controller import MPCList
from play.mpc_closed_loop import SimulationRunner

cfg = Config.from_yaml("config/mpc/linear_mpc_frenet.yaml")
cfg.mpc.record_data = False   # 避免覆盖 log/gpr_log 下的训练数据
track = Trajectory("config/track/full_track_opt.yaml")
mpc = MPCList[cfg.mpc.mpc_name](cfg, track.path, track.speed_profile)
runner = SimulationRunner(mpc, cfg)
state = runner.get_initial_state()
for _ in range(5):
    state = runner.run_step(state)   # 预热

pr = cProfile.Profile(); pr.enable()
for _ in range(30):
    state = runner.run_step(state)
pr.disable()
pstats.Stats(pr).sort_stats("cumtime").print_stats(15)
```

> 实际耗时与机器、`mpcWindow`、`nx`、GP 诱导点数相关；上面的数字仅作参考。

---

## 12. 常见问题（FAQ）

**Q1: `ModuleNotFoundError: No module named 'GPy' / 'casadi'`**
先 `python -m pip install -r requirements.txt`，并确认使用的解释器与 `python` 命令一致。

**Q2: `FileNotFoundError: config/mpc/...` 或数据文件不存在**
所有相对路径都假设当前目录是仓库根目录。用 `cd` 到 `GPRController/` 再运行；或把配置里的路径改成绝对路径。

**Q3: `RuntimeError: ... only maintains the mu_su ...`**
这是 quick 模式的预期行为：`add_new_inducing_quick()` 之后 Cholesky 状态失效。用 `predict_noiseless()` / `predict(method='mu_su')`；确需 Cholesky 时先 `gp.sync_cholesky()`。

**Q4: `ValueError: model file ... different kernel hyperparameters`**
`read_model: true` 时，配置文件里的 `gaussian.params` 必须与训练/保存时一致。改过超参后要么把 `read_model` 设回 `false` 重新 fit，要么重新训练并保存。

**Q5: `ValueError: Unknown config section(s) ...`**
`Config.from_yaml` 现在只接受 `mpc/gaussian/traj/sim`。删除拼错的顶层 key，或改成合法 section。

**Q6: `ValueError: Augmented models require gaussian.input_window >= 2`**
augmented 模型需要至少 2 步历史（`W-1 >= 1`）。把 `input_window` 设为 >=2（常用 3）；同时保证训练数据特征维度为 `4*input_window`、`Q/Q_final` 长度等于 `base_nx + 4*(W-1)`。

**Q7: `ValueError: LPFilter: cutoff_freq ... must be in (0, ...)`**
低通截止频率必须小于 Nyquist（`1/(2·dt)`）。例如 `dt=0.05` 时上限 10 Hz，配置用 5 Hz。

**Q8: `RuntimeError: LinearMPC solve: QP solver failed with status ...`**
求解器未收敛。可尝试增大 `mpcWindow` 之外的求解器迭代/精度、放宽 `max_delta_input`、检查 `Q/Q_final/R` 是否让问题病态；当前实现不会静默使用失败解。

**Q9: 仿真报 Frenet 相关错误**
闭环只支持 Cartesian `sim_model`；`sim_model` 请用 `Kinematic Cartesian` 或 `Kinematic Cartesian Delay`。

**Q10: 训练时提示“data has only N data!”**
`train.yaml` 的 `train_data_number` 大于原始日志行数，程序会截断到可用长度。把 `totals`/`total_data_number` 按实际行数调整即可。

**Q11: 训练后 MPC 读数据报列数/行数错误**
检查 `processed 文件行数 = total_data_number + 2*inducing_data_number`，且 `input_window` 与训练一致；`params` 的长度应为 `1 + 1 + 4*input_window`。

**Q12: 运行后训练日志被清空**
`record_data: true` 时构造控制器会用 `'w'` 覆盖 `record_data_path`。采集数据时建议每次使用独立文件名或先备份。

---

## 13. 已知限制与后续方向

- augmented 模型支持 `input_window >= 2`（`W=1` 明确报错）；窗口越大，特征维度和 QP 状态维度越高，需要同步调整训练数据与 `Q/Q_final` 长度。
- `play/mpc_closed_loop.py` 中的赛道路径仍写死为 `config/track/full_track_opt.yaml`；换赛道需要改代码或用自定义 runner。
- `play/mpc.py` 现在是 `test_mpc_closed_loop(visualize=2)` 的薄 wrapper；旧的“整段 horizon 回放 + 交互输入”调试模式已移除。
- `remove_inducing()` 未实现（显式 `NotImplementedError`）。
- `test_nonliner_speed_optimize()`（`trajectory/traj_optimizer.py`）依赖的 `config/track/bezier_path.yaml`、`config/speed_optimizer/bezier.yaml` 不在仓库中，需要自行补配置。
- `method: Exact` 的训练分支只适合离线评估，且其 `write_data` 路径是旧的硬编码路径，不建议用于 MPC 数据准备。
- 进程噪声按连续协方差率 `Q·dt` 采样（P1 修改），与旧版本 `Q/dt` 的数值略有差异；跨版本对比实验时注意。
- 剩余性能热点：GP 均值与梯度在每个 horizon 步分别评估，可进一步合并/向量化。

---

## 附录：最小 API 速查

```python
# 配置 / 轨迹 / 控制器
from utils import Config
from trajectory import Trajectory
from controller import MPCList

cfg   = Config.from_yaml("config/mpc/linear_mpc_frenet.yaml")
track = Trajectory("config/track/full_track_opt.yaml")
mpc   = MPCList[cfg.mpc.mpc_name](cfg, track.path, track.speed_profile)

# 闭环一步
mpc.initialize(state_5x1, cfg.mpc.control_dt)
u_full = mpc.solve()

# GP
from model.gp_model import GaussianModel
gp_model = GaussianModel(cfg.gaussian)
gp = gp_model.gps[0]
mu, var = gp.predict_noiseless(x, full_cov=True)   # mu_su
gp.add_new_inducing_quick(x)                        # 只维护 mu_su
gp.sync_cholesky()                                  # 需要 cholesky 时
mu, var = gp.predict(x, method="cholesky")

# 训练 / 训练配置
from train import test_train_gp
from utils.config import TrainConfig
train_cfg = TrainConfig.from_yaml("config/train/train.yaml")

# 日志指标
from play import plot_delta_state
with open(cfg.mpc.log_path) as f:
    inc_dv, inc_dw = plot_delta_state(f)
```

---

*如果本指南与代码不一致，以当前 `vibe_code` 分支代码为准；改动接口时请同步更新本文件。*
