# 闭环与离线实验工具（play/）

这一组脚本复现论文第 3 章的离线 GP 对比，以及第 4.6.2 节的闭环消融、在线学习、
跨域与方差感知实验。所有命令都在**仓库根目录**执行（代码里的路径都是相对路径）。

> 文档只描述当前代码的实际行为；论文中的图号/表号以脚本 docstring 与提交记录为准。
> 如果某个脚本的 docstring 与代码不一致，本文按代码记录，并在“注意事项”里标出差异。
>
> 闭环仿真的核心仍是 `play/mpc_closed_loop.py` 的 `SimulationRunner`；本目录脚本都复用它，
> 只是负责场景、种子、日志和图表编排。

## 前置条件

- conda 环境 `ai`（Python 3.12.4 / numpy 1.26.4 / casadi 3.6.7 / GPy 1.13.2），
  与 `GUIDE.md` 第 2 节一致；
- 仓库根目录执行；`config/track/full_track_opt.yaml` 存在；
- 需要镜像赛道时，`config/track/full_track_mirror.yaml` 已生成（可用
  `play/make_mirror_track.py` 重新生成）。

```bash
conda activate ai
python -c "import numpy, casadi, GPy; print('deps ok')"
```

---

## 1. 脚本索引

| 脚本 | 用途 | 论文位置 | 主要产物 |
| --- | --- | --- | --- |
| `play/pipeline_sweep.py` | 名义 MPC 录制 → 构造训练文件 → GPy 训练 → 闭环消融 | 4.6.2 | `log/record/<tag>.txt`、`log/processed_data/<tag>.txt`、`log/sweep/<tag>_ablation.csv`、`<tag>_params.txt` |
| `play/sweep_closed_loop.py` | 只用已有模型跑消融矩阵（不录制/训练） | 4.6.2 | `log/sweep/<tag>.csv`（追加） |
| `play/compare_updates.py` | 8 种在线更新规则对比 | 4.6.2 在线更新 | `log/sweep/updates_<scenario>.csv` |
| `play/cross_domain.py` | 在赛道 A 训练、赛道 B（镜像）部署 | 跨域实验 | `log/sweep/<tag>_<scenario>.csv`、`cross_train_params.txt` |
| `play/unlearn_rounds.py` | 故障工况训练 → 正常工况多轮部署/遗忘 | 4.6.2.3 | `log/sweep/<tag>_rounds.csv`（默认 `ul_rounds.csv`）、`online_learning_rounds.pdf` 等 |
| `play/aware_sweep.py` | 方差感知权重扫描（延迟工况，噪声开/关） | 表 4.4 | `log/sweep/aware_sweep.csv` |
| `play/aware_scenarios.py` | 方差感知权重在跨域/错误先验场景下的扫描 | 4.6.2 应力场景 | `log/sweep/aware_<case>.csv` |
| `play/dump_trajectories.py` | 重放指定工况并导出轨迹 npz | 4.6.2 插图 | `log/sweep/traj/*.npz` |
| `play/make_figures.py` | 由 npz 画轨迹图/误差图（PDF） | 4.6.2 插图 | `sim_traj_*.pdf`、`sim_err_*.pdf` |
| `play/eval_residual.py` | 残差预测精度（噪声开/关）与残差图 | 表 4.2、图 4.5/4.8 | `log/sweep/residual_<scenario>.csv`、`sim_residual_*.pdf` |
| `play/probe_wheel.py` | 名义 MPC 横向误差 vs 轮速效率 η_L | 参数选择 | 终端输出 + `log/sweep/probe_<η>.txt` |
| `play/probe_gp_alignment.py` | 检查 GP 预测与残差的时间对齐 | 指标定义 | 终端输出 |
| `play/probe_var_ratio.py` | 镜像赛道上后验方差比分布 | 选择 `variance_ratio` | 终端输出 |
| `play/pred_benchmark.py` | 纯预测对比：Exact / DTC / FITC / variational FITC | 3.4，表 3.1/3.2 | 终端表格 |
| `play/toy_benchmark.py` | 一维 toy 在线学习对比 | 3.5，表 3.3–3.5 | 终端表格 |
| `play/four_gp_figure.py` | 四种 GP 的预测对比图 | 图 3.2 | `compare_4gp.png` |
| `play/trained_figure.py` | DTC 训练前/后对比 | 图 3.3 | `trained.png` |
| `play/toy_figures.py` | toy 在线学习对比图 | 第 3 章插图 | `<name>.png` |
| `play/replot_rounds.py` | 从 CSV 重画轮次曲线（可读标签） | 4.6.2.3 | `lateral_error.pdf`、`heading_error.pdf` |
| `play/make_mirror_track.py` | 生成镜像赛道 yaml | 跨域实验 | `config/track/full_track_mirror.yaml` |
| `play/_agg.py` | 汇总 `<tag>_ablation.csv` 为均值±标准差 | 4.6.2 表格 | 终端输出 |
| `play/_agg2.py` | 汇总 `updates_<scenario>.csv` | 在线更新表格 | 终端输出 |

---

## 2. 第 4 章：闭环 / 在线学习实验

### 2.1 一条龙：`pipeline_sweep.py`

对单个工况执行：

1. 名义控制器（`gpr_method = -1`）录制残差数据；
2. 把录制数据转成 GP 训练文件（12 维特征 + 2 维响应 + `2M` 行诱导输入）；
3. 用 GPy 训练两个 FITC 稀疏 GP，保存超参数；
4. 用该工况模型跑闭环消融：

```text
nominal          (-1, F, F)  名义 MPC，不含 GP
gp0_frozen       ( 0, F, F)  零阶 GP，后验冻结
gp0_update       ( 0, T, F)  零阶 GP，在线后验更新
gp0_update_add   ( 0, T, T)  零阶 GP，在线更新 + 扩诱导点
gp1_update_add   ( 1, T, T)  一阶 GP，在线更新 + 扩诱导点
```

常用命令：

```bash
# 延迟工况，3 个随机种子
python -m play.pipeline_sweep --scenario delay --tag delay3 --repeats 3

# 单侧驱动力不足工况；论文基线用 eta_L = 0.925
python -m play.pipeline_sweep --scenario bad_wheel --tag bad_wheel925 \
    --wheel-left 0.925 --repeats 3
```

参数：

- `--scenario {delay,bad_wheel}`（必填）
- `--tag`：产物前缀，默认等于 scenario
- `--repeats`：每个 variant 的种子数（0..N-1）
- `--variants`：默认全部 5 个
- `--restarts`：GPy `optimize_restarts` 次数（默认 3）
- `--window` / `--gap` / `--n-inducing`：默认 3 / 0 / 40
  - 当前 `GaussianModel` 按 `input_window*4` 构造特征，`gap>0` 会让训练特征变成
    `4*(window-gap)`、与控制器不一致；除非同时改控制器 `input_window`，否则保持 `--gap 0`。
- `--wheel-left`：覆盖 bad_wheel 的 η_L（`SCENARIOS` 默认 0.975；论文基线需显式 0.925）

产物：

```text
log/record/<tag>.txt                 录制原始数据：Δv Δω v ω a δ
log/processed_data/<tag>.txt         特征+响应 + 2M 行诱导输入
log/sweep/<tag>_ablation.csv         每个 (variant × seed) 一行
log/sweep/<tag>_params.txt           训练超参数（data= / n= / gp0= / gp1=）
log/sweep/record_<tag>.txt           名义录制控制器日志
log/sweep/mpc_<scenario>_<variant>_<seed>.txt
```

约定：

- 训练/部署必须来自**同一工况**；`pipeline_sweep` 已保证。
- 训练时 `normalizer=False`（不随 yaml 的 `normalize_y` 走）；`GaussianConfig.params`
  为 `[noise_var, rbf_var, length_1..12]`。
- `R = diag(0.1, 0.1)`、`max_input = [1.0, 1.5]`，与 4.6.2.1 一致。

### 2.2 只用已有模型：`sweep_closed_loop.py`

不录制/训练，直接使用 `config/mpc/linear_mpc_frenet.yaml` 的 `gaussian.path`
和现有模型跑消融；CSV 采用**追加**模式，同 tag 重跑会产生重复行。

```bash
python -m play.sweep_closed_loop --tag smoke --variants nominal \
    --scenarios delay --repeats 1

python -m play.sweep_closed_loop --tag full --repeats 3 --paper-params
```

参数：

- `--tag`：CSV 文件名（`log/sweep/<tag>.csv`）
- `--variants` / `--scenarios` / `--repeats` / `--seed0`
- `--read-model`：从 `log/save_*.npy` 加载，而不是现场 fit
- `--paper-params`：强制 `R=diag(0.1,0.1)`、`max_input=[1.0,1.5]`

### 2.3 在线更新规则：`compare_updates.py`

在同一份训练模型上比较 8 个配置（docstring 写的是 seven，以代码为准）：

```text
nominal / frozen / osvgp / osvgp_add / first / bijl / agp_window / agp
```

依赖 `pipeline_sweep` 生成的 `<params-tag>_params.txt`：

```bash
python -m play.compare_updates --scenario delay --params-tag delay3 --repeats 3
python -m play.compare_updates --scenario bad_wheel --params-tag bad_wheel925 --repeats 3
```

参数：`--scenario`、`--params-tag`、`--variants`、`--repeats`。

产物：`log/sweep/updates_<scenario>.csv`、`log/sweep/upd_<scenario>_<variant>_<seed>.txt`。

更新规则（`gaussian.update_method`）：

| 值 | 行为 |
| --- | --- |
| `vfe` | 秩一变分/卡尔曼式后验更新（默认），`lambda_` 作为遗忘因子 |
| `bijl` | Bijl & van Wingerden 的非变分 FITC 秩一更新 |
| `window` | `add_new_data` + `remove_data(1)` 的定长滑窗 |
| `agp` | 遗忘更新 + 按边际贡献加/删诱导点（预算 `2 × inducing_data_number`） |

### 2.4 跨域：`make_mirror_track.py` + `cross_domain.py`

`make_mirror_track.py` 把赛道沿 x 轴镜像（`y / yaw / curvature / omega / acc_omega` 反号），
动力学和故障不变，只翻转曲率方向：

```bash
python -m play.make_mirror_track \
    --src config/track/full_track_opt.yaml \
    --dst config/track/full_track_mirror.yaml
```

`cross_domain.py` 在训练赛道录制/训练，在测试赛道（默认镜像）部署：

```bash
python -m play.cross_domain \
    --train-track config/track/full_track_opt.yaml \
    --test-track  config/track/full_track_mirror.yaml \
    --scenario delay --tag cross --repeats 3
```

参数：

- `--scenario`：训练/部署共用故障（`delay` / `bad_wheel` / `normal`）
- `--train-scenario` / `--test-scenario`：可分别覆盖
- `--criterion {subspace,variance}`、`--ratio`（默认 3.5）
- `--rounds > 1`：多轮保存/加载（`log/save_round_*`）
- `--variants`：默认 8 个（nominal/frozen/osvgp/osvgp_add/first/bijl/agp/agp_window）

产物：`log/sweep/<tag>_<scenario>.csv`、`log/sweep/<tag>_train_params.txt`、
`log/sweep/cd_<scenario>_<variant>_<seed>.txt`。

### 2.5 多轮遗忘/反学习：`unlearn_rounds.py`

流程（对应 4.6.2.3）：

1. 用异常工况（`bad_wheel, wheel_left=0.925`）录制并用名义控制器训练；
2. 在正常工况（`bad_wheel=False`）连续部署 `--rounds` 轮：
   - 第 0 轮：现场 fit，**不做在线更新**；
   - 第 r>0 轮：加载上一轮保存的模型，做变分更新（incremental 变体再加诱导点）；
3. 四类对比：`trained/default` 超参数 × `non-inc/inc`。

```bash
python -m play.unlearn_rounds --rounds 10 \
    --figure-dir ../zjuthesis/figure/gpmpc
```

参数：`--rounds`（默认 10）、`--restarts`（默认 2）、`--tag`（默认 `ul`）、
`--figure-dir`（可选）。

产物：

```text
log/sweep/<tag>_rounds.csv
log/sweep/ul_<class>_<round>.txt
log/save_<tag>_<class>_0.npy / _1.npy   （每轮覆盖，用于下一轮加载）
<figure-dir>/online_learning_rounds.pdf
<figure-dir>/lateral_error.pdf
<figure-dir>/heading_error.pdf
```

> 默认 `--tag ul`，所以是 `log/sweep/ul_rounds.csv` 和 `log/save_ul_<class>_{0,1}.npy`。

### 2.6 方差感知权重

#### `aware_sweep.py`（表 4.4，延迟工况）

一阶 GP-MPC，`(lambda_v, lambda_omega)` 取 `0, 0.05, 0.1, 0.25, 0.5`，
分别在 `add_noise=True/False` 下扫描：

```bash
python -m play.aware_sweep --params-tag delay3 --repeats 1
```

产物：`log/sweep/aware_sweep.csv`、`log/sweep/aware_<lam>_<n|nn>.txt`。

> 脚本 docstring 里的候选值 `{(0,0),(1,1),(5,5),(10,10)}` 已过时；以 `LAMBDAS` 为准。

#### `aware_scenarios.py`（跨域/错误先验）

```bash
python -m play.aware_scenarios --case cross
python -m play.aware_scenarios --case unlearn0
```

- `cross`：`delay3` 模型 + 镜像赛道（回归量超出训练支撑，方差更大且分布不均）；
- `unlearn0`：`bad_wheel925` 模型 + 正常工况、不做在线更新（错误先验）；
- 两者 `lambda = 0, 1, 5, 10`。

产物：`log/sweep/aware_<case>.csv`、`log/sweep/aws_<case>_<lam>.txt`。

### 2.7 轨迹、图与残差精度

#### `dump_trajectories.py`

读取 `log/sweep/<tag>_params.txt`，重放三个配置（`nominal / gp0 / gp1`）：

```bash
python -m play.dump_trajectories --scenario delay --tag delay3
python -m play.dump_trajectories --scenario bad_wheel --tag bad_wheel925
```

产物：`log/sweep/traj/<scenario>_<name>.npz`（`state / sim_path / control / ref`）。

#### `make_figures.py`

```bash
python -m play.make_figures --outdir ../zjuthesis/figure/gpmpc
```

读取 `log/sweep/traj/*.npz`，输出 4 个 PDF：

```text
sim_traj_delay.pdf, sim_err_delay.pdf,
sim_traj_badwheel.pdf, sim_err_badwheel.pdf
```

#### `eval_residual.py`

用训练后验在日志里的 `GPR Input` 上重新预测，并与同一步的
`Delta State Nominal` 比较；分别在噪声开/关下跑：

```bash
python -m play.eval_residual --scenario delay --params-tag delay3 \
    --figure-dir ../zjuthesis/figure/gpmpc
python -m play.eval_residual --scenario bad_wheel --params-tag bad_wheel925 \
    --figure-dir ../zjuthesis/figure/gpmpc
```

产物：`log/sweep/residual_<scenario>.csv`、`log/sweep/res_*.txt`、`sim_residual_*.pdf`。

---

## 3. 第 3 章：离线 GP 对比

### 3.1 纯预测：`pred_benchmark.py`

数据：`[0,20]` 上 51 个 `x sin x + 0.33` 噪声样本，11 个诱导点，预测网格 `[-5,25]`。
对比 Exact GP / DTC / FITC / variational FITC，并额外打印训练点上（含噪/无噪）的误差；
表 3.2 还包含 DTC 超参与诱导点联合优化前后的结果。

```bash
python -m play.pred_benchmark
```

无文件输出，指标打印到终端（RMSE / MAE / NLL / ms）。

### 3.2 Toy 在线学习：`toy_benchmark.py`

两种设置：

- `same`：训练/更新都在 `[0,20]`，更新目标变为 `0.33 x sin x`（表 3.3）；
- `cross`：训练 `[0,10]`、更新 `[5,20]`，目标 `0.3 x sin x`（表 3.4/3.5）。

方法：

```text
standard    GPy 精确 GP，不更新（基线）
osvgp       SparseFITCGP.update             （秩一/变分）
bijl        SparseFITCGP.update_mu_su_bijl  （非变分 FITC）
agp_window  add_new_data/remove_data，lambda=0.99（滑窗，Graduate_Data 里的 AGP）
agp_addrm   变分更新 + 边际贡献加/删诱导点（本仓库的 AGP）
```

```bash
python -m play.toy_benchmark
python -m play.toy_benchmark --repeats 20
python -m play.toy_benchmark --cases same cross --incremental
```

> `--incremental` 会在更新时先做 `check_new_inducing_var` 阈值判断，再扩诱导点；
> 与 `agp_addrm` 的“加/删”是不同机制。

### 3.3 图 3.2 / 图 3.3

```bash
python -m play.four_gp_figure --outdir ../zjuthesis/figure/gpmpc   # compare_4gp.png
python -m play.trained_figure --outdir ../zjuthesis/figure/gpmpc   # trained.png
```

- `four_gp_figure`：Exact / DTC / FITC / variational FITC，同一数据、同一 11 个诱导点；
- `trained_figure`：DTC 训练前（未优化）vs 训练后（超参 + 诱导位置联合优化，
  5 次 restart），两栏诱导点数相同；
- 两图都把视野限制在数据区，避免稀疏模型在数据外的方差带把坐标轴拉爆。

### 3.4 Toy 对比图：`toy_figures.py`

```bash
python -m play.toy_figures --outdir ../zjuthesis/figure/gpmpc \
    --name compare_in_same
python -m play.toy_figures --outdir ../zjuthesis/figure/gpmpc \
    --name compare_cross --incremental
```

- `--name` 只用于决定 case（`compare_in_same` → `same`，其他 → `cross`）和输出文件名；
- 输出 `<name>.png`；四宫格顺序为 before / Bijl / OSVGP / ASGP。

### 3.5 重画轮次曲线：`replot_rounds.py`

从 `log/sweep/ul_rounds.csv` 重画 `lateral_error.pdf` / `heading_error.pdf`。

```bash
python -m play.replot_rounds
```

> 该脚本把 CSV 和输出目录写成了**绝对 Windows 路径**；换机器/换目录前先改
> `CSV` 与 `OUT`。

---

## 4. 探针与诊断

这些脚本用于“选参数/定义指标”，不是正式实验，路径多为硬编码。

### `probe_wheel.py`

名义 MPC（`gpr_method=-1`）在 `η_L = 0.975/0.95/0.925/0.90` 下的横向/航向误差：

```bash
python -m play.probe_wheel
```

输出到终端和 `log/sweep/probe_<η>.txt`。脚本里硬编码了一组旧 GP 参数和
`log/processed_data/bad_wheel.txt`；数据/参数变化后需要同步更新。

### `probe_gp_alignment.py`

检查训练后的 GP 预测对应的是未来第几步：把日志里的 `GPR Input` 送进模型，
对 `shift = 0..4` 计算与 `Delta State Nominal` 的误差，用来确定残差指标的时间对齐。

```bash
python -m play.probe_gp_alignment
```

依赖 `log/sweep/res_delay_gp0_n.txt` 和 `log/sweep/delay3_params.txt`。

### `probe_var_ratio.py`

在镜像赛道的运行日志上，统计后验预测方差相对训练集参考方差 `var_ref` 的比值分布，
用于选择 `inducing_criterion="variance"` 时的 `variance_ratio`。

```bash
python -m play.probe_var_ratio
```

依赖 `log/sweep/cd_delay_osvgp_add_0.txt` 和 `log/sweep/cross_train_params.txt`
（先跑 `cross_domain`）。

---

## 5. 辅助脚本

### `make_mirror_track.py`

见 2.4。输出 `config/track/full_track_mirror.yaml`；镜像后长度、速度曲线不变，
只翻转 y/航向/曲率/角速度符号。

### `_agg.py`

汇总 pipeline 的 `<tag>_ablation.csv`（硬编码 `delay3`、`bad_wheel925`）：

```bash
python -m play._agg
```

输出每个 variant 的 `mean±std`、最大横向误差、平均航向误差、ms/step、步数、最终 M。

### `_agg2.py`

汇总 `updates_<scenario>.csv`（硬编码 `delay`、`bad_wheel`）：

```bash
python -m play._agg2
```

---

## 6. 指标定义

### 6.1 闭环控制精度

由 `SimulationRunner.state` 逐步记录，取绝对值后统计：

- `mean_lat` / `max_lat`：横向误差均值/最大值；
- `mean_head` / `max_head`：航向误差均值/最大值；
- 部分脚本还记录 `final_lat` / `final_head`；
- 汇总脚本把横向误差换算成 mm，航向误差保持 rad。

### 6.2 时间

- `sim_s`：闭环仿真墙钟时间；
- `init_s`：构造控制器（可能含 GP fit/加载）的时间（部分脚本记录）；
- `ms_per_step = 1000 * sim_s / steps`，不含绘图。

### 6.3 残差预测精度

- 名义预测：视残差为 0，误差 `mean|d_k|`；
- GP 预测：训练后验在 `z_k` 上的均值，误差 `mean|d_k - μ(z_k)|`；
- 改善幅度：`1 - GP误差 / 名义误差`。

### 6.4 纯预测 / toy 指标

- RMSE、MAE、NLL：
  `NLL = mean(0.5 log(2π var) + (y-μ)²/(2 var))`；
- toy 额外统计每次更新的平均墙钟时间（ms/update）。

---

## 7. 配置与场景要点

| 项 | 取值 | 说明 |
| --- | --- | --- |
| `mpc.R` | `diag(0.1, 0.1)` | 脚本强制覆盖；yaml 默认 0.2 |
| `mpc.max_input` | `[1.0, 1.5]` | 线加速度 / 角加速度上限 |
| `sim.delay_window` | 2（delay）/ 1（bad_wheel） | 2 步 = 100 ms @ 50 ms |
| `sim.bad_wheel` | `True, wheel_left=0.925` | 论文基线；0.975 偏差较大 |
| `gaussian.input_window` | 3 | 12 维回归量 |
| `gaussian.inducing_data_number` | 40 | 初始诱导点数 |
| `gaussian.add_mu_su` | 在线扩充上限 `10×` | `LinearMPC._should_add_inducing`；`agp` 内部预算 `2×` |
| `gaussian.update_method` | `vfe / bijl / window / agp` | 见 2.3 |
| `gaussian.lambda_` | `1.0` 或 `0.99` | 遗忘因子；闭环 `agp` 用 0.99；`agp_window` 在 toy 里用 0.99，闭环 `window` 方法不使用遗忘（1.0） |
| `gaussian.inducing_criterion` | `subspace / variance` | 诱导点触发准则 |
| `gaussian.variance_ratio` | 默认 4.0；`cross_domain` 用 3.5 | 仅 `variance` 准则 |
| `mpc.variance_aware` | `aware_sweep` 中 `lam>0` 时开 | `lambda_var`、`aware_len=40` |
| `sim.add_noise` | 默认 true | 过程噪声按 `Q·dt` 采样；v/w 标准差约 2.2e-3 |

> 旧版 SWEEP 写的“观测噪声 σ=0.01”与当前 `noise_model` 不一致：当前按连续协方差率
> `Q·dt` 采样，`sim_noise` 的 v/w 项为 `1e-4`、`dt=0.05`，标准差约 `2.2e-3`。

---

## 8. 产物路径

| 路径 | 内容 |
| --- | --- |
| `log/record/<tag>.txt` | 名义录制原始数据：`Δv Δω v ω a δ` |
| `log/processed_data/<tag>.txt` | 特征(12)+响应(2)，后接 `2M` 行诱导输入 |
| `log/sweep/<tag>_ablation.csv` | pipeline 的消融结果 |
| `log/sweep/<tag>_params.txt` | 训练超参数（`data=/n=/gp0=/gp1=`） |
| `log/sweep/<tag>.csv` | `sweep_closed_loop` 的追加结果 |
| `log/sweep/updates_<scenario>.csv` | 在线更新规则对比 |
| `log/sweep/<tag>_<scenario>.csv` | 跨域结果 |
| `log/sweep/<tag>_rounds.csv` | 多轮遗忘/反学习结果（默认 `ul_rounds.csv`） |
| `log/sweep/aware_sweep.csv` | 方差感知权重扫描 |
| `log/sweep/aware_<case>.csv` | 跨域/错误先验的方差感知扫描 |
| `log/sweep/residual_<scenario>.csv` | 残差预测精度 |
| `log/sweep/traj/*.npz` | 轨迹、误差、控制、参考路径 |
| `log/sweep/mpc_*.txt` / `upd_*.txt` / `cd_*.txt` / `ul_*.txt` / `aws_*.txt` | 各实验的控制器日志 |
| `log/save_*` | GP 模型缓存（在线更新/多轮实验用） |
| `<figure-dir>/*.pdf` / `*.png` | 论文插图 |

---

## 9. 注意事项与已知差异

1. **`pipeline_sweep` 的 `bad_wheel` 默认 η_L 是 0.975**；论文基线是 0.925，必须
   显式 `--wheel-left 0.925`。
2. **`sweep_closed_loop` 追加 CSV**：同一 `--tag` 重跑会重复行；需要干净结果时先删旧 CSV。
3. **`pipeline_sweep` / `compare_updates` / `aware_*` 覆盖 CSV**（`w` 模式），不要并行写同一路径。
4. **`compare_updates` 实际是 8 个 variant**（docstring 写 seven，以 `VARIANTS` 为准）。
5. **`aware_sweep` 的 docstring 候选 λ 已过时**；实际 `LAMBDAS=(0,0.05,0.1,0.25,0.5)`。
6. **`replot_rounds` 硬编码绝对路径**（CSV 与输出目录）；换机器必须改。
7. **`probe_*` 依赖已有日志/参数文件**，见第 4 节；数据/模型更新后检查硬编码路径。
8. **`probe_wheel` 的 GP 参数是旧常量**；它主要用 `gpr_method=-1`，GP 参数只影响加载。
9. **`make_figures` 依赖 `dump_trajectories` 的 npz**，顺序不能反。
10. **`toy_benchmark` 的 `agp_window` 在 `fit` 时用 `lambda_=0.99`**；
    `agp_addrm` 用变分更新 + 边际贡献加/删诱导点。
11. **Seeds**：脚本用 `np.random.seed(seed)`；`pipeline_sweep` 的录制固定 seed=0。
12. **训练特征与在线特征顺序必须一致**：训练用 `get_pre_processed_data`（newest-first），
    控制器日志的 `GPR Input` 也是同一顺序。
