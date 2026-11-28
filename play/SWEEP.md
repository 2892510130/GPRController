# 闭环实验工具（play/）

这一组脚本用于复现论文 4.6.2 节的仿真实验：从零开始录制数据、训练 GP、跑闭环消融，
并生成论文中的表格与插图。

## 前置条件

- conda 环境 `ai`（Python 3.12.4 / numpy 1.26.4 / casadi 3.6.7 / GPy 1.13.2），
  与 `GUIDE.md` 第 2 节一致；
- 所有命令都在**仓库根目录**执行（代码里的路径都是相对路径）。

```bash
conda activate ai
python -c "import numpy, casadi, GPy; print('deps ok')"
```

## 脚本一览

| 脚本 | 作用 |
| --- | --- |
| `play/pipeline_sweep.py` | 一条龙：名义 MPC 录制 → 构造训练文件 → GPy 训练 → 闭环消融 |
| `play/dump_trajectories.py` | 用已训练的模型重放指定工况，导出轨迹到 `log/sweep/traj/*.npz` |
| `play/make_figures.py` | 读上面的 npz，画轨迹图与误差图（矢量 PDF） |
| `play/eval_residual.py` | 残差预测精度评估（同一工况下噪声开/关），并画残差图 |
| `play/sweep_closed_loop.py` | 只用已有模型跑消融矩阵（不做录制/训练） |
| `play/probe_wheel.py` | 轮速效率 η_L 的敏感性探测（名义 MPC 的横向误差 vs η_L） |
| `play/probe_gp_alignment.py` | 检查 GP 预测与残差的时间对齐（决定误差指标怎么定义） |
| `play/_agg.py` | 把 `log/sweep/*_ablation.csv` 汇总成均值±标准差 |

## 典型流程

```bash
# 1) 延迟工况：录制 + 训练 + 消融（3 个随机种子）
python -m play.pipeline_sweep --scenario delay --tag delay3 --repeats 3

# 2) 单侧驱动力不足工况（论文用 η_L = 0.925）
python -m play.pipeline_sweep --scenario bad_wheel --tag bad_wheel925 \
    --wheel-left 0.925 --repeats 3

# 3) 汇总统计
python -m play._agg

# 4) 导出轨迹并画图（--outdir 指向论文仓库的 figure/gpmpc）
python -m play.dump_trajectories --scenario delay --tag delay3
python -m play.dump_trajectories --scenario bad_wheel --tag bad_wheel925
python -m play.make_figures --outdir ../zjuthesis/figure/gpmpc

# 5) 残差预测精度（表 4.2 与图 4.5/4.8）
python -m play.eval_residual --scenario delay --params-tag delay3 \
    --figure-dir ../zjuthesis/figure/gpmpc
python -m play.eval_residual --scenario bad_wheel --params-tag bad_wheel925 \
    --figure-dir ../zjuthesis/figure/gpmpc
```

## 产物

| 路径 | 内容 |
| --- | --- |
| `log/record/<tag>.txt` | 名义 MPC 录制的原始数据：`Δv Δω v ω a δ`，每步一行 |
| `log/processed_data/<tag>.txt` | GP 训练文件：前 N 行是“特征(12)+响应(2)”，后 2M 行是两组诱导点 |
| `log/sweep/<tag>_ablation.csv` | 每个（配置 × 种子）一行的控制精度 |
| `log/sweep/<tag>_params.txt` | 训练得到的超参数（可粘回 yaml 的 `gaussian.params`） |
| `log/sweep/residual_<scenario>.csv` | 残差预测误差 |
| `log/sweep/traj/*.npz` | 轨迹、误差、控制量与参考路径 |
| `log/sweep/mpc_*.txt` | MPC 调试日志（每步的实测残差、GP 预测、回归量） |

## 指标定义

- **控制精度**：`mean|d|`、`max|d|`（横向误差）、`mean|θe|`（航向误差），
  由 `SimulationRunner.state` 逐步记录后统计；每格取 N 个种子的均值。
- **时间**：单步 `solve + 传播` 的墙钟时间（不含绘图），`ms/step`。
- **残差预测误差**：名义预测取“残差恒为零”，误差为 `mean|d_k|`；
  GP 预测取训练后验在日志回归量 `z_k` 上的均值，误差为 `mean|d_k − μ(z_k)|`；
  改善幅度 = `1 − GP误差/名义误差`。

## 配置要点（与论文 4.6.2.1 对应）

| 项 | 取值 | 说明 |
| --- | --- | --- |
| `mpc.R` | `diag(0.1, 0.1)` | yaml 默认是 0.2，会改变闭环稳定性，脚本里强制覆盖 |
| `mpc.max_input` | `[1.0, 1.5]` | 线加速度 / 角加速度上限 |
| `sim.delay_window` | 2 | 100 ms 固定延迟 |
| `sim.bad_wheel` | true, `wheel_left=0.925` | 单侧驱动力不足；**0.925 才能复现论文基线**（0.95/0.975 偏差较大） |
| `gaussian.input_window` | 3 | 回归量 12 维 |
| `gaussian.inducing_data_number` | 40 | 初始诱导点数；在线扩充上限为 10 倍 |
| `sim.add_noise` | true | 观测量叠加 σ = 0.01 的高斯噪声 |

## 注意事项

- 训练数据必须**来自同一工况**：`pipeline_sweep` 先跑名义 MPC 录制，再用录制数据训练；
  用别的工况的数据训练会导致残差预测几乎无效。
- 录制时 `gpr_method=-1`（名义控制器），记录的是“真实系统 − 名义模型”的残差。
- 在线更新开关：`update_mu_su`（后验递推）、`add_mu_su`（诱导点扩充），
  判定条件在 `controller/linear_mpc.py::update_gp_model`。
- 所有脚本都用 `np.random.seed(seed)` 固定随机性；换种子即可得到不同实现。