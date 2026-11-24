# 来自 Python GPRController 的跨项目交接经验

> 适用对象：C++ / ROS 版本的 GPRController（实车 + Gazebo）。
> 性质：**同级项目的踩坑记录与审计方法**，不是 C++ 实现的权威说明。
> 原则：所有结论都要在 C++ 仓库重新验证；如果与 C++ 代码/文档冲突，以 C++ 仓库为准。

---

## 0. 怎么使用这份文档

推荐做法：

1. 把本文件复制到 C++ 仓库：`docs/agent/FROM_PYTHON.md`；
2. 在 C++ 仓库根目录的 `AGENTS.md` 中加一段“先读这份文件”的引用（见 `AGENTS.snippet.md`）；
3. 之后每次在这个 C++ 工作区新开会话，`AGENTS.md` 会自动加载，并指示 agent 先读本文件；
4. 在 C++ 仓库里维护自己的 `docs/agent/FINDINGS.md`、`BASELINE.md`、`HANDOFF.md`——本文件只提供“可能的坑”和“怎么查”，不记录 C++ 项目的实时状态。

不要：

- 把本文件当成 C++ 代码规范；
- 直接照搬 Python 的公式、命名、采样率、坐标系或参数默认值；
- 没复现就修改实车行为。

---

## 1. 源项目是什么

Python 版本是一个 **GPR（高斯过程回归）增强的线性 MPC 轨迹跟踪控制器**：

- 车辆运动学模型：Cartesian（`x, y, yaw, v, w`）与 Frenet（`e_y, e_phi, v, w`）；
- 支持延迟模型（一阶速度/角速度滞后）和“坏轮子”模型；
- 线性 MPC：CasADi + OSQP/QPOASES，支持热启动、控制增量约束、终端权重；
- GP：零阶（只补偿均值）与一阶（均值 + 梯度进入 A/B/C）；
- 稀疏 GP：DTC / FITC，支持在线更新和快速添加诱导点；
- augmented 状态：把 `input_window-1` 步的 `(v, w, u_v, u_w)` 历史拼进状态；
- 训练/保存/加载：GPy 训练、模型缓存、数据记录、日志指标；
- 闭环仿真：Cartesian 仿真模型 + 噪声/滤波/延迟队列，可作为 Gazebo 场景的对照。

对应的 C++/ROS 版本通常会把上述能力拆成多个包/节点。**能力边界比文件名更重要**。

---

## 2. 组件映射（用于定位 C++ 代码）

| Python 模块 | C++ / ROS 中大概率对应 | 备注 |
|---|---|---|
| `controller/linear_mpc.py` | MPC/控制器节点、controller plugin、QP 组装与求解 | 重点看状态维度、约束、solver 状态 |
| `model/frenet_model.py`、`cartesian_model.py` | 车辆运动学/动力学模型、状态预测器 | 单位、坐标系、积分器 |
| `model/*_gp_model.py` | GP 推理模块、学习型补偿器、模型加载 | 均值/梯度、缓存、线程安全 |
| `train/train_gp.py` | 离线训练工具、标定/模型导出 | 训练与在线特征必须一致 |
| `trajectory/` | 路径/轨迹服务器、样条/回放 | 弧长、曲率、yaw 连续性 |
| `play/mpc_closed_loop.py` | Gazebo 驱动、rosbag 回放、实车实验脚本 | 时间源、延迟、噪声 |
| `utils/config.py` | ROS 参数、launch 参数、YAML 配置 | 默认值、校验、覆盖关系 |
| `utils/filter.py` | 低通/平滑滤波器 | 初始化、截止频率、Nyquist |
| `utils/log.py` + `log/` | ROS logger、rosbag、指标 CSV/JSON | 可观测性、回放 |
| `test/` | gtest、launch_testing、bag replay 回归 | 单元 + 集成 + 场景 |
| `run.py` | 实验编排脚本 | 多次 run、保存/加载 |

---

## 3. 已经踩过的 bug 类型 → 在 C++/ROS 里怎么查

下面每一条都来自 Python 版本的真实修复。C++ 版不一定会犯同样的错，但**检查方法是通用的**。

### 3.1 滤波器“看起来有，实际没有历史”

**Python 现象**：二阶 Butterworth 低通的输入/输出 deque 用 `[] * 3` 初始化，实际是空 deque；每次 `pop` 再 `appendleft`，长度永远是 1。滤波器退化成“当前输入 × 增益”，阶跃输入输出恒为 `0.226`，稳态永远到不了 1。

**根因**：状态缓冲区没有初始化为 N 个 0；只测了“有没有输出”，没测阶跃/频响。

**C++/ROS 检查**：

- 找到所有 `std::deque`、`std::array`、环形缓冲区、`Eigen::VectorXd`、`std::vector::resize` 的滤波器状态；
- 构造后是否 `setZero()` / 填 0 / 用最近 N 个真实样本预热；
- 检查 `pop_front/pop_back` 与 `push` 的方向，历史索引是否越界；
- 对阶跃输入验证稳态增益为 1、−3 dB 截止频率符合配置；
- 检查截止频率是否低于 Nyquist：`fc < 1/(2*dt)`；ROS 参数里 `dt` 和 `fc` 的单位是否一致。

**验证**：单位阶跃 200 个样本，最后 10 个样本均值≈1；前几个样本不等于 `b0`；与 scipy/MATLAB 或解析系数对拍。

---

### 3.2 历史窗口 / 缓冲区尺寸只在某个 `W` 下刚好成立

**Python 现象**：augmented 状态的控制缓冲区分配为 `(W-1)*3`，而 `change_vector_to_model2` 需要的长度是 `2 + 2*(W-1)`。两者只在 `W=3` 时相等：

- `W=2`：多出的维度被 NumPy **静默广播**，最后一行补 0，模型状态错误；
- `W>=4`：直接 `ValueError: cannot broadcast (7,1) into (6,1)`；
- 修复：控制缓冲区改为 `2*W`，并在拼接处断言 `state_size + control_history == nx`；补 `W=2/4` 端到端测试。

**C++/ROS 检查**：

- 找到 fixed-size arrays、`std::array<T, N>`、`Eigen::Matrix<double, N, 1>`、`Eigen::Block`、环形历史缓冲区；
- 列出维度恒等式并写成 `static_assert` 或运行时断言：
  - `nx = base_nx + 4*(W-1)`
  - `history_state = 2*(W-1)`
  - `history_control = 2*(W-1)`
  - `base_nx + 2*(W-1) + 2*(W-1) = nx`
- 检查 `W=2..8` 的构造/初始化/预测是否都成立；
- 警惕“刚好 W=3 的默认配置能跑”，因此测试永远只覆盖默认值。

**验证**：写一个 `W` 参数化测试；断言每一维都被写入（例如初始化为 NaN 后检查无 NaN 残留）；用 `Eigen::Map` 做尺寸检查。

---

### 3.3 GP 特定字段被无条件访问（没有 GP 的模型也走 GP 路径）

**Python 现象**：`LinearMPC.solve()` 无条件访问 `x_guess_gpr`、`gpr_input_step_1`、`model.gaussian_model`；普通运动学模型直接 `AttributeError`。另外 Cartesian GP 模型收到不认识的 `m_curvature` 参数，直接 `TypeError`。

**C++/ROS 检查**：

- 控制器是否假设“一定有 GP 对象/学习模块/缓存”？检查 `std::optional`、指针、`shared_ptr`、variant、能力 flag；
- 无 GP、零阶 GP、一阶 GP 三种模式是否都能构造、初始化、跑一步？
- 模型工厂是否把“模型族”与“是否 GP/是否 augmented/是否 Frenet”混在一个字符串/枚举里；
- 插件加载失败时是否有明确的 fail-fast，而不是空指针/默认对象。

**验证**：对每种模型组合做 `initialize + solve` 冒烟测试；无 GP 时不能访问 GP 日志/指标字段。

---

### 3.4 用字符串/名称做分发，改名即静默走错分支

**Python 现象**：`"Gaussian" in model.name`、`model.name == "Kinematic Frenet"` 等散落在控制器；模型改名/加前缀后，分支静默跳过。

**C++/ROS 检查**：

- 是否存在 `typeid(...).name()`、`dynamic_cast`、插件 ID/字符串比较、`if (name == "...")`；
- 是否有工厂注册表？未知类型是否报错，而不是 fallback 到默认模型；
- 能力查询是否用虚函数/接口（`supports_gp()`、`needs_curvature()`）而不是名字匹配。

**验证**：注册一个未知模型，确认构造失败且错误信息包含可用选项。

---

### 3.5 配置/参数缺失、默认值不一致、未知键被忽略

**Python 现象**：`Config.__post_init__` 在缺少 `sim` 段时访问 `self.sim.control_dt` 崩溃；`from_yaml` 静默忽略未知顶层 section；`log_path` 曾硬编码。

**C++/ROS 检查**：

- `declare_parameter` 是否全部有默认值？默认值是否与 launch/YAML 一致？
- 必填参数缺失时是否在启动阶段 fail-fast，而不是运行到一半用 0/NaN；
- 参数范围/单位/枚举校验（控制周期、时域、最大速度、`fc < Nyquist`）；
- launch 参数 → node 参数 → 配置文件 的覆盖链是否清楚；
- `ros2 param dump /node` 是否能看到实际生效值？sim 与 real 的 dump 是否一致？

**验证**：用最小参数集启动；删除必填参数应报错；修改参数后 dump 确认；未知参数是否被忽略/警告。

---

### 3.6 模型/标定文件缺少元信息，静默用错

**Python 现象**：旧格式按固定顺序 `np.save` 数组，没有保存 GP 类型、核超参、`normalize_y`、特征维度；加载后可能与当前配置不一致但照样预测。修复：带版本/元数据的 npz；加载时校验 GP 类型、核参数、normalize、特征维度、Cholesky 有效性；保留旧格式兼容读取；不匹配直接报错。

**C++/ROS 检查**：

- 模型/标定/权重文件是否有：格式版本、生成时间、git SHA、配置哈希、特征窗口、归一化参数、稀疏方法、GP 类型；
- 加载时是否校验维度/归一化/方法；不匹配是报错还是静默继续；
- 文件写入是否原子（临时文件 + rename），避免掉电/中断写坏；
- 训练与在线推理的特征顺序、窗口、单位是否完全一致。

**验证**：保存→加载→预测数值一致；改一个超参/归一化后加载应报错；损坏文件应报错；旧版本文件有迁移策略。

---

### 3.7 在线更新与缓存分解不一致（quick vs Cholesky）

**Python 现象**：`add_new_inducing_quick` 只更新 `mu/Su`，故意不更新 `Kuf/W/diag/L`；之后 `predict_cholesky` 形状不匹配，`update()` 还会尝试重建旧缓存而崩溃。修复：显式 `cholesky_valid` 标记；快速路径只允许 `mu_su` 预测；需要 Cholesky 时 `sync_cholesky()`；`update_L()` 在无效状态下 no-op；保存时只写有效状态。

**C++/ROS 检查**：

- 是否有“增量更新后仍复用旧分解/旧增益/旧矩阵”的路径？
- 缓存失效标志是否覆盖所有修改点（加诱导点、加数据、删数据、改超参、改归一化）；
- 是否混用两套预测器（例如控制器用 A，监控用 B）？
- 多线程下缓存读写是否有锁/双缓冲/版本号？

**验证**：更新后立即预测；故意让缓存失效，确认走重建路径；保存/加载后再预测；并发压测（如适用）。

---

### 3.8 求解器失败被静默接受

**Python 现象**：`error_on_fail: False`，求解后不检查状态/NaN；失败解可能进入控制量。修复：检查 `solver.stats()["success"]`、`return_status`、控制量有限性；失败抛 `RuntimeError`。

**C++/ROS 检查**：

- OSQP/QPOASES/HPIPM/其他 solver 的返回码是否检查？超时、最大迭代、不可行、数值错误分别怎么处理？
- 失败时是保持上一帧控制、限幅、进入安全状态，还是继续用垃圾解？
- 是否检查 `NaN/Inf`、控制量越界、约束违反？
- 是否统计成功率/失败率并告警？

**验证**：人为构造不可行约束/坏 QP；确认 fail-safe 行为；监控日志中状态码分布。

---

### 3.9 时间、单位、坐标系

**Python 现象**：`control_dt` 与 `mpc_dt` 混用；Frenet 与 Cartesian 的 yaw/曲率约定；滤波器截止频率高于 Nyquist；`sim_time` 与 wall time 混用。

**C++/ROS 检查**：

- 所有 `dt` 是仿真时间还是墙钟？`use_sim_time` 是否一致？
- rad/deg、m/mm、m/s 与 km/h、曲率与角速度的换算；
- `frame_id` 与 TF 链；`map/odom/base_link` 方向；变换的时间戳；
- 控制周期、传感器周期、MPC 时域步长是否一致；
- 实车与 Gazebo 的单位/坐标系是否一致（尤其 IMU/轮速）。

**验证**：直线 + 定半径圆；`tf2_echo` 检查静态/动态 TF；bag 回放时检查 `/clock` 与时间戳；打印关键量纲。

---

### 3.10 过程噪声离散化缩放

**Python 现象**：`Q/dt` 与 `Q*dt` 混用；`Q` 是连续时间协方差率还是每步方差没有文档。修复为每步协方差 `Q*dt`，用平方根分解采样，并补蒙特卡洛测试。

**C++/ROS 检查**：

- 过程噪声/测量噪声参数是连续谱密度还是离散方差？代码里乘/除 `dt` 的位置；
- `dt` 变化时噪声幅度是否按预期缩放；
- 是否用 Cholesky/平方根采样，协方差是否半正定；
- 仿真与实车的噪声标定是否分开。

**验证**：固定 `Q`，改变 `dt`，蒙特卡洛估计每步协方差；零 `Q` 应无噪声。

---

### 3.11 标量/矩阵形状与 Eigen 陷阱

**Python 现象**：误差函数返回 `(1,)` 数组而不是 `float`，触发 NumPy 弃用警告；预测轨迹形状 `(N*nx,1)` 与 `(N,5)` 不一致，绘图/下游崩。

**C++/ROS 检查**：

- Eigen `VectorXd` vs `MatrixXd`、行/列向量、`1x1` 矩阵被当成标量；
- `auto` 表达式模板悬垂/别名：`a = a.transpose()` 需 `.eval()`/`.noalias()`；
- 维度断言（`assert`/`Eigen::MatrixBase::rows()` 检查）；编译开 `-Wconversion -Wshadow`；
- 数组/消息字段的类型与长度；`std::vector` 与 Eigen 映射的生命周期。

**验证**：ASan/UBSan 仿真构建；单元测试覆盖 1 维/多维；形状/维度断言在 debug 构建中开启。

---

### 3.12 热路径：每周期重建、重复计算、分配

**Python 现象**：一阶 GP 对每个 horizon 步、每个特征维度调用 GPy 核导数，占 83% 时间（约 158 ms/步）。修复：解析 RBF 梯度、缓存 solver、三角求解，降到约 31 ms/步（同一台机器，仅参考）。

**C++/ROS 检查**：

- 控制循环内是否 `new/malloc`、`std::vector` 扩容、字符串格式化、日志、异常；
- 是否每个周期重建 QP/矩阵/分解/solver，而不是复用工作区；
- Eigen 是否产生临时量（`noalias`、预分配）；
- 编译器优化/`-ffast-math`（控制代码慎用）、`-march`、LTO；
- 用 `perf`、`ros2_tracing`、循环超时计数、solver 计时定位；
- 实时线程优先级/CPU 亲和性/内存锁（如适用）。

**验证**：记录 p50/p95/p99 周期时间、超时次数、CPU；优化前后同场景对比；确认数值结果在容差内不变。

---

### 3.13 最近点/赛道索引/自交赛道

**Python 现象**：每步对整条路径做全局最近点搜索，O(N) 且自交赛道可能跳到错误分支；改为上一 index 附近窗口 + 距离过大时全局回退，并返回标量误差。

**C++/ROS 检查**：

- 最近点搜索是否单调/局部窗口？自交或发卡赛道是否会跳变？
- 投影是“最近点”还是“沿路径投影”？横向误差符号是否一致？
- 圈数/索引回绕；路径终点保持；重定位/初始化时的索引。
- 地图/路径更新时索引是否重置。

**验证**：8 字/自交/发卡路径；初始化在路径中段；检查 index 连续性/误差符号；与解析圆对比。

---

### 3.14 日志与可观测性

**Python 现象**：自定义日志格式缺 `]`，关闭/刷新依赖 GC；日志解析脚本与格式耦合。修复：标准 `logging`、自动建目录、flush/close、可配置路径，并用解析测试保证格式。

**C++/ROS 检查**：

- 控制循环里是否每周期打日志（分配/阻塞）？用 `RCLCPP_*_THROTTLE`/采样；
- 日志字段是否结构化（时间、状态、控制、solver 状态、延迟）；
- rosbag 是否记录输入/输出/参数快照/git SHA；
- 日志/指标解析脚本是否有回归测试。

**验证**：bag 回放后离线解析；字段缺失/格式变化应测试失败。

---

## 4. C++ / ROS 专项检查清单

### 4.1 构建与静态检查

```bash
# ROS 2 示例
colcon build --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo \
               -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
colcon test --packages-select <pkg> --event-handlers console_direct+
colcon test-result --verbose
```

- 编译警告：`-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wdouble-promotion`；
- `clang-tidy -p build`（或 `run-clang-tidy -p build`）；
- 仿真构建加 ASan/UBSan（**不要上实车**）：
  `-DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"`；
- 检查 `compile_commands.json` 中是否包含所有包；
- 确认没有用 `-ffast-math` 破坏控制/估计的数值语义。

ROS 1 对应：`catkin build` / `catkin_make`、`catkin_make run_tests`、`rostest`。

### 4.2 ROS 接口与参数

```bash
ros2 node list
ros2 node info /<node>
ros2 topic list -t
ros2 topic hz /<topic>
ros2 topic bw /<topic>
ros2 param dump /<node>
ros2 interface show <pkg>/msg/<Msg>
ros2 run tf2_ros tf2_echo map base_link
```

检查：

- topic 类型/QoS/队列深度/发布频率；
- `use_sim_time` 与 `/clock`；仿真/实车切换；
- 参数默认值与 launch 覆盖；未知参数；
- TF 是否存在、时间戳是否同步、frame_id 是否拼写正确；
- 消息时间戳是传感器时间、接收时间还是控制周期时间。

ROS 1：`rostopic hz`、`rosparam dump`、`rosrun tf tf_echo`、`rosbag`。

### 4.3 实时与并发

- 控制回调是否在实时线程？executor/callback group 是否互相阻塞？
- 互斥锁、条件变量、无锁队列；是否在持锁时做重计算/日志/分配？
- 传感器回调与控制循环共享数据时是否双缓冲/原子交换？
- 看门狗：控制指令超时、传感器断流、solver 超时；
- 异常/内存分配失败时的行为。

### 4.4 数值与安全

- NaN/Inf 检查；单位/边界；积分器稳定性；
- 控制量/增量限幅；抗积分饱和；
- 失败时保持/减速/停车策略；
- 实车默认禁止自动运行；急停、限速、安全员。

### 4.5 仿真 vs 实车

- 同一段轨迹/同一组参数分别跑 Gazebo 和实车，记录对比；
- 传感器噪声、延迟、轮速/IMU 偏差、执行器滞后；
- 先在仿真/bag 复现，再到实车；
- 用 golden bag 做确定性回放回归。

---

## 5. 推荐工作流

### 5.1 首会话 Bootstrap（60–90 分钟）

1. 记录 git SHA、分支、ROS 发行版、构建类型、工作区依赖；
2. 干净构建，记录警告数量；
3. 跑全部测试，记录失败/跳过；
4. 找到或录制一条 golden bag，回放一次，确认能复现基线轨迹；
5. 记录基线指标（见 `BASELINE` 模板）；
6. 运行静态检查（clang-tidy/ASan 仅仿真）；
7. 阅读 `FINDINGS/HANDOFF`，列出 top 3 风险；
8. 写入新的 `HANDOFF.md`。

### 5.2 审计循环

1. **复现**：命令 + bag + launch 参数 + 期望/实际；
2. **定位**：代码路径/日志/时序/参数；
3. **最小修复**：一次只改一个根因；
4. **回归测试**：gtest/launch test/bag replay；
5. **指标对比**：跟踪误差、solver 时间、超时、CPU；
6. **记录**：FINDINGS 状态、HANDOFF 下一步。

### 5.3 结论等级（避免过度声明）

- E0：读代码猜测；
- E1：静态分析/编译警告；
- E2：仿真或 bag 复现；
- E3：实车复现/验证。
只有 E2/E3 才能说“已确认”；E0/E1 写成“怀疑/待验证”。

---

## 6. 记录模板

### 6.1 `FINDINGS.md`

```markdown
| ID | 严重度 | 区域 | 现象 | 复现 | 证据 | 根因 | 修复/测试 | 状态 |
|----|--------|------|------|------|------|------|-----------|------|
| F-001 | P1 | controller | 转弯时横向误差突增 | bag X + launch Y | 日志/时序 | 待定 | - | open |
```

详细条目：

```text
F-001 [P1][controller] 标题
Symptom:
Expected:
Actual:
Repro: 命令 + bag + 参数 + git SHA
Evidence: 日志/话题/时序/截图
Root cause:
Fix + regression test:
Status: open / investigating / fixed / wontfix
```

### 6.2 `BASELINE.md`

```markdown
| 日期 | git SHA | 场景 | 跟踪 RMSE | 最大误差 | solver p50/p99 | 超时次数 | QP 成功率 | CPU |
|------|---------|------|-----------|----------|----------------|----------|-----------|-----|
```

### 6.3 `HANDOFF.md`

```markdown
# Current state
- branch/SHA:
- ROS distro / build type:
- last baseline:
- open findings (top 3):
- current hypotheses:
- next actions:
- environment caveats:
```

---

## 7. 不要照搬的假设

- Python 的 Frenet 误差定义、曲率符号、yaw 归一化；
- GP 的特征顺序/窗口/归一化方式；
- Python 的控制周期、`mpcWindow`、权重、约束；
- “默认配置能跑”不等于“所有 `input_window`/模型组合都正确”；
- 仿真结果不等于实车结果；
- Python 的性能数字（158→31 ms）只说明优化方向，不代表 C++ 的预算；
- Python 的日志/模型文件格式。

---

## 8. 高收益检查 Top 10

1. 滤波器阶跃响应（历史是否真的存在）；
2. 历史窗口/矩阵维度恒等式（`W` 扫描 + 断言）；
3. 无 GP / 零阶 / 一阶 / augmented 全组合冒烟；
4. `ros2 param dump` 与 launch 默认值对照；
5. 模型/标定文件版本与兼容性校验；
6. 在线更新后的缓存失效标志；
7. solver 返回码、NaN、超时、fail-safe；
8. TF/时间源/单位（rad/deg、m/mm、sim/real）；
9. 控制循环内的分配/锁/日志；
10. bag 回放确定性与 sim/real 指标对比。

---

*本文件来自 Python GPRController 的 P0–P2 修复经验；在 C++ 仓库中使用时，请把“检查方法”当作起点，把“结论”留给证据。*
