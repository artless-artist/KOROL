# KOROL 复现尝试（Door 任务）：未跑通的归档

对 KOROL（*Learning Visualizable Object Feature with Koopman Operator Rollout for Manipulation*, CoRL 2024）的一次复现尝试，2026 年 10 月完成。**结论是没有跑通**：Koopman 算子学得不差，但用来驱动机械臂的逆动力学控制器精度不足以把门推开，闭环成功率恒为 0。

代码与环境配置原样保留，失败归因和踩过的坑写在下面。归档的目的不是展示成果，是让下一次（或别人）不用从头踩一遍。

> 本仓库不含论文官方数据集（原 Google Drive 链接已清空），也不含本次跑出的权重、渲染缓存和视频（约 2 GB，见「未归档的产物」一节）。

三个关键数字：

| 指标 | 结果 | 说明 |
|---|---|---|
| Koopman 开环误差（69 步手部状态平均绝对误差） | **0.0582** | 真值线 0.0569，同一量级，算子这一环成功 |
| 逆动力学控制器单步 MSE | **0.0126** | 判据要求 < 1e-3，RMSE 0.112 ≈ 动作 std 的 21% |
| 闭环成功率（4 次评估 × 20 个初值 × 70 步） | **0.000000** | 但评估窗口本身不足以开门，见 §3.5 |

---

## 1. 论文在做什么

一句话：用 ResNet18 从 RGBD 里学一个 8 维、可解释的物体特征，和 28 维手部关节状态拼成系统状态，再用 Koopman 算子在提升空间里做线性预测，最后由一个逆动力学网络把「下一步手该在哪」翻译成关节动作。

| 环节 | 实现 | 维度 |
|---|---|---|
| 视觉编码 | ResNet18（`utils/resnet.py`），输出 `feat_dim=8` | 8 |
| 系统状态 | 手部关节角 + 物体特征 | 28 + 8 |
| 字典提升 | `utils/Observables.py` 的 `DraftedObservable(28, num_obj)`，2 阶多项式 + 3 阶单项式 | **570**（num_obj=8）／**546**（num_obj=7，真值线） |
| 算子求解 | EDMD，`K = A · G⁺`（`train_koopman`），闭式最小二乘，**没有梯度、没有 epoch** | 570×570 |
| 闭环控制 | `z_{t+1} = K z_t` → 取前 28 维当目标位姿 → 控制器输出 28 维动作 | 28 |
| 交替训练 | 固定 K 训 ResNet（L1 损失，在 `--N` 步 rollout 上），每 50 轮重提特征、重解 K | — |

DraftedObservable 的维数公式（`Observables.py:109`）：
`2(h+o) + o(o-1)/2 + h + o² + h(h-1)/2`，h=28。
代进去：o=8 → 570，o=7 → 546；Tool/Relocate/Reorientation 的 o=6 但 h 不同，分别是 466/588/411。

---

## 2. 复现路径与实际差异

| 环节 | 论文／官方仓库 | 本次做法 | 后果 |
|---|---|---|---|
| 数据集 | Google Drive 链接。Door 训练脚本里写 `num_demo = 25`，但 BC 基线脚本里写 `num_demo = 200` | Drive 已清空，改为从 `mj_envs/robohive/hand_dapg/dapg/demonstrations/door-v0_demos.pickle` 就地重建（25 条、6729 帧） | **训练数据可能只有官方的 1/8**，无法验证 |
| 控制器 `NN_controller_best.pt` | 仓库里 31 个文件引用它，全是 `Controller.load_state_dict(torch.load(...))`，**没有任何一处把它写出来**，也没有训练脚本 | 按其网络结构 `FCNetwork([56,112,112,56,28])` 用示范自训逆动力学 BC（56 = hand_t ⊕ hand_{t+1} → 28 = action_t），6704 个样本、60 轮 | **失败的直接来源**，详见 §3 |
| 物体状态 | 真值线用 `objpos(3) + objvel(1) + handle_init(3)` = 7 维；KOROL 线用 8 维特征 | 两条线都跑通，真值线 546、KOROL 线 570 | 口径不能混用，K 不能互换 |
| 闭环评估 | `horizon = 70`，判据 `door_pos > 1.35` | 完全沿用 | 信号恒为 0，掩盖了所有其他指标 |

数据重建的做法（`Korol/my_demos_build.py`）：`demo_loader` 里回放时 `e.step()` 是注释掉的，说明**数据管线只依赖确定性回放和渲染，不依赖任何外部文件**。所以只要 `init_state_dict` 和 `actions` 还在，就能重建出和官方等价的 `observations_visualization` 与 `rgbd_*.npy`。回放 25 条示范得到 6729 帧 256×256×4 的 npy（RGB 归一化 `(uint8−128)/128`，depth 作第 4 通道）。

---

## 3. 失败归因

按「可干预程度」排序，不是按影响大小。

### 3.1 控制器从哪来，是这次复现的硬卡点

仓库提供了加载 `Door/controller/NN_controller_best.pt` 的代码，但没有产生它的代码。`Korol/baselines/BC_door_training_gt.py` 是**另一条基线**（学状态转移，输入 `hand+objfeat`、输出 `next hand+objfeat`），它自己也要 load 作者的 controller 来跑闭环。

我们按其网络结构反推：56 维输入 = 当前手部 28 维 + 目标手部 28 维，28 维输出 = 动作，本质是逆动力学。用 25 条示范的全部相邻帧对训练（每条少最后一步，6729 − 25 = 6704 个样本），60 轮后 MSE 0.2308 → 0.0126。

这条路径能不能替代作者的 controller，事前无法判断，事后看是不能。

### 3.2 控制器单步精度差一个量级

动作实测：范围 −1.2788 ~ 1.3662，均值 −0.099，**std 0.5463**。

| MSE | RMSE | 占动作 std | 状态 |
|---|---|---|---|
| 1e-3 | 0.032 | 6% | 合格线 |
| **0.0126** | **0.112** | **21%** | 本次结果 |
| 1e-1 | 0.32 | 58% | 等于没学会 |

21% 的单步误差在闭环里会累积。同一套数据上算 K 的开环误差：第 1 步 0.006，第 69 步 0.118——**误差基本线性增长，69 步后比示范中单帧的手部变化量（0.01 量级）大一个数量级**。K 本身的漂移不算离谱，但控制器每一个单步的 21% 会顺着这条链路放大。

### 3.3 训练和推理的输入分布不一致

`utils/Koopman_evaluation.py:139` 那一行是关键：

```python
current = z_t[:num_hand]   #e.get_env_state()['qpos'][:28]  # current state
```

训练时 `robot_current` 是示范里的**真实** handpos；闭环时喂给控制器的却是 **K 预测出来的** handpos。作者的注释里留着实测写法，但被注释掉了——原论文就是纯开环设定。于是控制器在推理时看到的输入分布和训练时不同，误差越大、偏移越远，输出越不可靠。

这条是设计选择，不是 bug。但它意味着**控制器必须能容忍目标噪声**，而单纯回归示范的 BC 恰好看不到这种噪声。

### 3.4 回归损失只能学到「平均动作」

逆动力学是多模态的：同一个「从 A 到 B」，人类可以有多种走法。L1/L2 回归对这些解取平均，落到闭环里表现为「手朝一个折中姿势飘、门纹丝不动」。

判断依据不该是 MSE，而是逐维相关系数与均值基线的对比（`tools/check_controller.py`）。本次结果：相关均值 > 0.8，判「可用」——**但闭环仍然失败**。这本身是一条教训：单步回归指标好，不等于闭环可用。

### 3.5 评估口径让所有信号变成 0

`horizon = 70` 意味着每个回合只跑 69 步。这套示范的时间尺度是：

| 步号 | 20 | 40 | 60 | **69** | 100 | 140 | 170 | 200 |
|---|---|---|---|---|---|---|---|---|
| 示范 door_pos 均值 | 0.012 | 0.012 | 0.011 | **0.011** | 0.010 | 0.031 | 0.415 | 0.923 |

- 第 69 步门才转了 0.011 rad，手刚摸到把手（掌-把手距离从 0.379 m 收到最小值 0.045 m，也就是全程最接近把手的位置）。
- `door_pos` 首次超过 0.2 是第 140 步（中位 171），超过 1.0 是第 157 步，**超过 1.35 最早第 166 步**。
- 25 条示范里**只有 7 条**最终能到 1.35，其余到 300 步结束也只开到 0.8~1.3。

也就是说：把人类示范原样回放 70 步，成功率同样是 0。`door_pos > 1.35` 在 70 步窗口里物理上不可达。

这条不是失败的原因，但它让「0.000000」这个数字完全失去信息量——四次评估全 0，从日志上分不出「模型差」还是「窗口短」。**修评估口径应该排在修模型前面。**

---

## 4. 如果重做，优先动这几处

1. **先把 horizon 改到能开门**（≥ 200 步），并让评估函数输出逐回合 `door_pos` 曲线而不是 0/1。`tools/eval_door_curve.py` 就是干这个的。
2. **闭环里改用实测 qpos 做 `current`**（把 §3.3 那行注释掉的部分打开），先把控制器和 K 的能力与「纯开环漂移」解耦；确认能开后，再回去评纯开环，作为消融。
3. **控制器换掉**：Diffusion Policy / ACT / IRL 这类能表达多模态的，或者用 DAgger 在闭环里补数据。仓库的 `baselines/Diffusion_*` 有可参考的结构。
4. **数据**：官方 Door 数据到底多少条无法确认。若确实是 200 条，那 25 条只够跑通链路，不够训出可用策略；补数据只能自己采（DAPG 的 door 只有 25 条）。
5. **保存权重**：`Door_optimal_feature_training.py` 里一行 `torch.save` 都没有，ResNet 权重随进程消失，570 维闭环事后无法复评。

---

## 5. 环境

WSL2 Ubuntu 26.04，conda 环境 `mjrl-env`（Python 3.8）：

| 组件 | 版本 | 备注 |
|---|---|---|
| mujoco | 2.1.0（`~/.mujoco/mujoco210`） | 必须 2.1.0，配 mujoco-py 2.1.2.14 |
| mujoco-py | 2.1.2.14 | 首次 import 时编译；Cython 报错就 `pip install "cython<3"` |
| gym | 0.13 | **不会自动加载 entry_points**，见 §7 第 1 条 |
| pytorch | 2.3.1 | 论文给的是 2.3.1+cu121 |
| mjrl / mj_envs | 仓库内源码，`pip install -e .` | `mj_envs` 提供 `robohive` 包 |

`install.txt` 是完整的安装顺序，`check_env.py` 用来一次性确认依赖是否齐全（含 mujoco_py 是否真的编译成功）。

```bash
conda activate mjrl-env
python check_env.py
```

---

## 6. 从零复跑

```bash
# 1. 环境（见 install.txt），然后自检
python check_env.py

# 2. 重建数据：回放 DAPG 示范 + 渲染 RGBD，产物在 Korol/Door/Data/
cd Korol
python my_demos_build.py          # 25 条示范 → 6729 帧 256×256×4 npy + Testing.pickle
python my_demos_test.py           # 自检：条数、键、图像范围

# 3. 真值线（KODex）：先确认闭环管线本身是通的，这一步不需要 ResNet
python Koopman_gt_operator_training.py     # 默认 door_main()，产出 gt_koopmanMatrix.npy 546×546

# 4. 控制器（本项目自训的替代品，非官方权重）
mkdir -p Door/controller
python ../tools/train_controller.py        # 6704 样本，60 轮，产出 NN_controller_best.pt

# 5. KOROL 线：先关评估，只验证能训能存
#    把第 137 行 epoch % 50 临时改成 % 10000
python -u Door_optimal_feature_training.py --N 10 2>&1 | tee ~/log_door_train.txt
#    确认产出 Door/koopmanMatrix.npy（570×570）后，再开评估：137 行改 % 100

# 6. 开环误差（不需要仿真、不需要 ResNet，秒级）
python ../tools/eval_open_loop.py --K Door/koopmanMatrix.npy

# 7. 闭环成功率
python Koopman_gt_operator_eval.py --task Door --K gt_koopmanMatrix.npy --num-test 20
```

几个必须注意的点：

- **`--N` 是动力学预测长度，不是轮数。** 默认 40，本次用 10。
- **`Koopman_gt_operator_eval.py` 在原版上改过**：加了 `--task / --K / --num-test`，默认 `--K gt_koopmanMatrix.npy`，并加了字典维度自检（546 和 570 混用会直接说明原因，而不是 `shapes not aligned`）。
- **真值线文件名不统一**：`door_main()` 存 `gt_koopmanMatrix.npy`（第 84 行），Tool/Relocate/Reorientation 三个 `main()` 都存 `koopmanMatrix.npy`（145/191/249 行）。别看文件名判断「真值/学到的」，看维度。
- **`python -u` 不要省**：`generate_feature` 一行输出都没有，6729 张图在 CPU 上要 3~5 分钟，再叠上 stdout 块缓冲 + `tee`，日志会静止十几分钟，非常像卡死。

---

## 7. 踩坑清单

| 现象 | 根因 | 解法 |
|---|---|---|
| `gym.error.UnregisteredEnv: No registered env with id: door-v0` | gym 0.13 不扫 entry_points，`robohive` 没被 import 时环境没注册 | 脚本开头 `import robohive`（作者自己的脚本也在第 26 行这么做） |
| `RuntimeError: Parent directory ./Door/controller does not exist` | `torch.save` 不创建父目录，且报错发生在**60 轮训练跑完之后**，等于白训 | 先 `mkdir -p Door/controller`；或 `os.makedirs(os.path.dirname(p), exist_ok=True)` |
| 训练到 epoch 5/10 长时间无输出，主机卡到 WSL 无响应 | 把第 137 行的 `epoch % 50` 改成了 `% 5`。这是**评估频率**不是训练量，一触发就要重提 6729 张图特征 | 改回 `% 50`；只想跑通就 `% 10000`。判断是「在算」还是「真死」：另开终端 `top` / `free -h` |
| 日志长时间空白 | `generate_feature` 无进度输出 + stdout 块缓冲 + `tee` | `python -u`；`Koopman_evaluation.py:53` 的 dataloader 套一层 `tqdm(...)` |
| `img_size` 与实际帧数不符 | 原代码写 `num_demo * 70` | 两处（`:91`、`:140`）改成 `sum(len(d["actions"]) for d in demo_data[:num_demo])` = 6729 |
| 真值线打印 `avg error 3.9279` | `Koopman_gt_operator_training.py:100` 的 `error` 在 69 步循环里累加，最后只除以 `len(Training_data)`（25），放了 69 倍 | 除以总步数；正确值 0.0569 |
| `FileNotFoundError: .../NN_controller_best.pt` | 仓库不提供该权重 | 自己训（`tools/train_controller.py`），或只跑真值线验证管线 |
| 570 维闭环 `shapes not aligned` | 真值线字典 546（num_obj=7）、KOROL 线 570（num_obj=8），而 `Koopman_gt_evaluation.py` 里 `obj_OriState` 只拼 7 维 | 两条线不能换 K；评估脚本按维度自检 |
| 201 轮训完找不到 ResNet 权重 | `Door_optimal_feature_training.py` **全文没有 `torch.save`**，只存了 K 和特征缓存 | 在 `for epoch` 循环之后加 `torch.save(resnet_model.state_dict(), './Door/controller/ResNet_door.pt')` |
| 训练收敛比预期慢 | `optimizer_feature` 定义在 epoch 循环**内部**（第 111 行），每轮重建，Adam 的一阶/二阶矩每轮清零 | 优化器提到循环外；改 lr 时手动同步 `param_groups` |
| 201 轮跑完看不到 loss | 循环里没有任何 print，`loss_results` 只在内存 | 加 `print("epoch %3d loss %.4f" % (epoch, total_loss), flush=True)` |
| `Success rate (sim) = 0.000000` | 不一定模型问题，先看 horizon（§3.5） | 用 `tools/eval_door_curve.py` 看 `door_pos` 曲线走到哪 |
| 想录训练前后的 3D 动画 | 渲染一直在跑（离屏上下文），只是画面被喂给 ResNet 后丢弃了 | `tools/record_rollout.py`，monkey-patch `GymEnv.step` 抓帧，仓库零改动 |

另外两处「看着像 bug 其实不是」：`demo_loader.py:250` 的 `if False: e.env.mj_render()` 是作者关掉的渲染开关；`Koopman_evaluation.py:117/:165` 的 `gif_frames = []` 从来没被 append 过——作者本来打算存 gif，没写完。

---

## 8. 经验教训

**1. 先确认评估口径，再判断模型好坏。**
这次四次评估全 0，第一反应是「练坏了」，实际是 70 步门还没开始转。评估函数只吐 0/1 的时候，它就分不清「模型差」「窗口短」「初值难」这三件事。**任何只有二值输出的评估，都该先改成曲线再看。**

**2. 复现的硬卡点往往在「作者没提供的那部分」。**
数据没了还能从仓库里的 demo 重建（`demo_loader` 不回放，只依赖确定性渲染，这从代码能读出来）；但 controller 权重没给、训练脚本也没有，就只剩反推结构这一条路，而反推出来的东西精度不可控。**接手一个仓库时，先盘点「哪些文件是必需的但没有生成路径」，这个清单比 README 有用。**

**3. 单步指标好 ≠ 闭环可用。**
控制器逐维相关均值 > 0.8、MSE 从 0.23 降到 0.012，单看都还行，闭环仍然 0。逆动力学是序列决策问题，单步回归把误差平摊掉了。要么在 rollout 上算损失（论文本身就这么干），要么直接在闭环里评。

**4. 训练和推理的输入必须同一来源。**
闭环里 `current` 用 K 预测值、训练时用真实值，这个不一致在开环设定下是设计选择，但它把「控制器误差」和「算子漂移」耦合在一起，出了事分不开。**评估时先解耦，再谈单个模块的性能。**

**5. 回归对多模态任务会退化成平均。**
L1/L2 损失下学出来的「平均动作」在 MSE 上不难看，闭环里表现是「原地飘」。这类任务该上生成式策略或 DAgger，不是调网络深度能解决的。

**6. 别把「跑完没报错」当成「跑通」。**
`Door_optimal_feature_training.py` 跑完 201 轮、存下 K、日志干净，但 ResNet 权重没存、loss 没打印、评估恒 0。**一次成功的运行应该留下可复评的产物**：权重、日志、逐回合指标。否则等于没跑。

**7. 时间预算按评估频率算，不是按 epoch 算。**
200 轮听着不多，但每 50 轮一次的全量特征重提（6729 张图）在 CPU 上是分钟级；把 `% 50` 改成 `% 5`，总耗时不是降而是涨 3 倍以上。改超参之前先算清「这个参数触发了几次重活」。

---

## 9. 目录

```
KOROL-reproduction/
├── Korol/                      # KOROL 主体（作者代码 + 本次改动）
│   ├── Door_optimal_feature_training.py    # 主线：ResNet + EDMD 交替训练
│   ├── Koopman_gt_operator_training.py     # 真值线 KODex，存 gt_koopmanMatrix.npy
│   ├── Koopman_gt_operator_eval.py         # 真值线闭环（本次改成带参数的版本）
│   ├── my_demos_build.py                   # 本次新增：从 DAPG demo 重建数据
│   ├── my_demos_test.py / my_demos_read_test.py
│   ├── my_run_gt_eval.py                   # 真值线闭环的手写 driver
│   ├── baselines/                          # BC / Diffusion / NDP 基线（均需作者的 controller）
│   └── utils/                              # Observables / Koopman_evaluation / demo_loader …
├── tools/                      # 本次新增的诊断与评估脚本
│   ├── train_controller.py     # 逆动力学 BC（重建版）
│   ├── check_controller.py     # MSE / RMSE / 逐维相关 / 均值基线
│   ├── eval_open_loop.py       # 开环 rollout 误差
│   ├── eval_door_curve.py      # 闭环并记录 door_pos 曲线
│   └── record_rollout.py       # 3D rollout 录 GIF
├── mj_envs/                    # 环境源码（robohive，含 DAPG 示范 pickle）
├── mjrl/                       # mjrl，pip install -e .
├── install.txt / command.txt   # 官方安装与运行说明（保留原样）
├── check_env.py                # 依赖自检
└── .gitignore                  # 忽略 Korol/Door/（数据与权重）、Korol/gt_*.npy、视频
```

`Korol/Door/` 被 gitignore：里面有 6729 个 `rgbd_*.npy`（约 1.9 GB）、`Testing.pickle`、特征缓存和权重，全部可由 §6 的第 2、4、5 步重建。

---

## 10. 未归档的产物

| 文件 | 大小 | 重建方式 |
|---|---|---|
| `Korol/Door/Data/rgbd_*.npy` × 6729 | ~1.9 GB | `python my_demos_build.py`（约 20 分钟，取决于渲染速度） |
| `Korol/Door/Data/Testing.pickle` | 6.6 MB | 同上 |
| `Korol/Door/feature_door_demo_full.pickle` | 6729×8 | `Door_optimal_feature_training.py` 首次 `generate_feature` 时生成 |
| `Korol/Door/koopmanMatrix.npy` | 2.6 MB（570² float64） | 同上，第 99 行 `train_koopman` |
| `Korol/gt_koopmanMatrix.npy` | 546² | `Koopman_gt_operator_training.py` |
| `Korol/Door/controller/NN_controller_best.pt` | 小 | `python tools/train_controller.py` |
| `Korol/Door/controller/ResNet_door.pt` | — | **本次没有**：原脚本不保存，需补 `torch.save` 后重训 201 轮 |

---

## 11. 参考

- 论文：[arXiv:2407.00548](https://arxiv.org/abs/2407.00548) · [OpenReview](https://openreview.net/forum?id=A6ikGJRaKL)
- `mj_envs` / `mjrl` 为 robohive 与 mjrl 的源码副本
- DAPG 示范：Rajeswaran et al., *Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations*, RSS 2018

```bibtex
@inproceedings{chenkorol,
  title={KOROL: Learning Visualizable Object Feature with Koopman Operator Rollout for Manipulation},
  author={Chen, Hongyi and Abuduweili, Abulikemu and Agrawal, Aviral and Han, Yunhai and Ravichandaran, Harish and Liu, Changliu and Ichnowski, Jeffrey},
  booktitle={8th Annual Conference on Robot Learning},
  year={2024}
}
```
