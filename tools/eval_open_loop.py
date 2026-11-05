# -*- coding: utf-8 -*-
"""开环误差（open-loop rollout error）单独评估。

从 Koopman_gt_operator_training.py:86-100 那段抽出来，去掉「训练 K」，只保留「用 K 开环推、和真值比」。

与仓库原版的三处差别（都是有意的）：
  1. 原版 print 的 avg error 是 `error / len(Training_data)`——error 里已经累加了
     25 条 x 69 步的逐步子均值，除以条数等于没除够，数值被放大 69 倍。这里改成对全部
     (条 x 步) 取均值。
  2. 原版把 horizon 写死为 70，且不管示范长度。这里 horizon 可调，遇到短于 horizon 的
     示范自动截断（本机 25 条最短 223 步，够用）。
  3. 原版只认真值线（num_obj=7，物体状态由 objpos/objvel/handle_init 拼）。这里按 K 的
     维度自动判断：546=真值线、570=学到的线（物体状态是 8 维 rgbd_feature）。

用法（必须在 ~/project/KOROL/Korol 下执行，数据按 ./Door/Data 相对当前目录解析）：
    python eval_open_loop.py                                  # 默认真值线
    python eval_open_loop.py --K gt_koopmanMatrix.npy
    python eval_open_loop.py --K Door/koopmanMatrix.npy --horizon 70
    python eval_open_loop.py --K gt_koopmanMatrix.npy --per-demo

注意：K 就是在这 25 条示范上拟合出来的，所以这里算的是**训练集上的开环误差**，
不是泛化误差。真值线（最小二乘闭式解）和学到的线（ResNet 特征）口径一致，可直接横向比。

和 Koopman_gt_operator_eval.py 的分工——两者**互不依赖，不用排队跑**：
  本脚本只读 Testing.pickle（+ 570 维时读 feature 缓存）和 K，**不加载控制器、不建
  mujoco 仿真、不现场采测试初值**，所以秒级出数，控制器还没训好也能跑。
  那边走闭环成功率，要 NN_controller_best.pt、要现场 generate_unseen_data_door(200)、
  每个初值再跑 70 步物理步进，分钟级。
  用法上是「先筛后验」：开环误差差的 K，闭环成功率必然差，不必浪费时间去跑仿真；
  开环误差好只是必要条件，还得靠闭环成功率确认控制器能跟上。
"""
# 允许从仓库根或 tools/ 下直接运行：把 Korol/ 加进 sys.path，并把工作目录切过去
# （脚本里的 ./Door/... 这类相对路径都相对 Korol/）
import os
import sys
_KOROL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Korol")
if os.path.isdir(_KOROL):
    sys.path.insert(0, _KOROL)
    os.chdir(_KOROL)

import argparse
import os
import pickle

import numpy as np

from utils.Observables import DraftedObservable


# ---- 由 demo_loader.py 反推出来的字段口径，和 door_demo_playback() 逐位一致 ----
NUM_HAND = 28          # handpos = observations_visualization[t][:28]
IDX_OBJPOS = slice(32, 35)     # observations[t][32:35]
IDX_OBJVEL = slice(58, 59)     # observations_visualization[t][58:59]  （只有 1 维）
NUM_OBJ_GT = 7         # 3(objpos) + 1(objvel) + 3(door_body_pos)
NUM_OBJ_LEARNED = 8    # rgbd_feature 维度


def load_demos(data_dir, num_demo):
    """只取开环误差用得到的四个字段，不建仿真、不读 rgbd 图像。

    返回 [(handpos[T,28], objstates[T,num_obj])]，objstates 这里先塞 GT 口径，
    学到的线在用的时候替换成 rgbd_feature。
    """
    path = os.path.join(data_dir, "Testing.pickle")
    if not os.path.exists(path):
        raise SystemExit("找不到 %s" % path)
    demos = pickle.load(open(path, "rb"))
    demos = demos[:num_demo]

    out = []
    for d in demos:
        obs = d["observations"]
        obs_vis = d["observations_visualization"]
        handle_init = d["init_state_dict"]["door_body_pos"]
        handpos = np.asarray([o[:NUM_HAND] for o in obs_vis], dtype=np.float64)
        obj = np.asarray(
            [np.concatenate([o[IDX_OBJPOS], v[IDX_OBJVEL], handle_init]) for o, v in zip(obs, obs_vis)],
            dtype=np.float64,
        )
        out.append((handpos, obj))
    return out


def load_features(feature_path, demos, num_demo):
    """rgbd_feature 缓存，key 是跨示范连续累加的 count（demo_loader 里那个 count）。"""
    if not os.path.exists(feature_path):
        raise SystemExit(
            "找不到 %s\n学到的线（570 维 K）需要它；只评估真值线请用 --K gt_koopmanMatrix.npy"
            % feature_path
        )
    feature_data = pickle.load(open(feature_path, "rb"))
    feats, count = [], 0
    for d in demos:
        n = len(d["actions"])
        arr = np.asarray(
            [list(feature_data[count + i].values())[0] for i in range(n)], dtype=np.float64
        )
        feats.append(arr)
        count += n
    return feats


def rollout(K, Ko, handpos, obj, horizon):
    """开环推 horizon-1 步（与 Koopman_gt_operator_training.py 的 range(horizon - 1) 同口径），
    逐步返回 |Δhandpos| 的均值。第 0 步是初值，不参与误差。"""
    steps = min(horizon - 1, len(handpos) - 1)
    z = Ko.z(handpos[0], obj[0])
    errs = np.zeros(steps)
    for t in range(steps):
        z = K.dot(z)                      # 只看上一步的预测，不回读真值
        errs[t] = np.abs(z[:NUM_HAND] - handpos[t + 1]).mean()
    return errs


def main():
    p = argparse.ArgumentParser(description="用一行 Koopman 矩阵算开环 rollout 误差")
    p.add_argument("--K", dest="k_path", default="gt_koopmanMatrix.npy",
                   help="K 矩阵文件，相对当前工作目录。默认真值线 gt_koopmanMatrix.npy")
    p.add_argument("--horizon", type=int, default=70,
                   help="开环步数，默认 70（与 Koopman_gt_operator_training.py 的 horizon 一致）")
    p.add_argument("--num-demo", dest="num_demo", type=int, default=25,
                   help="参与评估的示范条数，默认 25（= 训练 K 用的条数）")
    p.add_argument("--data-dir", dest="data_dir", default="./Door/Data")
    p.add_argument("--feature", dest="feature_path", default="./Door/feature_door_demo_full.pickle",
                   help="rgbd 特征缓存，只有学到的线（570 维 K）才需要")
    p.add_argument("--per-demo", action="store_true", help="额外打印每条示范自己的误差")
    args = p.parse_args()

    if not os.path.exists(args.k_path):
        npy = sorted(f for f in os.listdir(".") if f.endswith(".npy"))
        raise SystemExit("找不到 %s\n当前目录的 .npy：%s" % (args.k_path, npy))
    K = np.load(args.k_path)

    # 546 = DraftedObservable(28,7)（真值线）；570 = DraftedObservable(28,8)（学到的线）
    if K.shape[0] == 546:
        num_obj, line = NUM_OBJ_GT, "真值线（objpos+objvel+handle_init）"
    elif K.shape[0] == 570:
        num_obj, line = NUM_OBJ_LEARNED, "学到的线（8 维 rgbd_feature）"
    else:
        raise SystemExit(
            "K 是 %s，既不是真值线的 546x546 也不是学到的线的 570x570" % (K.shape,)
        )
    if K.shape[0] != K.shape[1]:
        raise SystemExit("K 不是方阵：%s" % (K.shape,))
    print("K: %s  shape=%s" % (args.k_path, K.shape))
    print("口径: %s  num_obj=%d" % (line, num_obj))

    demos = load_demos(args.data_dir, args.num_demo)
    if num_obj == NUM_OBJ_LEARNED:
        raw = pickle.load(open(os.path.join(args.data_dir, "Testing.pickle"), "rb"))[: args.num_demo]
        feats = load_features(args.feature_path, raw, args.num_demo)
        demos = [(hp, f) for (hp, _), f in zip(demos, feats)]

    Ko = DraftedObservable(NUM_HAND, num_obj)
    n_obs = Ko.compute_observable(NUM_HAND, num_obj)
    if n_obs != K.shape[0]:
        raise SystemExit("字典维数对不上：z 是 %d 维，K 是 %s" % (n_obs, K.shape))

    curves = [rollout(K, Ko, hp, ob, args.horizon) for hp, ob in demos]
    H = min(len(c) for c in curves)
    curve = np.stack([c[:H] for c in curves])          # (num_demo, H)

    print("\n%-8s %s" % ("step", "mean |Δhandpos|"))
    marks = set([1, 5, 10, 20, 30, 40, 50, 60, H])
    for t in range(H):
        if (t + 1) in marks:
            print("%-8d %.6f" % (t + 1, curve[:, t].mean()))
    print("\n开环 %d 步整体均值: %.6f  （%d 条 x %d 步）" % (H, curve.mean(), len(demos), H))
    print("第 %d 步均值: %.6f  最大: %.6f" % (H, curve[:, -1].mean(), curve[:, -1].max()))

    if args.per_demo:
        print("\n%-6s %-6s %-12s %-12s" % ("demo", "帧数", "首步误差", "末步误差"))
        for i, (c, (hp, _)) in enumerate(zip(curves, demos)):
            print("%-6d %-6d %-12.6f %-12.6f" % (i, len(hp), c[0], c[-1]))

    np.save("open_loop_err_%s.npy" % os.path.splitext(os.path.basename(args.k_path))[0], curve)
    print("\n逐步误差曲线已存 open_loop_err_*.npy，形状 (num_demo, horizon)")


if __name__ == "__main__":
    main()
