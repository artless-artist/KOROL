# -*- coding: utf-8 -*-
"""控制器体检：MSE / RMSE / 逐维相关系数 / 「全输出均值」基线对比。

为什么需要第二个脚本：训练脚本打出的 MSE 单独看没有意义。示范动作的 std 只有 0.55，
网络很容易学成「输出接近均值」——MSE 看着低，闭环里却表现为「手缓慢飘向固定姿势、
门纹丝不动」。所以要看 ①RMSE 占动作幅度的百分比 ②逐维相关系数 ③和均值基线的差距。

用法（必须在 ~/project/KOROL/Korol 下，数据按 ./Door/Data 相对当前目录解析）：
    python check_controller.py
    python check_controller.py --ckpt ./Door/controller/NN_controller_best.pt --num-demo 25

判据：
    相关均值 > 0.8            → 单步可用，可以进闭环试
    0.5 ~ 0.8                → 学得动但精度不够，闭环大概率能推开一点、达不到阈值
    < 0.5，且 MSE ≈ 均值基线  → 学成了「平均动作」，必须重训（先查 X 拼接顺序）
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
import torch

from utils.fc_network import FCNetwork

NUM_HAND = 28


def build_xy(demos, num_demo):
    """和训练脚本同一口径：(hand[t], hand[t+1]) -> action[t]。

    hand 取 observations_visualization 的前 28 维（与 demo_loader.py:41 一致），
    对应评估时 controller 的输入 (qpos_当前, K 预测的下一步 qpos)。
    """
    X, Y = [], []
    for p in demos[:num_demo]:
        ov = np.asarray(p["observations_visualization"], dtype=np.float64)
        hand = ov[:, :NUM_HAND]
        act = np.asarray(p["actions"], dtype=np.float64)
        for t in range(len(act) - 1):
            X.append(np.concatenate([hand[t], hand[t + 1]]))
            Y.append(act[t])
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description="检查 NN_controller_best.pt 训得怎么样")
    ap.add_argument("--ckpt", default="./Door/controller/NN_controller_best.pt")
    ap.add_argument("--data", default="./Door/Data/Testing.pickle")
    ap.add_argument("--num-demo", dest="num_demo", type=int, default=25,
                    help="训练时用的示范条数，默认 25（要和训练一致才能复现那个 MSE）")
    a = ap.parse_args()

    if not os.path.exists(a.ckpt):
        raise SystemExit("找不到 %s\n训练脚本只 torch.save 一次，目录缺失时那轮白训，需重跑" % a.ckpt)

    demos = pickle.load(open(a.data, "rb"))
    X, Y = build_xy(demos, a.num_demo)
    print("样本 %s -> %s   （%d 条示范）" % (X.shape, Y.shape, min(a.num_demo, len(demos))))

    net = FCNetwork([2 * NUM_HAND, 4 * NUM_HAND, 4 * NUM_HAND, 2 * NUM_HAND, NUM_HAND],
                    nonlinearity="relu")
    net.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    net.eval()
    with torch.no_grad():
        P = net(torch.from_numpy(X)).numpy()

    mse = float(((P - Y) ** 2).mean())
    rmse = mse ** 0.5
    std = float(Y.std())
    base_mse = float(((Y.mean(0) - Y) ** 2).mean())     # 「不管输入、一律输出均值」的基线

    print("\nMSE            %.6f" % mse)
    print("RMSE           %.4f   （占动作 std %.3f 的 %.0f%%）" % (rmse, std, rmse / std * 100))
    print("均值基线 MSE   %.6f   （模型比基线好 %.1f%%）" % (base_mse, (1 - mse / base_mse) * 100))

    # 逐维相关：std 为 0 的维（几乎不动的关节）跳过，否则 nan
    keep = [i for i in range(Y.shape[1]) if Y[:, i].std() > 1e-8]
    cors = np.array([np.corrcoef(P[:, i], Y[:, i])[0, 1] for i in keep])
    cors = cors[np.isfinite(cors)]
    print("\n逐维相关  均值 %.3f   中位 %.3f   最小 %.3f   （%d/%d 维有效）"
          % (cors.mean(), np.median(cors), cors.min(), len(cors), Y.shape[1]))
    worst = np.argsort(cors)[:5]
    print("最差 5 维（按有效维排序）:", ", ".join(" %.3f" % cors[i] for i in worst))

    print("\n判定：", end="")
    if mse >= base_mse * 0.95:
        print("学成了平均动作（相关性 %.2f），闭环必败，重训。" % cors.mean())
    elif cors.mean() > 0.8:
        print("单步指标可用，可以进闭环试。（注意：单步好不等于闭环成，见 README §3.3）")
    elif cors.mean() > 0.5:
        print("学得动但精度不够，闭环大概率推不到 1.35，建议加轮次/降 lr 再训。")
    else:
        print("相关性偏低，先查 X 是不是 [x_t, x_{t+1}]、hand 是否取自 observations_visualization。")


if __name__ == "__main__":
    main()
