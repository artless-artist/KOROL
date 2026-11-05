"""逆动力学控制器训练（重建版）。

背景：作者的仓库只提供 `Controller.load_state_dict(torch.load('./Door/controller/NN_controller_best.pt'))`，
21 个脚本里没有任何一处 `torch.save` 过这个文件，也没有训练脚本。这份是按它的网络结构
（`FCNetwork([56, 112, 112, 56, 28])`）从示范里回归出来的替代品，不是官方权重。

口径（与 utils/Koopman_evaluation.py 的闭环一致）：
    输入 56 = 当前 handpos(28) ⊕ 下一帧 handpos(28)
    输出 28 = 该帧实际执行的动作
    样本数 = 6729 - 25 = 6704（每条示范少最后一步，最后一步没有「下一状态」）

本次复现的实际结果：60 轮后 MSE 0.0126（RMSE 0.112 ≈ 动作 std 0.546 的 21%），
闭环成功率 0。详情见仓库 README §3。

用法：
    python tools/train_controller.py
    python tools/train_controller.py --epochs 300 --lr 3e-4
    python tools/train_controller.py --out ./Door/controller/NN_controller_best.pt

超参（60 轮 / batch 64 / lr 1e-3）是本次复现用的设置，不是论文配置。
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch

# 允许从仓库根或 tools/ 下直接运行
_KOROL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Korol")
if os.path.isdir(_KOROL):
    sys.path.insert(0, _KOROL)
    os.chdir(_KOROL)

from utils.fc_network import FCNetwork  # noqa: E402

NUM_HAND = 28
NET_SIZE = [2 * NUM_HAND, 2 * 2 * NUM_HAND, 2 * 2 * NUM_HAND, 2 * NUM_HAND, NUM_HAND]


def build_xy(demos, num_demo):
    X, Y = [], []
    for p in demos[:num_demo]:
        ov = p.get("observations_visualization")
        if ov is None:
            raise SystemExit(
                "示范里没有 observations_visualization —— 数据没重建过。\n"
                "先跑 python Korol/my_demos_build.py 生成 ./Door/Data/Testing.pickle")
        hand = np.asarray(ov, dtype=np.float64)[:, :NUM_HAND]
        act = np.asarray(p["actions"], dtype=np.float64)
        for t in range(len(act) - 1):
            X.append(np.concatenate([hand[t], hand[t + 1]]))
            Y.append(act[t])
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description="训练逆动力学控制器（BC 替代品）")
    ap.add_argument("--data", default="./Door/Data/Testing.pickle")
    ap.add_argument("--num-demo", dest="num_demo", type=int, default=25,
                    help="用前几条示范训练，默认 25（Door 任务 DAPG 全部）")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="./Door/controller/NN_controller_best.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    demos = pickle.load(open(args.data, "rb"))
    X, Y = build_xy(demos, args.num_demo)
    print("示范 %d 条，样本 %s → %s" % (len(demos), tuple(X.shape), tuple(Y.shape)))
    print("动作范围 %.4f ~ %.4f，std %.4f" % (Y.min(), Y.max(), Y.std()))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = FCNetwork(NET_SIZE, nonlinearity="relu").to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    Xt = torch.from_numpy(X).to(device)
    Yt = torch.from_numpy(Y).to(device)
    n = len(Xt)

    for epoch in range(args.epochs):
        net.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            loss = loss_fn(net(Xt[idx]), Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        mse = total / n
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print("epoch %3d  mse %.6f" % (epoch, mse), flush=True)

    # torch.save 不会创建父目录，缺了它会在训练跑完之后才抛
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(net.state_dict(), args.out)
    print("已保存 %s" % args.out)
    print("下一步：python tools/check_controller.py（看逐维相关与均值基线）")


if __name__ == "__main__":
    main()
