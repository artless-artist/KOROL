# -*- coding: utf-8 -*-
"""闭环成功率，但把「门到底开到多少」也打出来。

为什么需要它
------------
`Koopman_evaluation.py:105 koopman_policy_control()` 只 print 一个
`Success rate (sim) = 0.000000`，到此为止。0% 到底是「控制器没跟上」「K 不准」
还是「窗口太短、门还没开始动」，从那个数字完全看不出来。

这个脚本把每个回合的 `door_pos` 逐步记下来，并把**人类示范的同一曲线**放在旁边对比：

    示范里 door_pos 第 69 步只有 0.01 rad，第一次超过 0.2 是第 140 步（中位 171），
    第一次超过 1.35 最早第 166 步，而且 25 条示范里只有 7 条最终能到 1.35。

所以「闭环 70 步、阈值 1.35」这个口径下，成功率恒为 0 是结构性的，
跟你训出来的 K 好不好没关系。把 horizon 放长、或把阈值放低，
再看曲线，才能判断 K 和控制器到底行不行。

用法（在 ~/project/KOROL/Korol 下执行）
--------------------------------------
    # 真值线：只需要 controller + gt K，最便宜
    python eval_door_curve.py --K gt_koopmanMatrix.npy --num-test 20

    # 换窗口和阈值
    python eval_door_curve.py --K gt_koopmanMatrix.npy --horizon 260 --threshold 1.0

    # 学到的线（570 维）：还需要 ResNet 编码器权重
    python eval_door_curve.py --K Door/koopmanMatrix.npy \\
        --resnet Door/controller/ResNet_door.pt --horizon 260

    # 每步用实测手部状态重新提升 z（不再靠 K 一路推 260 步）
    python eval_door_curve.py --K gt_koopmanMatrix.npy --horizon 260 --reanchor

产物：`door_pos_curve_<K文件名>.npy`，形状 (回合数, horizon)，每行是一条 door_pos 轨迹。
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

# robohive 必须在 gym.make 之前 import，否则 UnregisteredEnv: door-v0
import robohive  # noqa: F401
import torch

from utils.gym_env import GymEnv
from utils.Observables import DraftedObservable
from utils.fc_network import FCNetwork

NUM_HAND = 28
IDX_OBJPOS = slice(32, 35)
IDX_OBJVEL = slice(58, 59)


# --------------------------------------------------------------------------- #
# 示范参考曲线：不需要仿真，只读 pickle，秒出
# --------------------------------------------------------------------------- #
def demo_reference(data_dir="./Door/Data", num_demo=25):
    """示范里 door_pos 的统计，作为「这套数据本来多慢」的标尺。

    door_pos 在 observations 的第 28 列 —— 布局是
    hand_jnt(0..26) | latch_pos(27) | door_pos(28) | palm(29..31) | handle(32..34) | reach_err(35..37) | (38)
    已用 reward 公式反推校验过：第 218/250 步的 +2/+8 奖励跳变正好对应
    col28 越过 0.2 / 1.0（door_v0.get_reward_dict 的 bonus 项）。
    """
    path = os.path.join(data_dir, "Testing.pickle")
    if not os.path.exists(path):
        return None
    demos = pickle.load(open(path, "rb"))[:num_demo]
    curves = [np.asarray(d["observations"], dtype=np.float64)[:, 28] for d in demos]
    n = min(len(c) for c in curves)
    return np.stack([c[:n] for c in curves])          # (num_demo, T)


def first_cross(curves, th):
    """每条示范首次超过 th 的步号；没到过的返回 None。"""
    out = []
    for c in curves:
        idx = np.where(c > th)[0]
        out.append(int(idx[0]) if len(idx) else None)
    return out


def print_reference(ref, thresholds):
    print("\n" + "=" * 68)
    print("人类示范的 door_pos 参考（%d 条，长度 %d 步）" % ref.shape)
    print("=" * 68)
    marks = [0, 20, 40, 60, 69, 100, 140, 170, 200, 250, ref.shape[1] - 1]
    marks = sorted(set(m for m in marks if 0 <= m < ref.shape[1]))
    print("  步号   " + "".join("%9d" % m for m in marks))
    print("  均值   " + "".join("%9.3f" % ref[:, m].mean() for m in marks))
    print("  最大   " + "".join("%9.3f" % ref[:, m].max() for m in marks))
    for th in thresholds:
        need = first_cross(ref, th)
        ok = [v for v in need if v is not None]
        if ok:
            print("  >%.2f : %2d/%d 条达到，最早第 %d 步，中位第 %d 步"
                  % (th, len(ok), len(need), min(ok), int(np.median(ok))))
        else:
            print("  >%.2f : 0/%d 条达到" % (th, len(need)))


# --------------------------------------------------------------------------- #
# 载入
# --------------------------------------------------------------------------- #
def load_controller(ckpt, device):
    net = FCNetwork([56, 112, 112, 56, 28], nonlinearity="relu")
    if not os.path.exists(ckpt):
        raise SystemExit("找不到控制器 %s" % ckpt)
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    net.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
    net.eval()
    net.to(device)
    print("[info] controller <- %s" % ckpt)
    return net


def load_resnet(ckpt, device):
    from utils.resnet import ClassificationNetwork18
    net = ClassificationNetwork18(feat_dim=8).float()
    if not os.path.exists(ckpt):
        raise SystemExit(
            "找不到 ResNet 编码器 %s\n"
            "学到的线（570 维）闭环需要它，而 Door_optimal_feature_training.py\n"
            "从来没有 torch.save 过 → 之前那次训练的编码器已经随进程丢了。\n"
            "要么重训并加保存，要么先用真值线 --K gt_koopmanMatrix.npy 验证闭环管线。" % ckpt)
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net.eval()
    net.to(device)
    print("[info] resnet   <- %s" % ckpt)
    return net


# --------------------------------------------------------------------------- #
# 闭环：与 koopman_policy_control 同口径，但记下每一步
# --------------------------------------------------------------------------- #
def run(controller, K, Koopman, num_obj, Test_data, horizon, use_resnet,
        resnet, device, reanchor):
    e = GymEnv("door-v0")
    e.reset()
    init_state = {}
    curves = []
    for k in range(len(Test_data)):
        item = Test_data[k][0]
        init_state["qpos"] = item["init"]["qpos"]
        init_state["qvel"] = item["init"]["qvel"]
        init_state["door_body_pos"] = item["init"]["door_body_pos"]
        e.set_env_state(init_state)

        hand = np.asarray(item["handpos"], dtype=np.float64)
        if use_resnet:
            rgb, depth = e.env.mj_render()
            rgb = (rgb.astype(np.uint8) - 128.0) / 128
            rgbd = np.concatenate([rgb, depth[..., np.newaxis]], axis=2)
            rgbd = np.transpose(rgbd, (2, 0, 1))[np.newaxis, ...]
            with torch.no_grad():
                _, feat = resnet(torch.from_numpy(rgbd).float().to(device))
            obj = feat[0].cpu().numpy().astype(np.float64)
        else:
            obj = np.append(item["objpos"],
                            np.append(item["objvel"], item["handle_init"])).astype(np.float64)

        z = Koopman.z(hand, obj)
        curve = np.zeros(horizon)
        for t in range(horizon):
            z_next = K.dot(z)
            target = z_next[:NUM_HAND]
            if reanchor:
                # 用实测手部状态重新提升 z：K 只负责一步预测，误差不再累积。
                # 物体特征保持初值（相机看不到门转了多少，这也是原脚本的设置）。
                measured = e.get_env_state()["qpos"][:NUM_HAND]
                z = Koopman.z(measured, obj)
                target = K.dot(z)[:NUM_HAND]
                current = measured
            else:
                current = z[:NUM_HAND]        # 与原脚本一致：开环，不回读真实状态
            z = z_next

            inp = torch.from_numpy(np.append(current, target))
            with torch.no_grad():
                act = controller(inp).cpu().numpy()
            e.step(act)
            curve[t] = float(e.env.get_obs_dict(e.env.sim)["door_pos"])
        curves.append(curve)
        print("  ep %2d  末值 %.3f  最大 %.3f" % (k, curve[-1], curve.max()))
    return np.stack(curves)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="记录闭环里 door_pos 的逐条曲线")
    p.add_argument("--K", dest="k_path", default="gt_koopmanMatrix.npy")
    p.add_argument("--controller", default="./Door/controller/NN_controller_best.pt")
    p.add_argument("--resnet", default="./Door/controller/ResNet_door.pt",
                   help="学到的线（570 维）才需要")
    p.add_argument("--num-test", dest="num_test", type=int, default=20,
                   help="测试初值条数，默认 20（与 Door_optimal_feature_training.py 一致）")
    p.add_argument("--horizon", type=int, default=70,
                   help="闭环步数，默认 70（原脚本值）。这套示范数据下 70 步门还没动，"
                        "想看到有意义的数字先试 260")
    p.add_argument("--threshold", type=float, default=1.35,
                   help="成功判据 door_pos > 该值，默认 1.35（door_v0 的 solved 阈值）")
    p.add_argument("--reanchor", action="store_true",
                   help="每步用实测手部状态重新提升 z，K 只做一步预测（长 horizon 下更稳）")
    p.add_argument("--data-dir", dest="data_dir", default="./Door/Data")
    p.add_argument("--num-demo", dest="num_demo", type=int, default=25)
    p.add_argument("--no-reference", action="store_true", help="不打印示范参考曲线")
    args = p.parse_args()

    if not os.path.exists(args.k_path):
        npy = sorted(f for f in os.listdir(".") if f.endswith(".npy"))
        raise SystemExit("找不到 %s\n当前目录的 .npy：%s" % (args.k_path, npy))
    K = np.load(args.k_path)
    if K.shape[0] == 546:
        num_obj, line, use_resnet = 7, "真值线（objpos+objvel+handle_init）", False
    elif K.shape[0] == 570:
        num_obj, line, use_resnet = 8, "学到的线（8 维 rgbd_feature）", True
    else:
        raise SystemExit("K 是 %s，既不是 546（真值线）也不是 570（学到的线）" % (K.shape,))
    print("K       : %s  shape=%s" % (args.k_path, K.shape))
    print("口径    : %s" % line)
    print("闭环    : horizon=%d  threshold=%.2f  reanchor=%s"
          % (args.horizon, args.threshold, args.reanchor))

    if not args.no_reference:
        ref = demo_reference(args.data_dir, args.num_demo)
        if ref is not None:
            print_reference(ref, [0.2, 1.0, args.threshold])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[info] device = %s" % device)
    controller = load_controller(args.controller, device)
    resnet = load_resnet(args.resnet, device) if use_resnet else None

    Koopman = DraftedObservable(NUM_HAND, num_obj)
    e0 = GymEnv("door-v0")
    e0.reset()
    Test_data = e0.generate_unseen_data_door(args.num_test)
    print("[info] 测试初值 %d 条" % len(Test_data))

    print("\n开始闭环 ……")
    curves = run(controller, K, Koopman, num_obj, Test_data, args.horizon,
                 use_resnet, resnet, device, args.reanchor)

    out = "door_pos_curve_%s.npy" % os.path.splitext(os.path.basename(args.k_path))[0]
    np.save(out, curves)

    print("\n" + "=" * 68)
    marks = sorted(set(m for m in (9, 29, 49, 69, 99, 149, 199, 249, args.horizon - 1)
                       if 0 <= m < args.horizon))
    print("%-8s %s" % ("第N步", "".join("%9d" % (m + 1) for m in marks)))
    print("%-8s %s" % ("均值", "".join("%9.3f" % curves[:, m].mean() for m in marks)))
    print("%-8s %s" % ("最大", "".join("%9.3f" % curves[:, m].max() for m in marks)))
    reached = (curves.max(axis=1) > args.threshold).sum()
    print("\n成功（末步 door_pos > %.2f）：%d/%d = %.3f"
          % (args.threshold, reached, len(curves), reached / len(curves)))
    print("任一步 door_pos > %.2f：%d/%d"
          % (args.threshold, (curves.max(axis=1) > args.threshold).sum(), len(curves)))
    print("全程最大 door_pos：%.3f（第 %d 步）"
          % (curves.max(), int(curves.max(axis=0).argmax()) + 1))
    print("\n曲线已存 %s，形状 %s" % (out, curves.shape))


if __name__ == "__main__":
    main()
