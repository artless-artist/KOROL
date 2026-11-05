#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
record_rollout.py —— 把 KOROL 的 rollout 录成 3D 动画（GIF / MP4）。

放在 ~/project/KOROL/Korol/ 下运行。**不修改仓库任何文件**：
用 monkey-patch 给 GymEnv.step / set_env_state 挂钩子抓帧，仓库原有逻辑一行不动。

为什么能抓：KOROL 走的 DoorEnvV0 用的是 mujoco_py 的离屏渲染上下文
（dopr_v0.py:197  MjRenderContextOffscreen），每调一次 mj_render() 都能取到
256x256 的 RGB。仓库里原来的用法是把它喂给 ResNet 当观测，从没存过盘。

四种模式
--------
--demo            回放专家示范（不需要任何训练产物，先验渲染通路通不通）
--mode gt         真值 K（546 维）+ 控制器 闭环      ← 现在就能跑
--mode learned    学到的 K（570 维）+ 控制器 + ResNet ← 缺 resnet 权重，暂不可用
--compare-random  同一组测试初值下，随机初始化控制器 vs 训好的控制器，输出左右对比

用法
----
# ① 先验通路（10 秒，不加载任何权重）
python record_rollout.py --demo --num-ep 3

# ② 控制器训练前/后对比（要 train_controller.py 训出的 NN_controller_best.pt）
python record_rollout.py --compare-random --num-test 3

# ③ 只看训练后
python record_rollout.py --mode gt --num-test 3 --tag after

# ④ 换机位（默认 azimuth=90 / distance=1.5，跟 door_v0.mj_viewer_setup 一致）
python record_rollout.py --mode gt --num-test 1 --azimuth 130 --elevation -20 --distance 1.8

# ⑤ 实时 3D 窗口（需要 WSLg 或 X11；WSL 里先确认 echo $DISPLAY 非空）
python record_rollout.py --demo --num-ep 1 --realtime
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
import sys
import time

import numpy as np
import torch

# 必须早于 gym.make：robohive 靠 import 副作用注册 door-v0
import robohive  # noqa: F401

from utils.gym_env import GymEnv
from utils.Observables import DraftedObservable
from utils.fc_network import FCNetwork
import utils.Koopman_gt_evaluation as GTE

NUM_HAND = 28
NUM_OBJ = 7          # 真值线的物体状态 = objpos(3) + objvel(1) + handle_init(3)
HORIZON = 70


# --------------------------------------------------------------------------- #
# 抓帧
# --------------------------------------------------------------------------- #
class Recorder:
    """把 GymEnv 每一步的画面收进 episodes[i] = [(HxWx3 uint8, door_pos), ...]"""

    def __init__(self, size=400, realtime=False, cam=None):
        self.size = size
        self.realtime = realtime
        self.cam = cam
        self.episodes = []
        self._warned = False

    # ---- 机位 ----
    def _apply_cam(self, env):
        if self.cam is not None:
            self.cam.apply(env)

    # ---- 回合边界 ----
    def start_episode(self):
        self.episodes.append([])

    def _ensure_episode(self):
        if not self.episodes:
            self.start_episode()

    # ---- 每步一张 ----
    def snap(self, e):
        env = e.env
        if self.realtime:
            # 真窗口模式不录，只刷新画面
            try:
                if getattr(env, "viewer", None) is None:
                    from mujoco_py import MjViewer
                    env.viewer = MjViewer(env.sim)
                    env.mujoco_render_frames = False
                    self._apply_cam(env)
                env.viewer.render()
            except Exception as exc:
                print("[warn] 实时窗口不可用，退回录制模式：%s" % exc)
                self.realtime = False
            return

        self._ensure_episode()
        img = self._grab(env)
        if img is None:
            return
        try:
            door = float(env.get_obs_dict(env.sim)["door_pos"])
        except Exception:
            door = float("nan")
        self.episodes[-1].append((img, door))

    def _grab(self, env):
        # 首次调用时 door_v0.mj_viewer_setup() 才会创建离屏上下文
        if getattr(env, "viewer", None) is None:
            env.mj_viewer_setup()
            self._apply_cam(env)

        v = env.viewer
        try:
            v.render(width=self.size, height=self.size)
            buf = v.read_pixels(self.size, self.size)
            if buf is not None:
                # mujoco_py 的原始 buffer 上下颠倒
                return np.asarray(buf)[::-1].copy()
        except Exception:
            pass

        # 回落到仓库自带的 mj_render（固定 256x256，返回值已翻正）
        try:
            rgb, _depth = env.mj_render()
            return np.asarray(rgb).copy()
        except Exception as exc:
            if not self._warned:
                print("[warn] 抓帧失败，后续帧将为空：%s" % exc)
                self._warned = True
            return None


def install_hooks(rec):
    """给 GymEnv.step / set_env_state 挂钩子，返回卸载函数。"""
    orig_step = GymEnv.step
    orig_set = GymEnv.set_env_state

    def step(self, action):
        out = orig_step(self, action)
        rec.snap(self)
        return out

    def set_env_state(self, state_dict):
        out = orig_set(self, state_dict)
        rec.start_episode()
        rec.snap(self)          # 把初始姿态也录进去
        return out

    GymEnv.step = step
    GymEnv.set_env_state = set_env_state

    def uninstall():
        GymEnv.step = orig_step
        GymEnv.set_env_state = orig_set

    return uninstall


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #
_FONT = None


def _font():
    global _FONT
    if _FONT is None:
        from PIL import ImageFont
        _FONT = ImageFont.load_default()
    return _FONT


def _overlay(frame, label=None, hud=None):
    from PIL import Image, ImageDraw

    im = Image.fromarray(frame)
    if label is None and hud is None:
        return im
    d = ImageDraw.Draw(im)
    font = _font()
    if label:
        d.rectangle([0, 0, im.width, 16], fill=(0, 0, 0))
        d.text((6, 3), label, fill=(255, 255, 255), font=font)
    if hud:
        d.rectangle([0, im.height - 16, im.width, im.height], fill=(0, 0, 0))
        d.text((6, im.height - 14), hud, fill=(255, 220, 0), font=font)
    return im


def save_gif(episode, path, fps, label=None):
    frames = [f for f, _ in episode]
    if not frames:
        print("[warn] 没有抓到帧，跳过 %s" % path)
        return None

    imgs = []
    n = len(episode)
    for i, (f, door) in enumerate(episode):
        hud = "step %2d/%d   door %.3f rad" % (i, n - 1, door)
        imgs.append(_overlay(f, label, hud))

    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=max(1, int(1000 / fps)), loop=0, optimize=True)
    size_mb = os.path.getsize(path) / 1e6
    print("  -> %s  (%d 帧, %.1f MB)" % (path, len(imgs), size_mb))
    return path


def save_mp4(episode, path, fps, label=None):
    """需要 imageio-ffmpeg；缺了就静默跳过（GIF 已经够看）。"""
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("[warn] 没有 imageio，跳过 mp4")
        return None

    frames = [np.asarray(_overlay(f, label, None)) for f, _ in episode]
    if not frames:
        return None
    try:
        imageio.mimsave(path, frames, fps=fps, codec="libx264", quality=8)
        print("  -> %s  (%.1f MB)" % (path, os.path.getsize(path) / 1e6))
        return path
    except Exception as exc:
        print("[warn] 写 mp4 失败（装一个 pip install imageio-ffmpeg 再试）：%s" % exc)
        return None


def save_compare(ep_a, ep_b, path, fps, label_a="BEFORE (random ctrl)",
                 label_b="AFTER (trained ctrl)"):
    """左右拼接。两段长度不同时，短的那段重复最后一帧补到等长。"""
    if not ep_a or not ep_b:
        print("[warn] 对比需要两段都有帧，跳过")
        return None

    n = max(len(ep_a), len(ep_b))
    from PIL import Image
    rows = []
    for i in range(n):
        fa, da = ep_a[min(i, len(ep_a) - 1)]
        fb, db = ep_b[min(i, len(ep_b) - 1)]
        ia = _overlay(fa, label_a, "step %2d  door %.3f" % (i, da))
        ib = _overlay(fb, label_b, "step %2d  door %.3f" % (i, db))
        canvas = Image.new("RGB", (ia.width + ib.width + 4,
                                   max(ia.height, ib.height)), (20, 20, 20))
        canvas.paste(ia, (0, 0))
        canvas.paste(ib, (ia.width + 4, 0))
        rows.append(canvas)

    rows[0].save(path, save_all=True, append_images=rows[1:],
                 duration=max(1, int(1000 / fps)), loop=0, optimize=True)
    print("  -> %s  (%d 帧, %.1f MB)" % (path, len(rows), os.path.getsize(path) / 1e6))
    return path


# --------------------------------------------------------------------------- #
# 三条跑法
# --------------------------------------------------------------------------- #
def run_demo(rec, num_ep, max_steps, cam):
    """回放 Testing.pickle 里的专家示范。不需要 controller、不需要 K。"""
    import pickle

    path = "./Door/Data/Testing.pickle"
    if not os.path.exists(path):
        raise SystemExit("找不到 %s，先跑 my_demos_build.py 重建数据" % path)
    demos = pickle.load(open(path, "rb"))
    print("[info] 示范 %d 条，回放前 %d 条" % (len(demos), num_ep))

    e = GymEnv("door-v0")
    e.reset()
    cam.apply(e)

    for k in range(min(num_ep, len(demos))):
        d = demos[k]
        acts = np.asarray(d["actions"])[:max_steps]
        t0 = time.time()
        e.set_env_state(d["init_state_dict"])
        for a in acts:
            e.step(np.asarray(a, dtype=np.float64))
        print("  ep %d: %d 步, %.1fs" % (k, len(acts), time.time() - t0))
    return e


def run_closed_loop(rec, controller, num_test, K_path, cam, seed_data=None):
    """真值线闭环：K(546) + controller。返回 Test_data。"""
    K_path = K_path or "gt_koopmanMatrix.npy"
    if not os.path.exists(K_path):
        raise SystemExit("找不到 %s，先跑 Koopman_gt_operator_training.py 生成真值 K" % K_path)

    K = np.load(K_path)
    Koopman = DraftedObservable(NUM_HAND, NUM_OBJ)
    dim = Koopman.compute_observable(NUM_HAND, NUM_OBJ)
    if K.shape[0] != dim:
        raise SystemExit(
            "K 是 %dx%d，但这个闭环只吃 %d 维（num_obj=%d）。\n"
            "570 维的学到的线走 eval_open_loop.py 算开环误差；闭环要 ResNet，"
            "而原仓库没存 ResNet 权重。"
            % (K.shape[0], K.shape[1], dim, NUM_OBJ))

    # 采测试初值（不录制，放 patch 之前）
    e_probe = GymEnv("door-v0")
    e_probe.reset()
    Test_data = seed_data if seed_data is not None else e_probe.generate_unseen_data_door(num_test)
    del e_probe
    print("[info] K = %s (%dx%d)，测试初值 %d 个"
          % (K_path, K.shape[0], K.shape[1], len(Test_data)))

    # 闭环函数会自己新建 env，机位在 Recorder 里懒应用
    uninstall = install_hooks(rec)
    try:
        t0 = time.time()
        GTE.door_koopman_policy_control_unseenTest(
            "door-v0", controller, Koopman, K, Test_data, NUM_HAND, NUM_OBJ)
        print("[info] 闭环耗时 %.1fs" % (time.time() - t0))
    finally:
        uninstall()
    return Test_data


def make_controller(ckpt, device="cpu"):
    net = FCNetwork([56, 112, 112, 56, 28], nonlinearity="relu")
    if ckpt and ckpt.lower() != "random":
        if not os.path.exists(ckpt):
            raise SystemExit("找不到控制器权重 %s" % ckpt)
        sd = torch.load(ckpt, map_location="cpu")
        # 兼容 DataParallel / 包了一层 dict 的情况
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        net.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
        print("[info] 控制器 %s 已加载" % ckpt)
    else:
        print("[info] 控制器 = 随机初始化（同一 seed=100，可复现）")
    net.eval()
    net.to(device)
    return net


class Camera:
    def __init__(self, azimuth=None, elevation=None, distance=None):
        self.azimuth = azimuth
        self.elevation = elevation
        self.distance = distance

    def apply(self, e):
        env = e.env
        if getattr(env, "viewer", None) is None:
            env.mj_viewer_setup()
        cam = env.viewer.cam
        if self.azimuth is not None:
            cam.azimuth = self.azimuth
        if self.elevation is not None:
            cam.elevation = self.elevation
        if self.distance is not None:
            cam.distance = self.distance


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="把 KOROL 的 rollout 录成 3D 动画（不改仓库文件）")
    ap.add_argument("--demo", action="store_true",
                    help="回放专家示范，不加载任何训练产物")
    ap.add_argument("--mode", default="gt", choices=["gt", "learned"],
                    help="gt=真值 K(546)，learned=学到的 K(570，需 ResNet，暂不可用)")
    ap.add_argument("--controller", default="Door/controller/NN_controller_best.pt",
                    help="控制器权重；传 random 表示随机初始化（=训练前）")
    ap.add_argument("--K", default="gt_koopmanMatrix.npy", help="Koopman 算子文件")
    ap.add_argument("--num-test", type=int, default=3, help="闭环测试回合数")
    ap.add_argument("--num-ep", type=int, default=3, help="demo 模式回放条数")
    ap.add_argument("--max-steps", type=int, default=300, help="demo 模式每条最多回放步数")
    ap.add_argument("--compare-random", action="store_true",
                    help="同一组测试初值下，随机控制器 vs 训好的控制器，出左右对比")
    ap.add_argument("--out", default="videos", help="输出目录")
    ap.add_argument("--tag", default=None, help="输出文件名前缀")
    ap.add_argument("--size", type=int, default=400, help="画面边长（正方形）")
    ap.add_argument("--fps", type=int, default=20, help="GIF 帧率")
    ap.add_argument("--mp4", action="store_true", help="额外写 mp4（需 imageio-ffmpeg）")
    ap.add_argument("--realtime", action="store_true", help="开实时 3D 窗口（需显示服务）")
    ap.add_argument("--azimuth", type=float, default=None)
    ap.add_argument("--elevation", type=float, default=None)
    ap.add_argument("--distance", type=float, default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cam = Camera(args.azimuth, args.elevation, args.distance)

    def dump(episodes, prefix):
        for i, ep in enumerate(episodes):
            base = os.path.join(args.out, "%s_ep%02d" % (prefix, i))
            save_gif(ep, base + ".gif", args.fps,
                     label=prefix.replace("_", " ").upper())
            if args.mp4:
                save_mp4(ep, base + ".mp4", args.fps,
                         label=prefix.replace("_", " ").upper())

    # ---------------- demo ----------------
    if args.demo:
        rec = Recorder(args.size, args.realtime, cam)
        uninstall = install_hooks(rec)
        try:
            run_demo(rec, args.num_ep, args.max_steps, cam)
        finally:
            uninstall()
        dump(rec.episodes, args.tag or "demo")
        return 0

    # ---------------- learned（暂不可用）----------------
    if args.mode == "learned":
        raise SystemExit(
            "学到的线（570 维）闭环需要 ResNet 编码器，而 "
            "Door_optimal_feature_training.py 里没有任何 torch.save，\n"
            "ResNet 权重从来没存过 —— 现在拿不回那个编码器。\n"
            "想补：在该脚本训练循环结束后加一行\n"
            "    torch.save(resnet_model.state_dict(), './Door/controller/ResNet_door.pt')\n"
            "然后重跑一遍（201 轮）。\n"
            "在此之前，学到的 K 用 eval_open_loop.py 算开环误差，那不需要 ResNet。")

    # ---------------- 真值线闭环 ----------------
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("[info] controller 跑在 %s" % dev)

    if args.compare_random:
        # 两段必须用同一组测试初值，否则比的是初值差异不是权重的差异
        rec_before = Recorder(args.size, args.realtime, cam)
        ctrl_rand = make_controller("random", dev)
        data = run_closed_loop(rec_before, ctrl_rand, args.num_test, args.K, cam)
        print("--- 随机控制器录完，换训好的 ---")

        rec_after = Recorder(args.size, args.realtime, cam)
        ctrl_trained = make_controller(args.controller, dev)
        run_closed_loop(rec_after, ctrl_trained, args.num_test, args.K, cam,
                        seed_data=data)

        for i in range(min(len(rec_before.episodes), len(rec_after.episodes))):
            save_compare(rec_before.episodes[i], rec_after.episodes[i],
                         os.path.join(args.out, "compare_ep%02d.gif" % i), args.fps)
        dump(rec_before.episodes, "before")
        dump(rec_after.episodes, "after")
        return 0

    rec = Recorder(args.size, args.realtime, cam)
    ctrl = make_controller(args.controller, dev)
    run_closed_loop(rec, ctrl, args.num_test, args.K, cam)
    dump(rec.episodes, args.tag or args.mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
