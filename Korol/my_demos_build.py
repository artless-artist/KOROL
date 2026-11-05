# 因为作者的网盘文件没了，现在下不到原项目提供的示范列表与得到的训练数据、训练出的神经网络
# 但是原本的真机轨迹示范在仓库里，我们可以自行处理以得到所需文件
# 回放 DAPG 示范：补 observations_visualization，并渲染 RGB-D 缓存
# 只处理 door 一个任务。产物落在 ./Door/Data/ 下。

import os, pickle
import numpy as np
import robohive
from utils.gym_env import GymEnv

SRC = "../mj_envs/robohive/hand_dapg/dapg/demonstrations/door-v0_demos.pickle"
OUT = "./Door/Data"
IMG_LIMIT = 10 ** 9          # 只验渲染时改成 300，并同时把 num_demo 改成 1

os.makedirs(OUT, exist_ok=True)
demos = pickle.load(open(SRC, "rb"))
print("载入示范", len(demos), "条，共", sum(len(d["actions"]) for d in demos), "步")

e = GymEnv("door-v0")
e.reset()

count = 0
for i, path in enumerate(demos):
    e.set_env_state(path["init_state_dict"])
    obs_vis = []
    for t in range(len(path["actions"])):
        # 与 Koopman_evaluation.py 的拼法逐字一致
        if count < IMG_LIMIT:
            rgb, depth = e.env.mj_render()
            rgb = (rgb.astype(np.uint8) - 128.0) / 128
            rgbd = np.concatenate([rgb, depth[..., None]], axis=2)   # 256×256×4
            np.save(os.path.join(OUT, f"rgbd_{count}.npy"), rgbd)
        # 记录「执行动作之前」的完整 qpos/qvel —— demo_loader 就是这个口径
        obs_vis.append(e.env.get_full_obs_visualization())
        count += 1
        e.step(path["actions"][t])
    path["observations_visualization"] = np.array(obs_vis)
    print(f"  第 {i:3d} 条完成，累计 {count} 帧")

pickle.dump(demos, open(os.path.join(OUT, "Testing.pickle"), "wb"))
print("写入", os.path.join(OUT, "Testing.pickle"), "与", count, "帧图像")