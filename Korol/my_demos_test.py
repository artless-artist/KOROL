import os, pickle, numpy as np

root = "./Door/Data"
demo = pickle.load(open(os.path.join(root, "Testing.pickle"), "rb"))
print(f"[示范] 条数 {len(demo)}")
print(f"[示范] 键  {sorted(demo[0].keys())}")
print(f"[示范] obs_visual 形状 {np.array(demo[0]['observations_visualization']).shape}（期望 (T, 60)）")
print(f"[示范] 首条长度 {len(demo[0]['actions'])}")

files = [f for f in os.listdir(root) if f.startswith("rgbd_")]
print(f"[图像] 数量 {len(files)}（应等于总步数）")
if files:
    a = np.load(os.path.join(root, "rgbd_0.npy"))
    print(f"[图像] shape={a.shape} dtype={a.dtype} 范围 {a.min():.2f}~{a.max():.2f}")
    print(f"[图像] 前 3 通道均值 {a[...,:3].mean():.3f}（接近 −1 说明全黑 = 渲染失败）")