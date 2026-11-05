import sys, importlib
mods = ["numpy","scipy","torch","torchvision","torch_dct","gym","mujoco","mujoco_py","robohive"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        print(f"OK   {m:14s} {v}")
    except Exception as e:
        print(f"FAIL {m:14s} {type(e).__name__}: {e}")

import numpy, torch
print("numpy", numpy.__version__, "(需要 1.23.x)")
print("cuda", torch.cuda.is_available())

import mujoco_py
print("mujoco_py compiled from", mujoco_py.__file__)