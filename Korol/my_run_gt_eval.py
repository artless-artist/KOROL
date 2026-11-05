import numpy as np, torch, os
import robohive
from utils.gym_env import GymEnv
from utils.Observables import DraftedObservable
from utils.fc_network import FCNetwork
import Koopman_gt_operator_eval as gte

e = GymEnv("door-v0"); e.reset()
Test_data = e.generate_unseen_data_door(200)      # 现场生成，与数据包无关

# 控制器：结构必须与权重一致
Controller = FCNetwork([56, 112, 112, 56, 28], nonlinearity="relu")
Controller.load_state_dict(torch.load("./Door/controller/NN_controller_best.pt", map_location="cpu"))
Controller.eval()

K = np.load("./gt_koopmanMatrix.npy")
print("K shape", K.shape)          # (546, 546)

# 真值线的物体维数：3(门体位置) + 1(门角速度) + 3(初始门体位置)
num_obj = 3 + 1 + 3
gte.door_koopman_policy_control_unseenTest("door-v0", Controller, DraftedObservable(28, num_obj), K, Test_data, 28, num_obj)