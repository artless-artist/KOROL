from cProfile import label
from glob import escape
from attr import asdict
import torch
#import mj_envs
import click 
import json
import os
import numpy as np
import gym
import pickle
from tqdm import tqdm
from utils.gym_env import GymEnv
from utils.Observables import *
from utils.Koopman_evaluation import *
from utils.Controller import *
from utils.quatmath import quat2euler, euler2quat
from utils.coord_trans import ori_transform, ori_transform_inverse
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.linalg import logm
import scipy
import sys
import os
import random
import time 
import shutil
import robohive
from utils.demo_loader import *
from utils.Koopman_gt_evaluation import * 

# 任务名由 main() 的参数决定，这里只是模块级默认值（仅用于 print/调试）
env_name = "Door"


def main(task="Door", matrix_file="gt_koopmanMatrix.npy", num_test=200):
    """评估一行 Koopman 矩阵。

    task         : "Door" / "Tool" / "Relocate" / "Reorientation"
    matrix_file  : 待评估的 K，相对当前工作目录。本脚本只吃**真值口径**的 K——物体状态
                   拼成 objpos + objvel + handle_init（Door 上是 7 维），字典维数由任务表
                   里的 num_obj 决定，不是 KOROL 学到的 rgbd_feature（8 维）。
                   文件名有个坑：Door 的 gt_train 存成 gt_koopmanMatrix.npy（第 84 行），
                   另外三个任务存成 koopmanMatrix.npy（145/191/249 行）；而 Door/ 下那份
                   koopmanMatrix.npy 是 KOROL 学到的 570 维，不能喂这里。
                   靠脚本的维度自检判别，别靠文件名猜。
    num_test     : 现场生成的测试初值数量（每条闭环 70 步）
    """
    global env_name
    env_name = task

    num_demo = 200 # if num_demo != 0 -> we manually define the num of demo for testing the sample efficiency
    folder_name = os.getcwd()

    if (env_name == "Door"):
        e = GymEnv("door-v0")
        e.reset()
        Testing_data = e.generate_unseen_data_door(num_test) 
        num_obj = 7 #implicit_obj_dim # + num_objvel + implicit_obj_dim
        num_hand = 28
        Controller_loc = './Door/controller/NN_controller_best.pt'
    elif (env_name == "Tool"):
        e = GymEnv("hammer-v0")
        e.reset()
        Testing_data = e.generate_unseen_data_hammer(num_test) 
        num_obj = 6 #implicit_obj_dim # + num_objvel + implicit_obj_dim
        num_hand = 26
        Controller_loc = './Tool/controller/NN_controller_best.pt'
    elif (env_name == "Relocate"):
        e = GymEnv("relocate-v0")
        e.reset()
        Testing_data = e.generate_unseen_data_relocate(num_test) 
        num_obj = 6 #implicit_obj_dim # + num_objvel + implicit_obj_dim
        num_hand = 30
        Controller_loc = './Relocation/controller/NN_controller_best.pt'
    elif (env_name == "Reorientation"):
        e = GymEnv("pen-v0")
        e.reset()
        Testing_data = e.generate_unseen_data_reorientate(num_test) 
        num_obj = 6 #implicit_obj_dim # + num_objvel + implicit_obj_dim
        num_hand = 24
        Controller_loc = './Reorientation/controller/NN_controller_best.pt'
    else:
        raise SystemExit("未知 task: %r，可选 Door / Tool / Relocate / Reorientation" % (env_name,))

    # define the input and outputs of the neural network
    NN_Input_size = 2 * num_hand
    NN_Output_size = num_hand
    NN_size = [NN_Input_size, 2 * NN_Input_size, 2 * NN_Input_size, NN_Input_size, NN_Output_size]  # define a neural network to learn the PID mapping
    Controller = FCNetwork(NN_size, nonlinearity='relu')
    Controller.load_state_dict(torch.load(Controller_loc, map_location="cpu"))
    Controller.eval() # eval mode

    print("Trained drafted koopman matrix loaded!")
    matrix_path = os.path.join(folder_name, matrix_file)
    if not os.path.exists(matrix_path):
        npy = sorted(f for f in os.listdir(folder_name) if f.endswith(".npy"))
        raise SystemExit("找不到 %s\n当前目录下的 .npy：%s\n（真值线的 K 默认叫 gt_koopmanMatrix.npy）" % (matrix_path, npy))
    cont_koopman_operator = np.load(matrix_path) # matrix_file
    print("K loaded from %s, shape = %s" % (matrix_file, cont_koopman_operator.shape))
    Koopman = DraftedObservable(num_hand, num_obj)

    # 维度自检：本脚本只评估真值线的 K，传学到的线会在 np.dot 里报一句看不懂的
    # "shapes (570,570) and (546,) not aligned"，不如在这里说清楚。
    n_obs = Koopman.compute_observable(num_hand, num_obj)
    if cont_koopman_operator.shape != (n_obs, n_obs):
        raise SystemExit(
            "K 的形状是 %s，但 %s 的真值口径字典是 %dx%d（num_obj=%d）。\n"
            "本脚本只评估真值线：物体状态 = objpos + objvel + handle_init，共 %d 维。\n"
            "学到的线 K 是 num_obj=8（8 维 rgbd_feature，570x570），不走这里——"
            "它唯一的闭环评估入口是 Door_optimal_feature_training.py:145，"
            "绑在 201 轮训练循环里（每 50 轮一次），没法拿一个现成的 koopmanMatrix.npy 单独评。\n"
            "想先判断学到的 K 好不好，用 my_eval_K.py 算开环误差。"
            % (cont_koopman_operator.shape, env_name, n_obs, n_obs, num_obj, num_obj)
        )
    
    if (env_name == "Door"):
        door_koopman_policy_control_unseenTest("door-v0", Controller, Koopman, cont_koopman_operator, Testing_data, num_hand, num_obj)
    elif (env_name == "Tool"):
        tool_koopman_policy_control_unseenTest("hammer-v0", Controller, Koopman, cont_koopman_operator, Testing_data, num_hand, num_obj)
    elif (env_name == "Relocate"):
        relocate_koopman_policy_control_unseenTest("relocate-v0", Controller, Koopman, cont_koopman_operator, Testing_data, num_hand, num_obj)
    elif (env_name == "Reorientation"):
        reori_koopman_policy_control_unseenTest("pen-v0", Controller, Koopman, cont_koopman_operator, Testing_data, num_hand, num_obj)

if __name__ == '__main__':
    # 命令行参数：把原来写死在工作目录 / 任务分支里的三项提出来
    #   --task      选任务（Door / Tool / Relocate / Reorientation），决定 env id、num_hand、
    #               num_obj、控制器路径、走哪个 *_unseenTest
    #   --K         要评估的那个 K 矩阵文件名（np.load 得到的 n*n 矩阵），相对当前工作目录。
    #               只接受真值口径的 K：Door 是 gt_koopmanMatrix.npy（546），Tool/
    #               Relocate/Reorientation 的 gt_train 存成 koopmanMatrix.npy（466/588/411）。
    #               KOROL 学到的 570 维 K 不走这里，脚本会在维度自检处退出并提示改走 my_eval_K.py
    #   --num-test  现场生成的测试初值条数，每个初值再闭环推 70 步；直接传给
    #               generate_unseen_data_door/hammer/relocate/reorientate
    # 用法（必须在 KOROL/Korol 下执行，因为 K 按 cwd 解析）：
    #   python Koopman_gt_operator_eval.py --task Door --K gt_koopmanMatrix.npy --num-test 200
    #   python Koopman_gt_operator_eval.py --task Tool --K koopmanMatrix.npy --num-test 200
    import argparse
    parser = argparse.ArgumentParser(description="评估一行 Koopman 矩阵在某个任务上的成功率")
    parser.add_argument("--task", default="Door",
                        choices=["Door", "Tool", "Relocate", "Reorientation"])
    parser.add_argument("--K", dest="matrix_file", default="gt_koopmanMatrix.npy",
                        help="要评估的 K 矩阵文件名，相对当前工作目录。只接受真值口径的 K："
                             "Door 是 gt_koopmanMatrix.npy；Tool/Relocate/Reorientation 的 "
                             "gt_train 存成 koopmanMatrix.npy")
    parser.add_argument("--num-test", dest="num_test", type=int, default=200,
                        help="测试初值条数（每条再闭环 70 步）；原版四个任务都写死 1000，"
                             "正式对照建议 1000，调试可降到 50")
    args = parser.parse_args()
    main(args.task, args.matrix_file, args.num_test)
