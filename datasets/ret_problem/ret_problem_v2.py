import torch
import pandas as pd
from phc.smpllib.smpl_parser import (
    SMPL_BONE_ORDER_NAMES,
)
import pickle

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from utils_test_2 import RetargetingProblem
import multiprocessing
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

multiprocessing.set_start_method("spawn")

# positon_path="./data/position/example_y.csv"
# root_rot_path="./data/gt_root_rot/example_root_rot.csv"
# root_trans_offset_path="./data/root_trans_offset/example_root_trans.csv"

positon_path= "data/position/example_y_2.csv"
root_rot_path= "data/gt_root_rot/example_root_rot_2.csv"
root_trans_offset_path= "data/root_trans_offset/example_root_trans_2.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


position_data = pd.read_csv(positon_path).values
position_data = torch.tensor(position_data)
frames = position_data.shape[0]
print("frames: ", frames)

position = position_data.view(frames, 24, 3)
# print("position shape: ", position.shape)

gt_root_rot = torch.tensor(pd.read_csv(root_rot_path).values)
# print("gt_root_rot shape: ", gt_root_rot.shape)

root_trans_offset = torch.tensor(pd.read_csv(root_trans_offset_path).values)

h1_joint_names = [
            "pelvis",
            "left_hip_yaw_link",
            "left_hip_roll_link",
            "left_hip_pitch_link",
            "left_knee_link",
            "left_ankle_link",
            "right_hip_yaw_link",
            "right_hip_roll_link",
            "right_hip_pitch_link",
            "right_knee_link",
            "right_ankle_link",
            "torso_link",
            "left_shoulder_pitch_link",
            "left_shoulder_roll_link",
            "left_shoulder_yaw_link",
            "left_elbow_link",
            "right_shoulder_pitch_link",
            "right_shoulder_roll_link",
            "right_shoulder_yaw_link",
            "right_elbow_link",
        ]
h1_joint_pick = [
            "pelvis",
            "left_knee_link",
            "left_ankle_link",
            "right_knee_link",
            "right_ankle_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_hand_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_hand_link",
        ]
smpl_joint_pick = [
            "Pelvis",
            "L_Knee",
            "L_Ankle",
            "R_Knee",
            "R_Ankle",
            "L_Shoulder",
            "L_Elbow",
            "L_Hand",
            "R_Shoulder",
            "R_Elbow",
            "R_Hand",
        ]
h1_joint_names_augment = h1_joint_names + [
            "left_hand_link",
            "right_hand_link",
        ]

# index for h1_joint_pick in h_1_joint_name_augment
h1_joint_pick_idx = [h1_joint_names_augment.index(j) for j in h1_joint_pick
        ]
# index for smpl_joint_pick in SMPL_BONE_ORDER_NAMES
smpl_joint_pick_idx = [
            SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick
        ]

h1_rotation_axis = torch.tensor(
            [
                [
                    [0, 0, 1],  # l_hip_yaw
                    [1, 0, 0],  # l_hip_roll
                    [0, 1, 0],  # l_hip_pitch
                    [0, 1, 0],  # kneel
                    [0, 1, 0],  # ankle
                    [0, 0, 1],  # r_hip_yaw
                    [1, 0, 0],  # r_hip_roll
                    [0, 1, 0],  # r_hip_pitch
                    [0, 1, 0],  # kneel
                    [0, 1, 0],  # ankle
                    [0, 0, 1],  # torso
                    [0, 1, 0],  # l_shoulder_pitch
                    [1, 0, 0],  # l_roll_pitch
                    [0, 0, 1],  # l_yaw_pitch
                    [0, 1, 0],  # l_elbow
                    [0, 1, 0],  # r_shoulder_pitch
                    [1, 0, 0],  # r_roll_pitch
                    [0, 0, 1],  # r_yaw_pitch
                    [0, 1, 0],  # r_elbow
                ]
            ]
        )
###################
# h marix
##################
robot_joints_range = torch.tensor(
            [
                [-0.4300, 0.4300],  # left_hip_yaw_link
                [-0.4300, 0.4300],  # left_hip_roll_link
                [-1.5700, 1.5700],  # left_hip_pitch_link
                [-0.2600, 2.0500],  # left_knee_link
                [-0.8700, 0.5200],  # left_ankle_link
                [-0.4300, 0.4300],  # right_hip_yaw_link
                [-0.4300, 0.4300],  # right_hip_roll_link
                [-1.5700, 1.5700],  # right_hip_pitch_link
                [-0.2600, 2.0500],  # right_knee_link
                [-0.8700, 0.5200],  # right_ankle_link
                [-2.3500, 2.3500],  # torso_link
                [-2.8700, 2.8700],  # left_shoulder_pitch_link
                [-0.3400, 3.1100],  # left_shoulder_roll_link
                [-1.3000, 4.4500],  # left_shoulder_yaw_link
                [-1.2500, 2.6100],  # left_elbow_link
                [-2.8700, 2.8700],  # right_shoulder_pitch_link
                [-3.1100, 0.3400],  # right_shoulder_roll_link
                [-4.4500, 1.3000],  # right_shoulder_yaw_link
                [-1.2500, 2.6100],  # right_elbow_link
            ],
        )
# print("robot_joints_range: ", robot_joints_range.shape)
min_values = robot_joints_range[:, 0]
max_values = robot_joints_range[:, 1]
h = torch.cat((min_values, max_values)).reshape(-1, 1)
# h = combined_values.numpy()

def G_ineq():
    G = torch.zeros((38, 19))
    G[:19, :19] = -torch.eye(19)
    G[19:38, :19] = torch.eye(19)
    return G

G=G_ineq()

R = h1_rotation_axis

R_root = gt_root_rot

R_root_trans = root_trans_offset

X = position[:, smpl_joint_pick_idx].reshape(frames, -1)

print('G shape: ', G.shape)
print('h shape: ', h.shape)
print('R shape: ', R.shape)
print('R_root shape: ', R_root.shape)
print('R_root_trans shape: ', R_root_trans.shape)
print("X shape: ", X.shape)

select_idx =h1_joint_pick_idx
problem = RetargetingProblem(frames, G, h, R, R_root, R_root_trans, select_idx,  X)
problem.calc_Y()
print(len(problem.Y))
print(problem.Y[0])

num_var = G.shape[1]
num_ineq = G.shape[0]
num_eq = 0
num_examples = frames

with open("./Retargeting_dataset_var{}_ineq{}_eq{}_ex{}_v2".format(num_var, num_ineq, num_eq, num_examples), 'wb') as f:
    pickle.dump(problem, f)