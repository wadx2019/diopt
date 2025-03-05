import torch
import pandas as pd
from phc.utils.torch_h1_humanoid_batch import Humanoid_Batch
from phc.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
    SMPL_BONE_ORDER_NAMES,
)

positon_path= "data/position/example_y.csv"
root_rot_path= "data/gt_root_rot/example_root_rot.csv"
position_data = pd.read_csv(positon_path).values
position_data = torch.tensor(position_data)
# print(position_data.shape)
class retargeting_problem:
    def __init__(
        self,
        positon_path="./data/position/example_y.csv",
        root_rot_path="./data/gt_root_rot/example_root_rot.csv",
        root_trans_offset_path="./data/root_trans_offset/example_root_trans.csv"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.positon_path = positon_path
        self.root_rot_path = root_rot_path
        self.root_trans_offset_path = root_trans_offset_path

        self.position_data = pd.read_csv(self.positon_path).values
        self.position_data = torch.tensor(self.position_data)
        self.frames = self.position_data.shape[0]
        print("frames: ", self.frames)
        self.position = position_data.view(self.frames, 24, 3)
        print("position shape: ", self.position.shape)

        self.gt_root_rot = torch.tensor(pd.read_csv(self.root_rot_path).values)
        print("gt_root_rot shape: ", self.gt_root_rot.shape)

        self.root_trans_offset = torch.tensor(pd.read_csv(root_trans_offset_path).values)
        print("root_trans_offset shape: ", self.root_trans_offset.shape)
        self.h1_joint_names = [
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
        self.h1_joint_pick = [
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
        self.smpl_joint_pick = [
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
        self.h1_joint_names_augment = self.h1_joint_names + [
            "left_hand_link",
            "right_hand_link",
        ]

        # index for h1_joint_pick in h_1_joint_name_augment
        self.h1_joint_pick_idx = [
            self.h1_joint_names_augment.index(j) for j in self.h1_joint_pick
        ]
        # index for smpl_joint_pick in SMPL_BONE_ORDER_NAMES
        self.smpl_joint_pick_idx = [
            SMPL_BONE_ORDER_NAMES.index(j) for j in self.smpl_joint_pick
        ]

        self.h1_rotation_axis = torch.tensor(
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
        # H1机器人19个关节的旋转上下限
        self.robot_joints_range = torch.tensor(
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

    def forward_kinematics(self, joints_pos):
        h1_fk = Humanoid_Batch(device=self.device)
        print('1',self.gt_root_rot.shape)
        print('2',self.gt_root_rot[None, :, None].shape)
        print('3',(self.h1_rotation_axis * joints_pos).shape)
        print('4', torch.zeros((1, self.frames, 2, 3)).shape)
        pose_aa_h1 = torch.cat([ self.gt_root_rot[None, :, None], self.h1_rotation_axis * joints_pos, torch.zeros((1, self.frames, 2, 3)),], axis=2,).to(self.device)
        return h1_fk.fk_batch(pose_aa_h1.to(torch.float32),  self.root_trans_offset[None,].to(torch.float32).to(self.device))

    def problem(self, joints_pos):

        loss = ( self.forward_kinematics(joints_pos)["global_translation_extend"][ :, :, self.h1_joint_pick_idx] - self.position[:, self.smpl_joint_pick_idx].to(self.device))
        ## 以上loss的约束条件是joints_pos位于self.robot_joints_range之间
        ## joints_pos是一个表征19个关节位置的变量
        print('fk:', self.forward_kinematics(joints_pos)["global_translation_extend"].shape)
        print('fk_s:',
              self.forward_kinematics(joints_pos)["global_translation_extend"][:, :, self.h1_joint_pick_idx].shape)
        print('spmlpos:',
              self.position.shape)
        print('spmlpos_s:',
              self.position[:, self.smpl_joint_pick_idx].shape)
        # print("loss: ", loss.shape)
        loss = loss.norm(dim=-1).sum(dim=-1).view(-1,1)# 对19个关节以及所有帧取平均
        print("loss: ", loss.shape)
        loss = loss.mean()
        # print("loss: ", loss.shape)
        loss.backward()

        return joints_pos.grad

test = retargeting_problem()
test_loss_grad = test.problem( torch.zeros((1, 1836, 19, 1), requires_grad=True))
  # 应该是 True
print(test_loss_grad.shape)


