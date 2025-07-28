from copy import deepcopy

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor

try:
    from utils import * 
except:
    from . utils import *
    
@register_agent()
class OpenArm(BaseAgent):
    # ManiSkill URDF
    uid = "openarm"
    # urdf_path = "/home/h-ito/openarm_Isaac_description/urdf/openarm_v1_0.urdf"
    _urdf_path = "/home/h-ito/ros2_ws/openarm_pickup.urdf"
    urdf_path = "/tmp/openarm.urdf"
    pkg_root = "/home/h-ito/ros2_ws/src/openarm_description"
    replace_package(_urdf_path, urdf_path, pkg_root)
    
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2, dynamic_friction=2, restitution=0.0)
        ),
        link=dict(
            openarm_left_finger=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            openarm_right_finger=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1)
        )
    )
    
    # disable self (link) collisions
    disable_self_collisions = True

    keyframes = dict(
        reset=Keyframe(
            qpos=np.array([
                -1.561,
                -0.174,
                -1.561,
                1.532,
                0,
                0,
                -1.347,
                0,
                0,
            ]),
            pose=sapien.Pose(p=[0,0,0])
        )
    )
    
    arm_joint_names = [
        "openarm_joint1",
        "openarm_joint2",
        "openarm_joint3",
        "openarm_joint4",
        "openarm_joint5",
        "openarm_joint6",
        "openarm_joint7",
    ]
    gripper_joint_names = [
        "openarm_finger_joint1"
    ]
    
    ee_link_name = "openarm_hand_tcp"

    arm_stiffness = 2000
    arm_damping = 25
    arm_force_limit = 100

    gripper_stiffness = 2000
    gripper_damping = 25
    gripper_force_limit = 100
    
    @property
    def _controller_configs(self):
        #---- #
        # Arm # 
        # --- # 
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=None,
            pos_upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,
            normalize_action=False,
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # PD joint velocity
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,  # this might need to be tuned separately
            self.arm_force_limit,
        )

        # PD joint position and velocity
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )

        # ------- #
        # Gripper #
        # ------- #
        # NOTE(jigu): IssacGym uses large P and D but with force limit
        # However, tune a good force limit to have a good mimic behavior
        gripper_pd_mimic_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=[0],
            upper=[0.0451],
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            mimic={"openarm_finger_joint2": {"joint": "openarm_finger_joint1", "multiplier": 1.0}}
        )
        gripper_pd_joint_pos = PDJointPosControllerConfig(
            self.gripper_joint_names,
            lower=[0],
            upper=[0.0451],
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
        )
        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos,
                gripper=gripper_pd_joint_pos
            ),
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                gripper=gripper_pd_joint_pos
            ),
            pd_ee_delta_pos=dict(
                arm=arm_pd_ee_delta_pos,
                gripper=gripper_pd_joint_pos
            ),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose,
                gripper=gripper_pd_joint_pos
            ),
            pd_ee_pose=dict(
                arm=arm_pd_ee_pose,
                gripper=gripper_pd_joint_pos
            ),
            # TODO(jigu): how to add boundaries for the following controllers
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos,
                gripper=gripper_pd_joint_pos
            ),
            pd_ee_target_delta_pos=dict(
                arm=arm_pd_ee_target_delta_pos,
                gripper=gripper_pd_joint_pos
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose,
                gripper=gripper_pd_joint_pos
            ),
            # Caution to use the following controllers
            pd_joint_vel=dict(
                arm=arm_pd_joint_vel,
                gripper=gripper_pd_joint_pos
            ),
            pd_joint_pos_vel=dict(
                arm=arm_pd_joint_pos_vel,
                gripper=gripper_pd_joint_pos
            ),
            pd_joint_delta_pos_vel=dict(
                arm=arm_pd_joint_delta_pos_vel,
                gripper=gripper_pd_joint_pos
            ),
        )
        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def _after_init(self):
        super()._after_init()

        self.finger1_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "openarm_left_finger"
        )
        self.finger2_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "openarm_right_finger"
        )
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )
        
    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the robot is grasping an object
                
        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
                
        axis_l = -self.finger1_link.pose.to_transformation_matrix()[..., :3, 2]
        axis_r = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 2]

        lforce_vec = -l_contact_forces
        rforce_vec = -r_contact_forces

        lforce = torch.linalg.norm(lforce_vec, axis=1)
        rforce = torch.linalg.norm(rforce_vec, axis=1)

        langle = common.compute_angle_between(axis_l, lforce_vec)
        rangle = common.compute_angle_between(axis_r, rforce_vec)
        
        lflag = torch.logical_and(lforce >= min_force, torch.rad2deg(langle) <= max_angle)
        rflag = torch.logical_and(rforce >= min_force, torch.rad2deg(rangle) <= max_angle)
    
        return torch.logical_and(lflag, rflag)
        
    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-2]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose
