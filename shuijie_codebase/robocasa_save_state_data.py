from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS
from robocasa.utils.env_utils import create_env, run_random_rollouts

from tqdm import tqdm


import pickle
import numpy as np
import cv2
import imageio

"""
obs is an ordered_dict
Robot State Information
    Joint-related observations:
    
    robot0_joint_pos: Joint angles (7 values for a 7-DOF robot arm) in radians
    robot0_joint_pos_cos/sin: Sine and cosine of joint angles (for neural network processing)
    robot0_joint_vel: Joint velocities (rad/s)
    robot0_joint_acc: Joint accelerations (rad/s²)
    
    End-effector (gripper) state:
    
    robot0_eef_pos: End-effector position in 3D space [x, y, z] (meters)
    robot0_eef_quat: End-effector orientation as quaternion [w, x, y, z]
    robot0_eef_quat_site: Alternative quaternion representation for end-effector
    
    Gripper state:
    
    robot0_gripper_qpos: Gripper finger positions (2 values for open/close)
    robot0_gripper_qvel: Gripper finger velocities
    
    Robot base information:
    
    robot0_base_pos: Robot base position in world coordinates
    robot0_base_quat: Robot base orientation
    robot0_base_to_eef_pos/quat: Relative position/orientation from base to end-effector
    
    Environment Objects
    Drawer object:
    
    drawer_obj_pos/quat: Drawer position and orientation
    drawer_obj_to_robot0_eef_pos/quat: Relative position/orientation from drawer to robot end-effector
    
    Counter objects (3 counters):
    
    distr_counter_[1,2,3]_pos/quat: Position and orientation of each counter
    distr_counter_[1,2,3]_to_robot0_eef_pos/quat: Relative poses from each counter to robot end-effector
    
    Aggregated State Vectors
    robot0_proprio-state (66 values):
    Concatenated proprioceptive information including:
    
    All joint positions, velocities, accelerations
    End-effector pose, gripper state
    Base information and relative poses
    
    object-state (56 values):
    Concatenated object information including:
    
    All object positions and orientations
    All relative poses between objects and robot
    
    Value Formats
    
    Positions: 3D coordinates [x, y, z] in meters
    Quaternions: [w, x, y, z] format for rotations
    Joint angles: Radians
    Velocities/Accelerations: Standard SI units (rad/s, rad/s², m/s, etc.)
    All values: 32-bit or 64-bit floating point numbers

reward is a single value
done is a bool
info is empty dict
"""

def generate_pi05_inputs(obs, env, current_prompt=""):
    joint_position_npy = obs["robot0_joint_pos"]

    gripper_position_npy = obs["robot0_gripper_qpos"]

    video_img_main_left = env.sim.render(
        height=512, width=768, camera_name="robot0_agentview_center"
    )[::-1]

    video_img_wrist = env.sim.render(
        height=512, width=768, camera_name="robot0_eye_in_hand"
    )[::-1]

    return {
        'observation/exterior_image_1_left': video_img_main_left,
        'observation/wrist_image_left': video_img_wrist,
        'observation/joint_position': joint_position_npy,
        'observation/gripper_position': gripper_position_npy, 
        'prompt': current_prompt
    }


# choose random task
env_name = np.random.choice(list(ALL_KITCHEN_ENVIRONMENTS))

env = create_env(
    env_name=env_name,
    render_onscreen=False,
    seed=0, # set seed=None to run unseeded
)


num_steps = 100
num_rollouts = 1



info = {}
num_success_rollouts = 0
for rollout_i in range(num_rollouts):
    obs = env.reset()
    for step_i in tqdm(range(num_steps)):
        # sample and execute random action
        action = np.random.uniform(low=env.action_spec[0], high=env.action_spec[1])
        obs, _, _, _ = env.step(action)

        # video_img_main_left = env.sim.render(
        #     height=512, width=768, camera_name="robot0_agentview_center"
        # )[::-1]

        # video_img_wrist = env.sim.render(
        #     height=512, width=768, camera_name="robot0_eye_in_hand"
        # )[::-1]

        # joint_position_npy = obs["robot0_joint_pos"]

        # gripper_position_npy = obs["robot0_gripper_qpos"]

        # frame = np.concatenate((video_img_main_left, video_img_wrist), axis=1)

        pi05_input = generate_pi05_inputs(obs, env)

        # Save
        with open(f'./tmp/{step_i}.pkl', 'wb') as f:
            pickle.dump(pi05_input, f)




        # print(frame.shape)

        # cv2.imwrite(f"./tmp/{step_i}.png", frame)


        if env._check_success():
            num_success_rollouts += 1
            break

info["num_success_rollouts"] = num_success_rollouts


print(info)