from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS
from robocasa.utils.env_utils import create_env
import numpy as np

# choose random task
env_name = np.random.choice(list(ALL_KITCHEN_ENVIRONMENTS))

env = create_env(
    env_name=env_name,
    render_onscreen=True,
    seed=0, # set seed=None to run unseeded
)

print(env.robots[0].print_action_info())

# reset the environment
env.reset()

# get task language
lang = env.get_ep_meta()["lang"]
# print("Instruction:", lang)

for i in range(1):

    # print(env.get_ep_meta())
    # for key in env.get_ep_meta().keys():
    #     print(f"{key}: {env.get_ep_meta()[key]}\n")

    # print(*env.action_spec)

    # [robosuite INFO] Action Dimensions: [right: 6 dim, right_gripper: 1 dim, base: 3 dim, torso: 1 dim] (robot.py:972)
    # [robosuite INFO] Action Indices: [right: 0:6, right_gripper: 6:7, base: 7:10, torso: 10:11] (robot.py:975)
    action = np.random.randn(*env.action_spec[0].shape)
    # action[0: 6] = np.random.randn(6)

    # action is a 12 dim vector, -1 to 1
    # what if we pass in a 3 dim

    # print("action shape:", action.shape)
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
    obs, reward, done, info = env.step(action)  # take action in the environment

    # obs is an ordered_dict
    # print(f"obs: {obs.keys()}")
    # print(f"info: {info}")

    env.render()  # render on display