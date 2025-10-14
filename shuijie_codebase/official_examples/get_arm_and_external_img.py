from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS
from robocasa.utils.env_utils import create_env, run_random_rollouts
import numpy as np
import os

from robocasa.utils.dataset_registry import (
    get_ds_path,
    SINGLE_STAGE_TASK_DATASETS,
    MULTI_STAGE_TASK_DATASETS,
)
from robocasa.scripts.playback_dataset import get_env_metadata_from_dataset
from robosuite.controllers import load_composite_controller_config
import os
import robosuite
import imageio
import numpy as np
from tqdm import tqdm
from termcolor import colored


# choose random task
env_name = np.random.choice(list(ALL_KITCHEN_ENVIRONMENTS))

env = create_env(
    env_name=env_name,
    render_onscreen=False,
    seed=0, # set seed=None to run unseeded
)

video_path= "./tmp/test.mp4"



video_writer = None

video_dir = os.path.dirname(video_path)

num_steps=100

if not os.path.exists(video_dir):
    print(f"The video path {video_path} does not exist, creating the path")
    os.makedirs(video_dir)

if video_path is not None:
    video_writer = imageio.get_writer(video_path, fps=20)

info = {}
num_success_rollouts = 0

obs = env.reset()
for step_i in range(num_steps):
    # sample and execute random action
    # action = np.random.uniform(low=env.action_spec[0], high=env.action_spec[1])
    # Clip action bounds to prevent overflow
    low = np.clip(env.action_spec[0], -1e6, 1e6)
    high = np.clip(env.action_spec[1], -1e6, 1e6)
    action = np.random.uniform(low=low, high=high)

    obs, _, _, _ = env.step(action)

    if video_writer is not None:

        wrist_img = env.sim.render(
            height=512, width=768,
            # camera_name="robot0_agentview_center"
            camera_name="robot0_eye_in_hand"
        )[::-1]

        external_img = env.sim.render(
            height=512, width=768,
            # camera_name="robot0_agentview_center"
            camera_name="robot0_agentview_left"
        )[::-1]

        video_img = np.concatenate([wrist_img, external_img], axis=1)


        video_writer.append_data(video_img)

    if env._check_success():
        num_success_rollouts += 1
        break

if video_writer is not None:
    video_writer.close()
    print(colored(f"Saved video of rollouts to {video_path}", color="yellow"))




# # run rollouts with random actions and save video
# info = run_random_rollouts(
#     env, num_rollouts=3, num_steps=100, video_path="../../tmp/test.mp4"
# )
# print(info)