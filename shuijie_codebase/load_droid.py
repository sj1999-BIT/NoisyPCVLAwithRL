import cv2
import os
import torch
import pickle

import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm
from pi05_models import jax_model, pytorch_model
# from openpi.policies import droid_policy

"""
model takes in 
'observation/exterior_image_1_left', (h, w, 3)
'observation/wrist_image_left', (h, w, 3)
'observation/joint_position', (7,)
'observation/gripper_position', (1, )
'prompt'
"""

model = pytorch_model()

ds = tfds.load("droid_100",
    data_dir="./", split="train")

for episode in ds.take(1):
    for step in tqdm(episode["steps"], desc=f"running steps in"):


        image = step["observation"]["exterior_image_1_left"]
        wrist_image = step["observation"]["wrist_image_left"]

        action = step["action"]
        instruction = step["language_instruction"]

        joint_pos = step["action_dict"]["joint_position"]
        gripper_pos = step["action_dict"]["gripper_position"]

        print(step)

        input = {
            'observation/exterior_image_1_left': torch.from_numpy(image.numpy()),
            'observation/wrist_image_left': torch.from_numpy(wrist_image.numpy()),
            'observation/joint_position': torch.from_numpy(joint_pos.numpy()),
            'observation/gripper_position': torch.from_numpy(gripper_pos.numpy()), 
            'prompt': instruction.numpy()
        }

        print("shuijie debug ", input)

        print("Actions taken by model:", model.get_pi05_action(input)["actions"])

        print("\n\n compare with original action ", step["action_dict"])


