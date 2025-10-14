import cv2

import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm

ds = tfds.load("droid_100",
    data_dir="D:/Shuijie/PHD school/research/code/droid_policy_learning/data/", split="train")

for episode in ds.take(1):
    for step in tqdm(episode["steps"], desc=f"running steps in"):
        image = step["observation"]["exterior_image_1_left"]
        wrist_image = step["observation"]["wrist_image_left"]

        action = step["action"]
        instruction = step["language_instruction"]
        print(f"{instruction.numpy().decode('utf-8')}: action {action.numpy()}")
        print()
        print()
        cv2.imshow(wrist_image.numpy())