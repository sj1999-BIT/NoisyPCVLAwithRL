from pi05_models import jax_model, pytorch_model

import cv2
import os
import torch
import pickle

import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm
from pi05_models import jax_model, pytorch_model
from openpi.policies import droid_policy

"""
model takes in 
'observation/exterior_image_1_left', (h, w, 3)
'observation/wrist_image_left', (h, w, 3)
'observation/joint_position', (7,)
'observation/gripper_position', (1, )
'prompt'
"""

# for i in tqdm(range(100)):
#     example = droid_policy.make_droid_example()
#     with open(f"./tmp/random_samples/{i}.pkl", 'wb') as file:
#         pickle.dump(example, file)

model = pytorch_model(weight_path="./model_weights/rinf_p05.safetensors")

for i in tqdm(range(100)):
    with open(f"./tmp/random_samples/{i}.pkl", 'rb') as file:
        example = pickle.load(file)
    file.close()

    with open(f"./tmp/torch_model_outputs/{i}.npy", 'wb') as f:
        action = model.get_pi05_action(example)["actions"]
        np.save(f, action)
    f.close()

del model

for i in tqdm(range(100)):
    with open(f"./tmp/jax_model_outputs/{i}.npy", 'rb') as file:
        jax_output = np.load(file)
    file.close()
    with open(f"./tmp/torch_model_outputs/{i}.npy", 'rb') as file:
        torch_output = np.load(file)
    file.close()

    abs_diff = np.abs(jax_output - torch_output)
    
    print(f"Arrays are close (rtol=1e-5): {np.allclose(torch_output, jax_output, rtol=1e-5, atol=1e-6)}")
    print(f"\nMax absolute difference: {np.max(abs_diff):.8f}")
    print(f"Mean absolute difference: {np.mean(abs_diff):.8f}")
    print(f"Median absolute difference: {np.median(abs_diff):.8f}")
    print(f"Std of absolute difference: {np.std(abs_diff):.8f}")
    
    print(f"\nMax relative difference: {np.max(abs_diff / (np.abs(torch_output) + 1e-10)):.8f}")
    print(f"Mean relative difference: {np.mean(abs_diff / (np.abs(torch_output) + 1e-10)):.8f}")
    
    print("="*100)

