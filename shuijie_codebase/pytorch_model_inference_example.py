"""
The main goal now is to first 
1. load the jax model based on the original jax pipeline
2. create a pyotrch version of the model.
3. save the model weights
4. run the torch model
"""

from openpi.training import config as _config
from openpi.shared import download
from openpi.policies import droid_policy
from openpi.training import checkpoints as _checkpoints
from openpi import transforms as _transforms
from openpi.models_pytorch import pi0_pytorch



import openpi.models.pi0_config as pi0_config
import openpi.transforms as transforms
import openpi.models.model as _model
import openpi.shared.download as download

import sys
import os
import jax
import time

import torch
import numpy as np
import jax.numpy as jnp

# The JAX model uses NNX, so we need to get the state
from flax import nnx
from flax import traverse_util
from tqdm import tqdm

import safetensors

# first load the model in jax format
sys.path.append("../openpi/")

print("loading model in pytorch")

config = _config.get_config("pi05_droid")
checkpoint_dir = "./model_weights/pi05_droid/"
checkpoint_dir = download.maybe_download(str(checkpoint_dir))

weight_path = "./model_weights/pi05_droid_pytorch_model.safetensors"

model = pi0_pytorch.PI0Pytorch(config=config.model)
model.load_state_dict(safetensors.torch.load_file(weight_path))
model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

data_config = config.data.create(config.assets_dirs, config.model)
norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)


repack_transforms = transforms.Group()

default_prompt = None
sample_kwargs = None
noise = None


input_transform = _transforms.compose([
    *repack_transforms.inputs,
    transforms.InjectDefaultPrompt(default_prompt),
    *data_config.data_transforms.inputs,
    transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
    *data_config.model_transforms.inputs,
])

output_transforms = _transforms.compose([
    *data_config.model_transforms.outputs,
    transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
    *data_config.data_transforms.outputs,
    *repack_transforms.outputs,
])

sample_kwargs={}

metadata=config.policy_metadata

# sample_actions = nnx_utils.module_jit(model.sample_actions)
rng = jax.random.key(0)

is_pytorch=False

pytorch_device='cuda' if is_pytorch else None

print("model loaded")

print("processing inputs")

# Run inference on a dummy example.
obs = droid_policy.make_droid_example()

# Make a copy since transformations may modify the inputs in place.
inputs = jax.tree.map(lambda x: x, obs)
inputs = input_transform(inputs)



 # Make a batch and convert to jax.Array.
inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(pytorch_device)[None, ...], inputs)
sample_rng_or_pytorch_device = pytorch_device

 # Prepare kwargs for sample_actions
sample_kwargs = dict(sample_kwargs)


observation = _model.Observation.from_dict(inputs)
start_time = time.monotonic()
outputs = {
    "state": inputs["state"],
    "actions": model.sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
}
model_time = time.monotonic() - start_time

print("processing outputs")

outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
outputs = output_transforms(outputs)
outputs["policy_timing"] = {
    "infer_ms": model_time * 1000,
}



# result = policy.infer(example)['actions']

print(f"shuijie debug: policy.infer(example) {outputs}")

del model

# action_chunk = policy.infer(example)["actions"]