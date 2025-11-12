from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
from openpi.policies import droid_policy
from openpi.training import checkpoints as _checkpoints
from openpi import transforms as _transforms
from openpi.shared import nnx_utils

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


# Add absolute path
# sys.path.append("../openpi/")

print("loading model")

config = _config.get_config("pi05_droid")
checkpoint_dir = "./model_weights/pi05_droid"
checkpoint_dir = download.maybe_download(str(checkpoint_dir))

# Create a trained policy.
# policy = policy_config.create_trained_policy(config, checkpoint_dir)


pi05_config = pi0_config.Pi0Config(action_horizon=15, pi05=True)
model = pi05_config.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))

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



is_pytorch=False

pytorch_device='cuda' if is_pytorch else None

print("model loaded")

def get_jax_pi05_action(obs):
    

    metadata=config.policy_metadata

    sample_actions = nnx_utils.module_jit(model.sample_actions)
    rng = jax.random.key(0)
    

    print("processing inputs")

    # # Run inference on a dummy example.
    # obs = droid_policy.make_droid_example()


    # Make a copy since transformations may modify the inputs in place.
    inputs = jax.tree.map(lambda x: x, obs)
    inputs = input_transform(inputs)

    # print(f"inputs is now {inputs}")


    # Make a batch and convert to jax.Array.
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    rng = jax.random.key(0)
    rng, sample_rng_or_pytorch_device = jax.random.split(rng)


    # Prepare kwargs for sample_actions
    sample_kwargs={}
    sample_kwargs = dict(sample_kwargs)


    observation = _model.Observation.from_dict(inputs)
    start_time = time.monotonic()
    outputs = {
        "state": inputs["state"],
        "actions": sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
    }
    model_time = time.monotonic() - start_time

    print("processing outputs")

    outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
    outputs = output_transforms(outputs)
    outputs["policy_timing"] = {
        "infer_ms": model_time * 1000,
    }


    # result = policy.infer(example)['actions']

    # print(f"shuijie debug: policy.infer(example) {outputs}")
    return outputs

del model

# action_chunk = policy.infer(example)["actions"]