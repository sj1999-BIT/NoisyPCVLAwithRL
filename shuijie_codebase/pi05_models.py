from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
from openpi.policies import droid_policy
from openpi.training import checkpoints as _checkpoints
from openpi import transforms as _transforms
from openpi.shared import nnx_utils
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
import safetensors
import numpy as np
import jax.numpy as jnp

from abc import ABC, abstractmethod


class pi05_model(ABC):
    def __init__(self, 
                 config,
                 checkpoint_dir
                 ):
        self.config = _config.get_config(config)
        self.checkpoint_dir = download.maybe_download(str(checkpoint_dir))

        self.sample_kwargs = {}
        

        self.pi05_config = pi0_config.Pi0Config(action_horizon=15, pi05=True)



        data_config = self.config.data.create(self.config.assets_dirs, self.config.model)
        norm_stats = _checkpoints.load_norm_stats(self.checkpoint_dir / "assets", data_config.asset_id)

        repack_transforms = transforms.Group()

        default_prompt = None

        self.input_transform = _transforms.compose([
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ])

        self.output_transforms = _transforms.compose([
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ])

        

    
    @abstractmethod
    def get_pi05_action(self, obs):
        pass


class jax_model(pi05_model):
    def __init__(self):
        super().__init__(
            config = "pi05_droid",
            checkpoint_dir="./model_weights/pi05_droid"
            )
        
        self.rng = jax.random.key(0)
        self.model = self.pi05_config.load(_model.restore_params(self.checkpoint_dir / "params", dtype=jnp.bfloat16))
        self.sample_actions = nnx_utils.module_jit(self.model.sample_actions)

    def get_pi05_action(self, obs):
        

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self.input_transform(inputs)

        # print(f"inputs is now {inputs}")


        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self.rng, sample_rng_or_pytorch_device = jax.random.split(self.rng)


        # Prepare kwargs for sample_actions
        self.sample_kwargs = dict(self.sample_kwargs)


        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self.sample_actions(sample_rng_or_pytorch_device, observation, **self.sample_kwargs),
        }

        model_time = time.monotonic() - start_time

        print("processing outputs")

        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self.output_transforms(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }

        outputs["actions"] = np.asarray(outputs["actions"][:, :8])


        # result = policy.infer(example)['actions']

        # print(f"shuijie debug: policy.infer(example) {outputs}")
        return outputs
    
class pytorch_model(pi05_model):
    def __init__(self, weight_path = "./model_weights/pi05_droid_pytorch_model.safetensors"):
        super().__init__(
            config = "pi05_droid",
            checkpoint_dir="./model_weights/pi05_droid"
            )
        self.model = pi0_pytorch.PI0Pytorch(config=self.config.model)
        self.model.load_state_dict(safetensors.torch.load_file(weight_path))
        self.model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        self.pytorch_device='cuda:0'
        self.model = self.model.to(self.pytorch_device)
        self.model.eval()

        



    def get_pi05_action(self, obs):
        with torch.no_grad():
            inputs = jax.tree.map(lambda x: x, obs)
            inputs = self.input_transform(inputs)
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self.pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self.pytorch_device

            self.sample_kwargs = dict(self.sample_kwargs)


            observation = _model.Observation.from_dict(inputs)
            start_time = time.monotonic()
            outputs = {
                "state": inputs["state"],
                "actions": self.model.sample_actions(sample_rng_or_pytorch_device, observation, **self.sample_kwargs),
            }
            model_time = time.monotonic() - start_time

            print("processing outputs")

            # outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
            # Convert back to numpy (move to CPU first)
            outputs = jax.tree.map(
                lambda x: np.asarray(x[0, ...].detach().cpu() if isinstance(x, torch.Tensor) else x), 
                outputs
            )
            outputs = self.output_transforms(outputs)
            outputs["policy_timing"] = {
                "infer_ms": model_time * 1000,
            }

            return outputs


"""



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



print("model loaded")

print("processing inputs")

# Run inference on a dummy example.
obs = droid_policy.make_droid_example()

# Make a copy since transformations may modify the inputs in place.




 # Make a batch and convert to jax.Array.


 # Prepare kwargs for sample_actions



"""