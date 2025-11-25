from openpi.training import config as _config
from openpi.shared import download
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
import safetensors
import numpy as np
import jax.numpy as jnp

from abc import ABC, abstractmethod
from .pi0_pytorch import PI0Pytorch
from combine_vggt_pi05 import PI0_vggt_pytorch

from pointCloud_utils import run_vggt_model, load_vggt_model, filter_points_by_confidence, SimplePointCloudEmbedder

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
    def __init__(self, config = "pi05_droid",
            checkpoint_dir="./model_weights/pi05_droid_jax"):
        super().__init__(
            config,
            checkpoint_dir
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
    def __init__(self, checkpoint_dir, config, weights_filename = "model.safetensors", pc_conf_thres=0.9):
        super().__init__(
            config,
            checkpoint_dir
            )
        self.model = PI0_vggt_pytorch(config=self.config.model)
        # self.model.load_state_dict(safetensors.torch.load_file(weight_path))
        # safetensors.torch.load_model(self.model, os.path.join(checkpoint_dir, weights_filename))

        state_dict = safetensors.torch.load_file(os.path.join(checkpoint_dir, weights_filename))
        
        # Load with strict=False
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        # safetensors.torch.load_model(self, checkpoint_path)
        
        # print(f"Loaded pretrained weights. Missing keys: {len(missing_keys)}")

        if len(missing_keys) > 0:
            for name in missing_keys:
                print(name)
        
        self.model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        self.pytorch_device='cuda:0'
        self.model = self.model.to(self.pytorch_device)
        self.model.eval()


        self.pc_generator = load_vggt_model().to(self.pytorch_device)
        self.pc_conf_thres = pc_conf_thres

    def get_pi05_action(self, obs):
        with torch.no_grad():
            if isinstance(obs['observation/image'], np.ndarray) and isinstance(obs['observation/wrist_image'], np.ndarray):
                img_input_list = [obs['observation/image'], obs['observation/wrist_image']]
            else:
                img_input_list = [obs['observation/image'].numpy(), obs['observation/wrist_image'].numpy()]


            predictions = run_vggt_model(self.pc_generator, img_input_list)

            filtered_pointCloud_vertices, filtered_pointCloud_rgb =  filter_points_by_confidence(predictions, conf_thres=self.pc_conf_thres)

            pc_tensor = torch.from_numpy(filtered_pointCloud_vertices).to(self.pytorch_device)

            inputs = jax.tree.map(lambda x: x, obs)
            inputs = self.input_transform(inputs)
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self.pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self.pytorch_device

            self.sample_kwargs = dict(self.sample_kwargs)


            observation = _model.Observation.from_dict(inputs)
            start_time = time.monotonic()
            outputs = {
                "state": inputs["state"],
                "actions": self.model.sample_actions(sample_rng_or_pytorch_device, observation, pc_tensor,  **self.sample_kwargs),
            }
            model_time = time.monotonic() - start_time

            # print("processing outputs")

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

