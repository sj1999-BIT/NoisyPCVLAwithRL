"""
PI0.5 Model Inference Client

A client class for sending observations to the PI0.5 inference server
and receiving predicted actions.

Usage:
    from pi05_client import PI05Client
    
    client = PI05Client(server_url="http://localhost:5000")
    
    # Prepare observation
    observation = {
        'exterior_image': exterior_image_array,  # numpy array (H, W, 3)
        'wrist_image': wrist_image_array,        # numpy array (H, W, 3)
        'joint_position': joint_pos_array,       # numpy array (7,)
        'gripper_position': gripper_pos_array,   # numpy array (1,)
    }
    
    # Get action
    actions = client.get_action(observation, prompt="pick up the red block")
    print("Predicted actions:", actions)
"""
import os
import sys

import requests
import numpy as np
import base64
import io
from PIL import Image
from typing import Dict, List, Union, Optional
import time
import torch
import imageio

from tqdm import tqdm


class PI05Client:
    """Client for PI0.5 model inference server."""
    
    def __init__(self, server_url: str = "http://localhost:5000", timeout: int = 30):
        """
        Initialize the PI0.5 client.
        
        Args:
            server_url: URL of the inference server
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        # self._check_server_health()
    
    def _check_server_health(self):
        """Check if server is healthy and accessible."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ Connected to server at {self.server_url}")
                print(f"   Server status: {response.json()}")
            else:
                print(f"⚠️  Server responded but may not be healthy: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to server at {self.server_url}")
            print(f"   Please start the server first!")
            raise
        except Exception as e:
            print(f"⚠️  Error checking server health: {e}")



    
    @staticmethod
    def encode_image(image_array: np.ndarray) -> str:
        """
        Encode numpy image array to base64 string.
        
        Args:
            image_array: numpy array of shape (H, W, 3) with dtype uint8
            
        Returns:
            Base64 encoded string
        """
        if isinstance(image_array, torch.Tensor):
            image_array = image_array.numpy()

        # Ensure uint8
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        
        image = Image.fromarray(image_array)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    def prepare_request_droid(
        self,
        observation: Dict[str, np.ndarray],
        prompt: str
    ) -> Dict:
        """
        Prepare observation data for API request.
        
        Args:
            observation: Dictionary containing:
                - 'exterior_image': numpy array (H, W, 3)
                - 'wrist_image': numpy array (H, W, 3)
                - 'joint_position': numpy array (7,)
                - 'gripper_position': numpy array (1,)
            prompt: Language instruction string
            
        Returns:
            Dictionary formatted for API request
        """
        # Encode images
        exterior_image_b64 = self.encode_image(observation['observation/exterior_image_1_left'])
        wrist_image_b64 = self.encode_image(observation['observation/wrist_image_left'])
        
        # Convert positions to lists
        joint_position = observation['observation/joint_position'].tolist() if isinstance(
            observation['observation/joint_position'], np.ndarray
        ) else observation['observation/joint_position']
        
        gripper_position = observation['observation/gripper_position'].tolist() if isinstance(
            observation['observation/gripper_position'], np.ndarray
        ) else observation['observation/gripper_position']
        
        request_data = {
            "observation/exterior_image_1_left": exterior_image_b64,
            "observation/wrist_image_left": wrist_image_b64,
            "observation/joint_position": joint_position,
            "observation/gripper_position": gripper_position,
            "prompt": prompt
        }
        
        return request_data
    
    def prepare_request_libero(
        self,
        observation: Dict[str, np.ndarray],
        prompt: str
    ) -> Dict:
        """
        Prepare observation data for API request.
        
        Args:
            observation: Dictionary containing:
            "observation/image": base64 or numpy array,
            "observation/wrist_image": base64 or numpy array,
            "observation/state": [N floats],  # Combined joint + gripper
            "prompt": "text instruction"
            prompt: Language instruction string
            
        Returns:
            Dictionary formatted for API request
        """
        # print("shuijie debug, model type at client prepare request: ", model_type)

        # Encode images
        exterior_image_b64 = self.encode_image(observation['observation/image'])
        wrist_image_b64 = self.encode_image(observation['observation/wrist_image'])

        state = observation['observation/state']
        if isinstance(state, torch.Tensor):
            state = state.numpy().tolist()
        else:
            state = state.tolist()

        # num_joints = 7  # Adjust this for your robot
        # joint_position = state[:num_joints].tolist()
        # gripper_position = state[num_joints:].tolist()
    
        
        # Convert positions to lists
        # joint_position = observation['observation/joint_position'].tolist() if isinstance(
        #     observation['observation/joint_position'], np.ndarray
        # ) else observation['observation/joint_position']
        
        # gripper_position = observation['observation/gripper_position'].tolist() if isinstance(
        #     observation['observation/gripper_position'], np.ndarray
        # ) else observation['observation/gripper_position']
        
        request_data = {
            "observation/image": exterior_image_b64,
            "observation/wrist_image": wrist_image_b64,
            "observation/state": state,
            "prompt": prompt
        }
        
        return request_data
    
    def prepare_request(self, observation, model_type, prompt):

        
        if model_type == "droid":
            return self.prepare_request_droid(observation, prompt)
        elif model_type == "libero":
            return self.prepare_request_libero(observation, prompt)
        else:
            raise Exception

    
    def get_action(
        self,
        observation: Dict[str, np.ndarray],
        model_type: str,
        prompt: str,
        return_full_response: bool = False
    ) -> Union[np.ndarray, Dict]:
        """
        Get predicted action from the model.
        
        Args:
            observation: Dictionary containing robot observations
            prompt: Language instruction
            return_full_response: If True, return full server response
            
        Returns:
            Predicted actions as numpy array, or full response dict if requested
        """
        # print("sending message", observation)

        request_data = self.prepare_request(observation, model_type, prompt)

        try:
            response = requests.post(
                f"{self.server_url}/predict",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if return_full_response:
                    return result
                else:
                    return np.array(result['actions'])
            else:
                raise Exception(f"Server error: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            raise Exception(f"Request timeout after {self.timeout} seconds")
        except requests.exceptions.ConnectionError:
            raise Exception(f"Cannot connect to server at {self.server_url}")
        except Exception as e:
            raise Exception(f"Error getting action: {str(e)}")
    
    def check_model_performance(self, time_step=100, time_out=100):
        """Check if server model performance."""
        try:
            response = requests.post(
                f"{self.server_url}/perform",
                json={"timesteps": time_step},
                headers={"Content-Type": "application/json"},
                timeout=time_out
            )
            
            if response.status_code == 200:
                print(f"✅ Server status: {response.json()}")
            else:
                print(f"⚠️  Server responded but may not be healthy: {response.json()} {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to server at {self.server_url}")
            print(f"   Please start the server first!")
            raise
        except Exception as e:
            print(f"⚠️  Error checking server health: {e}")

class PI05ClientWithRetry(PI05Client):
    """PI0.5 client with automatic retry logic."""
    
    def __init__(
        self,
        server_url: str = "http://localhost:5000",
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize client with retry capability.
        
        Args:
            server_url: URL of the inference server
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        super().__init__(server_url, timeout)
    
    def get_action(
        self,
        observation: Dict[str, np.ndarray],
        model_type: str,
        prompt: str,
        return_full_response: bool = False
    ) -> Union[np.ndarray, Dict]:
        """Get action with automatic retry on failure."""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return super().get_action(observation, model_type, prompt, return_full_response)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    print(f"⚠️  Attempt {attempt + 1} failed: {e}")
                    print(f"   Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    print(f"❌ All {self.max_retries} attempts failed")
        
        raise last_exception

def run_droid_trajectory(client):

    import tensorflow_datasets as tfds
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

            observation = {
                'observation/exterior_image_1_left': image.numpy(),
                'observation/wrist_image_left': wrist_image.numpy(),
                'observation/joint_position': joint_pos.numpy(),
                'observation/gripper_position': gripper_pos.numpy(), 
                'prompt': instruction.numpy()
            }

            prompt = observation["prompt"]

            actions = client.get_action(observation, prompt)
            print(f"   Action shape: {actions.shape}")


def generate_pi05_droid_inputs(obs, env, current_prompt="do nothing"):
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

def generate_pi05_libero_inputs(obs, env, current_prompt="do nothing"):
    """
    Generate inputs in Libero format for π₀/π₀.5 models.
    
    Args:
        obs: Observation dict from Libero environment
        env: Libero environment instance
        current_prompt: Language instruction (string)
    
    Returns:
        dict: Inputs in Libero format compatible with LiberoInputs transform
    """
    # Extract joint positions and gripper state
    joint_position_npy = obs["robot0_joint_pos"]
    gripper_position_npy = obs["robot0_gripper_qpos"]
    
    # Concatenate into state vector (similar to how LiberoInputs expects it)
    # Note: Libero stores this as "observation/state" which includes both joints and gripper
    state = np.concatenate([joint_position_npy, gripper_position_npy])
    
    # Render images from cameras
    video_img_main_left = env.sim.render(
        height=512, width=768, camera_name="robot0_agentview_center"
    )[::-1]  # Flip vertically
    
    video_img_wrist = env.sim.render(
        height=512, width=768, camera_name="robot0_eye_in_hand"
    )[::-1]  # Flip vertically
    
    # Return in Libero format (keys match what LiberoInputs expects)
    return {
        'observation/image': video_img_main_left,           # Base/third-person camera
        'observation/wrist_image': video_img_wrist,        # Wrist camera
        'observation/state': state,                         # Combined joint + gripper state
        'prompt': current_prompt                            # Language instruction
    }


def generate_pi05_inputs(obs, env, model_type, current_prompt):
    if model_type == "droid":
        observation = generate_pi05_droid_inputs(obs, env, current_prompt)
    elif model_type == "libero":
        observation = generate_pi05_libero_inputs(obs, env, current_prompt)
    else:
        raise Exception
    
    return observation


def run_robocasa_trajectory(client, model_type, current_prompt="pick up vegetables and place in the plate", video_path="./tmp/demo.mp4"):

    import robosuite
    from robosuite.controllers import load_composite_controller_config
    from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS

    def create_env(
        env_name,
        # robosuite-related configs
        robots="PandaOmron",
        camera_names=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        camera_widths=128,
        camera_heights=128,
        seed=None,
        render_onscreen=False,
        # robocasa-related configs
        obj_instance_split=None,
        generative_textures=None,
        randomize_cameras=False,
        layout_and_style_ids=None,
        layout_ids=None,
        style_ids=None,
    ):
        controller_config =  load_composite_controller_config(
            controller=None,
            robot="PandaOmron"
        )

        env_kwargs = dict(
            env_name=env_name,
            robots=robots,
            controller_configs=controller_config,
            camera_names=camera_names,
            camera_widths=camera_widths,
            camera_heights=camera_heights,
            has_renderer=render_onscreen,
            has_offscreen_renderer=(not render_onscreen),
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=(not render_onscreen),
            camera_depths=False,
            seed=seed,
            obj_instance_split=obj_instance_split,
            generative_textures=generative_textures,
            randomize_cameras=randomize_cameras,
            layout_and_style_ids=layout_and_style_ids,
            layout_ids=layout_ids,
            style_ids=style_ids,
            translucent_robot=False,
        )

        env = robosuite.make(**env_kwargs)
        return env

        # choose random task
    env_name = np.random.choice(list(ALL_KITCHEN_ENVIRONMENTS))
    env_name = "OrganizeVegetables"

    # print(list(ALL_KITCHEN_ENVIRONMENTS))
    # print("env_name ", env_name)

    env = create_env(
        env_name=env_name,
        render_onscreen=False,
        seed=0, # set seed=None to run unseeded
    )

    num_steps = 70
    num_rollouts = 1

    num_success_rollouts = 0

    for rollout_i in range(num_rollouts):
        obs = env.reset()

        video_writer = None
        video_writer = imageio.get_writer(video_path, fps=20)

        for step_i in tqdm(range(num_steps)):

            observation = generate_pi05_inputs(obs, env, model_type, current_prompt)
            
            if video_writer is not None:
                if model_type == "droid":
                    video_writer.append_data(np.concatenate((observation['observation/exterior_image_1_left'], observation['observation/wrist_image_left'])
                                                    , axis=1))
                elif model_type == "libero":
                    video_writer.append_data(np.concatenate((observation['observation/image'], observation['observation/wrist_image'])
                                                        , axis=1))


            pi05_model_outputs = client.get_action(observation, model_type, observation['prompt'])
            # print("action output shape", pi05_model_outputs.shape)
            # pi05_model_outputs = np.zeros((15, 8))

            for cur_output in pi05_model_outputs:
                # output is size 8, present 8 delta values to 7 joint positions and gripper positions
                # controller takes in size 12, with additional 

                # next_join_pos_np = cur_output[:7]
                # next_join_pos_np[0] = 1.54562641
                # next_gripper_pos = cur_output[7:]
                # print("current action: ", )
                
                action = np.concatenate(
                    (cur_output, np.zeros(5)),
                    axis=0
                )

                # print(f"current action {action}")

            # sample and execute random action
            # action = np.random.uniform(low=env.action_spec[0], high=env.action_spec[1])
                obs, _, _, _ = env.step(action)
                observation = generate_pi05_inputs(obs, env, model_type, current_prompt)


            if env._check_success():
                num_success_rollouts += 1
                break

        if video_writer is not None:
            video_writer.close()
            
            # print(f"Action shape: {actions.shape}")

def run_libero_trajectory(client, model_type, current_prompt, video_path="./tmp/demo.mp4"):

    sys.path.append('../LIBERO')

    # import os
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    os.environ['MUJOCO_GL'] = 'egl'
    
    # sys.path.append('../robosuite')

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    


    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = "libero_10" # can also choose libero_spatial, libero_object, etc.
    task_suite = benchmark_dict[task_suite_name]()

    # retrieve a specific task
    task_id = 0
    task = task_suite.get_task(task_id)
    task_name = task.name
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
        f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

    # step over the environment
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 512,
        "camera_widths": 768
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    env.reset()
    init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
    init_state_id = 0
    env.set_init_state(init_states[init_state_id])

    video_writer = None
    video_writer = imageio.get_writer(video_path, fps=20)
    
    formatted_observation = {}

    dummy_action = [0.] * 7
    obs, reward, done, info = env.step(dummy_action)

    for step in tqdm(range(100), desc="running libero dataset"):
        
        formatted_observation['observation/image'] = np.flip(np.flip(obs['agentview_image'], 0), 1)
        formatted_observation['observation/wrist_image'] = np.flip(np.flip(obs['robot0_eye_in_hand_image'], 0), 1)
        formatted_observation['observation/state'] = np.concatenate([obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]])
        formatted_observation['prompt'] = current_prompt

        pi05_model_outputs = client.get_action(formatted_observation, model_type, formatted_observation['prompt'])

        for action in pi05_model_outputs:
            obs, _, done, _ = env.step(action)

            if video_writer is not None:
                video_writer.append_data(np.concatenate((formatted_observation['observation/image'],
                                                        formatted_observation['observation/wrist_image'])
                                                , axis=1))
            if done:
                break
        
        if done:
            break

    if video_writer is not None:
        video_writer.close()
    env.close()

if __name__ == "__main__":
    client = PI05ClientWithRetry(max_retries=3)
    client.check_model_performance(time_step=1, time_out=500)
    run_libero_trajectory(client, model_type="libero", current_prompt="put both the alphabet soup and the tomato sauce in the basket")
