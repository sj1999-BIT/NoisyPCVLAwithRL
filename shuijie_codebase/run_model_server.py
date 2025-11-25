"""
Flask-based REST API server for PI0.5 model inference.

Usage:
    python pi05_inference_server_flask.py

Example request:
    curl -X POST http://localhost:5000/predict \
      -H "Content-Type: application/json" \
      -d @sample_input.json
"""

from flask import Flask, request, jsonify
import torch
import numpy as np
import base64
import io
import time

from tqdm import tqdm
from PIL import Image
from models_pytorch.pi05_models import pytorch_model
from combine_vggt_pi05 import PI0_vggt_pytorch

from openpi.policies.libero_policy import make_libero_example

app = Flask(__name__)

# Initialize model globally
print("Loading PI0.5 model...")
# model = pytorch_model()
# model = pytorch_model(config="pi05_libero", checkpoint_dir="./models_pytorch/model_weights/pi05_vggt_libero_torch")

model = pytorch_model(config="pi05_libero", checkpoint_dir="./models_pytorch/model_weights/pi05_libero_torch")
# model = PI0_vggt_pytorch(config="pi05_libero", checkpoint_dir="./models_pytorch/model_weights/pi05_libero_torch")

print("Model loaded successfully!")


def decode_image(data):
    """Decode base64 encoded image to numpy array."""
    if isinstance(data, str):
        image_data = base64.b64decode(data)
        image = Image.open(io.BytesIO(image_data))
    return np.array(image)


def prepare_libero_input(data):
    """
    Prepare Libero format input for the model.
    
    Expected input format:
    {
        "observation/image": base64 or numpy array,
        "observation/wrist_image": base64 or numpy array,
        "observation/state": [N floats],  # Combined joint + gripper
        "prompt": "text instruction"
    }
    """
    # Decode images
    exterior_image = decode_image(data['observation/image'])
    wrist_image = decode_image(data['observation/wrist_image'])
    
    # Parse state
    state = data['observation/state']
    if isinstance(state, (list, tuple)):
        state = np.array(state, dtype=np.float32)
    elif isinstance(state, torch.Tensor):
        state = state.numpy()
    
    # Split state into joint_position and gripper_position
    # Adjust indices based on your robot configuration
    # Example: Franka Panda has 7 joints + gripper
    # num_joints = 7  # Adjust this for your robot
    # joint_position = state[:num_joints]
    # gripper_position = state[num_joints:]
    
    # Ensure gripper_position is at least 1D
    # if gripper_position.ndim == 0:
    #     gripper_position = gripper_position[np.newaxis]
    
    # Handle prompt
    prompt = data.get('prompt', 'do nothing')
    if isinstance(prompt, bytes):
        prompt = prompt.decode("utf-8")
    
    # Return in internal DROID format for model
    model_input = {
        'observation/image': torch.from_numpy(exterior_image),
        'observation/wrist_image': torch.from_numpy(wrist_image),
        'observation/state': torch.from_numpy(state),
        'prompt': prompt
    }
    
    return model_input


def prepare_droid_input(data):
    """
    Prepare DROID format input for the model.
    
    Expected input format:
    {
        "observation/exterior_image_1_left": base64 or numpy array,
        "observation/wrist_image_left": base64 or numpy array,
        "observation/joint_position": [7 floats],
        "observation/gripper_position": [1 float],
        "prompt": "text instruction"
    }
    """
    # Decode images
    exterior_image = decode_image(data['observation/exterior_image_1_left'])
    wrist_image = decode_image(data['observation/wrist_image_left'])
    
    # Parse arrays
    joint_position = data['observation/joint_position']
    if isinstance(joint_position, (list, tuple)):
        joint_position = np.array(joint_position, dtype=np.float32)
    elif isinstance(joint_position, torch.Tensor):
        joint_position = joint_position.numpy()
        
    gripper_position = data['observation/gripper_position']
    if isinstance(gripper_position, (list, tuple, float, int)):
        gripper_position = np.array(
            [gripper_position] if np.isscalar(gripper_position) else gripper_position, 
            dtype=np.float32
        )
    elif isinstance(gripper_position, torch.Tensor):
        gripper_position = gripper_position.numpy()
    
    # Handle prompt
    prompt = data.get('prompt', 'do nothing')
    if isinstance(prompt, str):
        prompt = prompt.encode('utf-8')
    elif isinstance(prompt, bytes):
        prompt = str(prompt).encode('utf-8')
    
    model_input = {
        'observation/exterior_image_1_left': torch.from_numpy(exterior_image),
        'observation/wrist_image_left': torch.from_numpy(wrist_image),
        'observation/joint_position': torch.tensor(joint_position, dtype=torch.float32),
        'observation/gripper_position': torch.tensor(gripper_position, dtype=torch.float32),
        'prompt': prompt
    }
    
    return model_input


def prepare_input(data):
    """
    Auto-detect format and prepare input for the model.
    Supports both Libero and DROID formats.
    """
    if 'observation/image' in data and 'observation/state' in data:
        return prepare_libero_input(data)
    elif 'observation/exterior_image_1_left' in data and 'observation/joint_position' in data:
        return prepare_droid_input(data)
    else:
        raise ValueError(
            "Unrecognized input format. Expected either:\n"
            "  - Libero: 'observation/image' and 'observation/state'\n"
            "  - DROID: 'observation/exterior_image_1_left' and 'observation/joint_position'"
        )


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model": "pi05_pytorch"}), 200

@app.route('/perform', methods=['POST'])
def run_performance():
    try:
        # Get JSON data from request
        data = request.get_json()
        # print("received inputs", data)
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        time_step_num = data["timesteps"]
        
        # Track performance metrics
        start_time = time.time()

        for _ in tqdm(range(time_step_num), desc="run model performance"):
            # step_start = time.time()
            
            example = make_libero_example()
            # Get model prediction
            with torch.no_grad():
                model.get_pi05_action(example)
            
            # step_end = time.time()
            # step_duration = step_end - step_start
            # step_hz = 1.0 / step_duration if step_duration > 0 else 0
            
            # Optional: print per-step Hz
            # print(f"Step {time_step}: {step_hz:.2f} Hz")

        end_time = time.time()
        total_duration = end_time - start_time
        average_hz = time_step_num / total_duration if total_duration > 0 else 0

        response = {
            "status": "success",
            "performance": {
                "total_duration_seconds": round(total_duration, 3),
                "average_hz": round(average_hz, 2),
                "timesteps": time_step_num
            }
        }
        
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    
    Request body should contain:
    {
        "'observation/exterior_image_1_left": base64 string or array,
        "observation/wrist_image_left": base64 string or array,
        "observation/joint_position": [7 floats],
        "observation/gripper_position": [1 float],
        "prompt": "instruction text"
    }
    
    Returns:
    {
        "actions": [...],
        "status": "success"
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        # print("received inputs", data)
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # # Validate required fields
        # required_fields = ['observation/exterior_image_1_left', 'observation/wrist_image_left', 'observation/joint_position', 
        #                   'observation/gripper_position', 'prompt']
        # missing_fields = [field for field in required_fields if field not in data]
        
        # if missing_fields:
        #     return jsonify({
        #         "error": f"Missing required fields: {missing_fields}"
        #     }), 400
        
        

        # Prepare input for model
        model_input = prepare_input(data)

        # print("passing into model", model_input)
        
        # Get model prediction
        with torch.no_grad():
            output = model.get_pi05_action(model_input)

        # print("get model output", output.keys())
        
        # Convert output to JSON-serializable format
        actions = output["actions"]
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy().tolist()
        elif isinstance(actions, np.ndarray):
            actions = actions.tolist()
        
        response = {
            "status": "success",
            "actions": actions
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


if __name__ == '__main__':
    # Run server
    # For production, use a proper WSGI server like gunicorn:
    # gunicorn -w 4 -b 0.0.0.0:5000 pi05_inference_server_flask:app
    app.run(host='0.0.0.0', port=5000, debug=False)
