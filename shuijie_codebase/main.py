
import os 
print(os.getcwd())

import torch
from combine_vggt_pi05 import PI0_vggt_pytorch
from models_pytorch.pi05_models import pytorch_model

# Count total parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print("=" * 80)
print("loading new model")
print("=" * 80)
weight_path = "./combined_vggt_with_pi05.safetensors"
new_model = PI0_vggt_pytorch(config="pi05_libero", checkpoint_path=weight_path)
new_model_state_dict = new_model.state_dict()
new_model_params = set(new_model_state_dict.keys())
new_total = count_parameters(new_model)


# new_model.save_model(save_path=weight_path)


# new_model.load_model(weight_path)






print("=" * 80)
print("loading old model")
print("=" * 80)
original_model = pytorch_model(config="pi05_libero", checkpoint_dir="./models_pytorch/model_weights/pi05_libero_torch").model
old_model_params = set(original_model.state_dict().keys())
old_total = count_parameters(original_model)
del(original_model)





# # Find new parameters (in new model but not in old checkpoint)
# new_params = new_model_params - old_model_params

# # Find removed parameters (in old checkpoint but not in new model)
# removed_params = old_model_params - new_model_params

# # Find common parameters
# common_params = new_model_params & old_model_params

# print("=" * 80)
# print("NEW PARAMETERS (Point Cloud & VGGT components):")
# print("=" * 80)
# for param in sorted(new_params):
#     print(f"  {param}: {new_model_state_dict[param].shape}")

# print("\n" + "=" * 80)
# print("REMOVED PARAMETERS:")
# print("=" * 80)
# for param in sorted(removed_params):
#     print(f"  {param}")

# print("\n" + "=" * 80)
# print("COMMON PARAMETERS (will be loaded from checkpoint):")
# print("=" * 80)
# print(f"Total: {len(common_params)} parameters")



# print("\n" + "=" * 80)
# print("PARAMETER STATISTICS:")
# print("=" * 80)
# print(f"Old model total parameters: {old_total:,}")
# print(f"New model total parameters: {new_total:,}")
# print(f"Additional parameters: {new_total - old_total:,}")
# print(f"Percentage increase: {((new_total - old_total) / old_total * 100):.2f}%")
