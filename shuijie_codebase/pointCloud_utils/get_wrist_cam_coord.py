"""
We just want to get the camera views
"""

from PC_utils import *
from vggt.utils.geometry import depth_to_cam_coords_points, depth_to_world_coords_points, unproject_depth_map_to_point_map, project_world_points_to_camera_points_batch, closed_form_inverse_se3

if __name__ == "__main__":

    print("Initializing and loading VGGT model...")
    # model = VGGT.from_pretrained("facebook/VGGT-1B")  # another way to load the model



    vggt_model = load_vggt_model()

    # in actual pipeline this will just be an array of numpy images from the sensor
    example_img_dir = "./example_input_imgs"
    input_images_list = [cv2.imread(os.path.join(example_img_dir, img_filename))
                         for img_filename in os.listdir(example_img_dir)]

    # get predictions
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Move model to device
    model = vggt_model.to(device)
    model.eval()

    # format input images into model input
    model_input = load_and_preprocess_images(input_images_list).to(device)

    # Run inference
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(model_input)



    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic_tensor, intrinsic_tensor = pose_encoding_to_extri_intri(predictions["pose_enc"], model_input.shape[-2:])
    predictions["extrinsic"] = extrinsic_tensor
    predictions["intrinsic"] = intrinsic_tensor

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension

    # print("model predictions", predictions.keys())

    print("Computing cam points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    # print("depth map shape", depth_map.shape)


    cam_coords_points_list = []
    for frame_idx in range(depth_map.shape[0]):
        cur_world_points = depth_to_cam_coords_points(
            depth_map[frame_idx].squeeze(-1), predictions["intrinsic"][frame_idx]
        )
        cam_coords_points_list.append(cur_world_points)
    cam_coords_points = np.stack(cam_coords_points_list, axis=0)


    # cam_coords_points = depth_to_cam_coords_points(depth_map_tensor.cpu().numpy(), intrinsic_tensor.cpu().numpy())

    # select one view
    view_index = 1

    images = predictions["images"][:, [2, 1, 0], :, :]

    print("images.shape", images.shape)



    # Handle different image formats - check if images need transposing
    if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
        colors_bgr = np.transpose(images, (0, 2, 3, 1))
    else:  # Assume already in NHWC format
        colors_bgr = images 
    colors_bgr = colors_bgr

    

    print("colors_bgr.shape", colors_bgr.shape)

    colors_bgr = (colors_bgr[view_index].reshape(-1, 3) * 255).astype(np.uint8)
    colors_rgb = colors_bgr

    print("colors_rgb.shape", colors_rgb.shape)
        




    conf = predictions.get("depth_conf", np.ones_like(cam_coords_points[..., 0]))[view_index].reshape(-1)

    conf_thres = 0.7

    # Convert inverse confidence to normal confidence
    # confidence_normal = 1.0 / np.maximum(conf, 1e-6)  # Avoid division by zero
    confidence_normal = conf

    # Now apply threshold (higher conf_thres = stricter filtering)
    if conf_thres == 0.0:
        # Keep all points
        conf_mask = np.ones(len(conf), dtype=bool)
    else:
        # Keep points with confidence >= threshold
        conf_mask = (confidence_normal >= conf_thres) & (conf > 1e-5)



    vertices_3d = cam_coords_points[view_index].reshape(-1, 3)


    


    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]

    glbscene = convert_to_glb(vertices_3d, colors_rgb)

    # print(f"predictions: {predictions}")

    glbfilepath = "./example_glb/example.glb"
    glbscene.export(file_obj=glbfilepath)


    # Add debugging:
    print("\n=== Camera Coordinate Statistics ===")
    print(f"Number of points: {vertices_3d.shape[0]}")
    print(f"X range: [{vertices_3d[:, 0].min():.3f}, {vertices_3d[:, 0].max():.3f}]")
    print(f"Y range: [{vertices_3d[:, 1].min():.3f}, {vertices_3d[:, 1].max():.3f}]")
    print(f"Z range: [{vertices_3d[:, 2].min():.3f}, {vertices_3d[:, 2].max():.3f}]")
    print(f"Mean position: [{vertices_3d[:, 0].mean():.3f}, {vertices_3d[:, 1].mean():.3f}, {vertices_3d[:, 2].mean():.3f}]")

    # After filtering:
    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]

    print(f"\nAfter confidence filtering ({conf_thres}):")
    print(f"Remaining points: {vertices_3d.shape[0]}")
    print(f"Z range: [{vertices_3d[:, 2].min():.3f}, {vertices_3d[:, 2].max():.3f}]")



    # predictions["extrinsic"] = extrinsic
    # predictions["intrinsic"] = intrinsic

    # world_pc = unproject_depth_map_to_point_map(depth_map, extrinsic_tensor , intrinsic_tensor)

    





    # # Convert tensors to numpy
    # for key in predictions.keys():
    #     if isinstance(predictions[key], torch.Tensor):
    #         predictions[key] = predictions[key].squeeze(0).cpu().numpy()  # remove batch dimension

    # print("Computing cam points from depth map...")
    # depth_map = predictions["depth"]  # (S, H, W, 1)
    # world_pc = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"] , predictions["intrinsic"])

    # extrinsic_batch_tensor = torch.from_numpy(predictions["extrinsic"][1]).unsqueeze(0).float().to(device)

    # all_cam_points = [] 


    # print("extrinsic_batch_tensor", extrinsic_batch_tensor.shape)
    # # extrinsic_batch_tensor = closed_form_inverse_se3(extrinsic_batch_tensor)[:,:3,:]

    # rotation_180_y_4x4 = torch.tensor([
    #     [[-1,  0,  0,  0],
    #     [ 0,  1,  0,  0],
    #     [ 0,  0, -1,  0],
    #     [ 0,  0,  0,  1]]
    # ], device=device, dtype=torch.float32)  # (1, 4, 4)

    # # extrinsic_batch_tensor = extrinsic_batch_tensor.squeeze(0)
    # # Add bottom row [0, 0, 0, 1] to make 4x4
    # bottom_row = torch.tensor([[[0, 0, 0, 1]]], device=device, dtype=torch.float32)  # (1, 1, 4)
    # bottom_row = bottom_row.expand(extrinsic_batch_tensor.shape[0], -1, -1)  # (2, 1, 4)

    # extrinsic_batch_tensor = torch.cat(
    #     [extrinsic_batch_tensor, bottom_row], 
    #     dim=1  # Concatenate along the rows dimension
    # )[:, :3, :]  # (2, 3, 4) + (2, 1, 4) = (2, 4, 4)
        
    # print("extrinsic_batch_tensor", extrinsic_batch_tensor.shape)

    # # Apply rotation: rotate_180 @ wrist_inverted
    # # extrinsic_batch_tensor = torch.bmm(rotation_180_y_4x4, extrinsic_batch_tensor)[:, :3, :]

    # print("extrinsic_batch_tensor", extrinsic_batch_tensor.shape)

    # for i, cur_world_frame_pc in enumerate(world_pc):
    #     world_pc_tensor = torch.from_numpy(cur_world_frame_pc).float().to(device)

    #     print("world_pc_tensor", world_pc_tensor.shape)
    #     print("torch.ones_like(world_pc_tensor[..., 0:1])", torch.ones_like(world_pc_tensor[..., 0:1]).shape)
    #     world_points_homogeneous = torch.cat(
    #         [world_pc_tensor, torch.ones_like(world_pc_tensor[..., 0:1])], dim=-1    
    #     ).unsqueeze(0)  # 1xNx4

       



    #     # Flatten spatial dimensions: (1, 294, 518, 4) → (1, 294*518, 4)
    #     B, H, W, C = world_points_homogeneous.shape
    #     world_points_flat = world_points_homogeneous.reshape(B, H * W, C)  # (1, 152532, 4)

    #     # Now transpose for bmm: (1, 152532, 4) → (1, 4, 152532)
    #     world_points_transposed = world_points_flat.transpose(-1, -2)

    #     # Apply extrinsics: (B, 3, 4) @ (B, 4, N) = (B, 3, N)
    #     cam_points = torch.bmm(
    #         extrinsic_batch_tensor,  # (B, 3, 4)
    #         world_points_transposed  # (B, 4, 152532)
    #     )  # Result: (B, 3, 152532)

    #     # Reshape back to spatial: (B, 3, H*W) → (B, H, W, 3)
    #     cam_points = cam_points.reshape(B, 3, H, W).permute(0, 2, 3, 1) 

    #         # Remove batch dimension and collect
    #     all_cam_points.append(cam_points.squeeze(0))  # (H, W, 3)

    # # Stack all frames: list of (H, W, 3) → (num_frames, H, W, 3)
    # cam_pc = torch.stack(all_cam_points, dim=0).cpu().numpy()




    # # # change both camera to wrist camera
    # # world_pc_tensor = torch.from_numpy(world_pc).float().to(device)
    # # extrinsic_batch_tensor = torch.from_numpy(predictions["extrinsic"]).float().to(device)

    # # B = extrinsic_batch_tensor.shape[0]  # Batch size, i.e., number of cameras

    # # # print("extrinsics", extrinsic_batch_tensor.shape)
    # # # print("world_pc", world_pc_batch_tensor.shape)

    # # # ext_4x4 = torch.from_numpy().unsqueeze(0)

    # # # new_ext = closed_form_inverse_se3(ext_4x4)[:3, :]

    # # # print("inverse", new_ext.shape)

    # # # inverse_ext_batch_npy = np.zeros((2,4,4))

    # # # inverse_ext_batch_npy[0] = np.vstack([extrinsic_batch_tensor[0][1].cpu().numpy(), [0, 0, 0, 1]])
    # # # inverse_ext_batch_npy[1] = inverse_ext_batch_npy[0]

    


    # # # extrinsic_batch_tensor = torch.from_numpy(closed_form_inverse_se3(inverse_ext_batch_npy)[:, :3, :]).unsqueeze(0).float().to(device)

    # # # print("extrinsic_batch_tensor",extrinsic_batch_tensor.shape)

    
    # # # cam_pc = project_world_points_to_camera_points_batch(world_pc_batch_tensor, extrinsic_batch_tensor).squeeze(0).cpu().numpy()
    # # world_points_homogeneous = torch.cat(
    # #         [world_pc_tensor, torch.ones_like(world_pc_tensor[..., 0:1])], dim=1
    # #     )  # Nx4
    # # # Reshape for batch processing
    # # world_points_homogeneous = world_points_homogeneous.unsqueeze(0).expand(
    # #     B, -1, -1
    # # )  # BxNx4

    # # # Step 1: Apply extrinsic parameters
    # # # Transform 3D points to camera coordinate system for all cameras
    # # cam_points = torch.bmm(
    # #     extrinsic_batch_tensor, world_points_homogeneous.transpose(-1, -2)
    # # )

    # # print("cam_pc", cam_pc.shape)

    
    # vertices_3d = cam_pc.reshape(-1, 3)

    # # print("vertices_3d", vertices_3d.shape)

    # images = predictions["images"]

    # # Handle different image formats - check if images need transposing
    # if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
    #     colors_bgr = np.transpose(images, (0, 2, 3, 1))
    # else:  # Assume already in NHWC format
    #     colors_bgr = images 
    # colors_bgr = (colors_bgr.reshape(-1, 3) * 255).astype(np.uint8)
    # colors_rgb = colors_bgr


    # conf = predictions.get("depth_conf", np.ones_like(world_pc[..., 0])).reshape(-1)

    # conf_thres = 0.7

    # # Convert inverse confidence to normal confidence
    # # confidence_normal = 1.0 / np.maximum(conf, 1e-6)  # Avoid division by zero
    # confidence_normal = conf

    # # Now apply threshold (higher conf_thres = stricter filtering)
    # if conf_thres == 0.0:
    #     # Keep all points
    #     conf_mask = np.ones(len(conf), dtype=bool)
    # else:
    #     # Keep points with confidence >= threshold
    #     conf_mask = (confidence_normal >= conf_thres) & (conf > 1e-5)

    # vertices_3d = vertices_3d[conf_mask]
    # colors_rgb = colors_rgb[conf_mask]

    # glbscene = convert_to_glb(vertices_3d, colors_rgb)

    # # print(f"predictions: {predictions}")

    # glbfilepath = "./example_glb/example.glb"
    # glbscene.export(file_obj=glbfilepath)