import os
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

# Import all your custom modules
from src.components.dataset import YCBVideoDataset
from src.components.model_trainer import PoseNet
from src.utils import draw_3d_bounding_box, class_id_to_name, get_add_s_score
from torchvision import transforms

def evaluate_model():
    print("--- Starting Final Model Evaluation ---")
    
    # 1. Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Define path to the validation data
    validation_folder_path = 'validation_data'
    
    # --- Create data_paths list from the validation folder ---
    val_paths = []
    frame_files = os.listdir(validation_folder_path)
    frame_ids = sorted(list(set([f.split('-')[0] for f in frame_files])))
    
    for frame_id in frame_ids:
        rgb_path = os.path.join(validation_folder_path, f"{frame_id}-color.png")
        meta_path = os.path.join(validation_folder_path, f"{frame_id}-meta.mat")
        
        # Only add if both color and meta file exist
        if os.path.exists(rgb_path) and os.path.exists(meta_path):
            val_paths.append({'rgb_path': rgb_path}) # Simplified for this script's needs
    print(f"Found {len(val_paths)} samples in the validation folder.")
    
    # 3. Define transforms and create DataLoader
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = YCBVideoDataset(data_paths=val_paths, transform=image_transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 4. Load the trained model
    model = PoseNet(pretrained=False)
    model.load_state_dict(torch.load('posenet_model.pth', map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    project_root = os.path.abspath(os.path.dirname(__file__))
    all_scores = []
    threshold = 0.02 # 2cm threshold

    print("--- Calculating ADD-S Metric on Validation Set ---")
    for i, data in enumerate(tqdm(val_loader)):
        image_tensor = data['image'].to(device).float()
        true_rotation_6d = data['rotation']
        true_translation = data['translation']
        
        with torch.no_grad():
            pred_translation, pred_rotation_6d = model(image_tensor)
    
        # Convert predicted 6D to 3x3 matrix and move to CPU
        b, _ = pred_rotation_6d.shape
        r1 = pred_rotation_6d[:, 0:3]; r2 = pred_rotation_6d[:, 3:6]
        r1 = torch.nn.functional.normalize(r1, dim=1)
        r2 = r2 - (r1 * r2).sum(dim=1, keepdim=True) * r1
        r2 = torch.nn.functional.normalize(r2, dim=1)
        r3 = torch.cross(r1, r2, dim=1)
        pred_rotation_matrix = torch.stack((r1, r2, r3), dim=2).cpu().numpy()[0]
        
        pred_translation = pred_translation.cpu().numpy()[0]
        
        pred_pose = np.eye(4); pred_pose[:3, :3] = pred_rotation_matrix; pred_pose[:3, 3] = pred_translation
        
        # Recreate the ground truth pose matrix
        true_rotation_matrix = true_rotation_6d.numpy()[0].reshape(3, 2)
        r1_true, r2_true = true_rotation_matrix[:, 0], true_rotation_matrix[:, 1]
        r3_true = np.cross(r1_true, r2_true)
        full_rot_true = np.stack((r1_true, r2_true, r3_true), axis=1)
        true_pose = np.eye(4); true_pose[:3, :3] = full_rot_true; true_pose[:3, 3] = true_translation.numpy()[0]

        # Get object model points
        class_id = val_loader.dataset.data_paths[i]['objects'][0]['class_id']
        model_name = class_id_to_name.get(class_id)
        obj_model_path = os.path.join(project_root, 'datasets', 'models', model_name, 'textured.obj')
        
        if not os.path.exists(obj_model_path):
            continue

        mesh = o3d.io.read_triangle_mesh(obj_model_path)
        model_points = np.asarray(mesh.vertices)
        
        score = get_add_s_score(true_pose, pred_pose, model_points)
        all_scores.append(score)

    # Calculate final accuracy
    correct_predictions = sum(s < threshold for s in all_scores)
    accuracy = (correct_predictions / len(all_scores)) * 100 if all_scores else 0
    
    print("\n--- Evaluation Complete ---")
    print(f"ADD-S Accuracy ({threshold*100:.1f}cm threshold): {accuracy:.2f}%")

    # --- Visualize one random result and save it ---
    print("\n--- Saving a visualization of a random prediction ---")
    random_index = random.randint(0, len(val_dataset)-1)
    data = val_dataset[random_index]
    
    # Run model on this one sample
    image_tensor_viz = data['image'].unsqueeze(0).to(device).float()
    with torch.no_grad():
        pred_translation_viz, pred_rotation_6d_viz = model(image_tensor_viz)

    # (Same conversion logic as above for the visualization sample)
    r1_viz = pred_rotation_6d_viz[:, 0:3]; r2_viz = pred_rotation_6d_viz[:, 3:6]
    r1_viz = torch.nn.functional.normalize(r1_viz, dim=1); r2_viz = r2_viz - (r1_viz * r2_viz).sum(dim=1, keepdim=True) * r1_viz; r2_viz = torch.nn.functional.normalize(r2_viz, dim=1); r3_viz = torch.cross(r1_viz, r2_viz, dim=1)
    pred_rotation_matrix_viz = torch.stack((r1_viz, r2_viz, r3_viz), dim=2).cpu().numpy()[0]
    pred_translation_viz = pred_translation_viz.cpu().numpy()[0]
    pred_pose_viz = np.eye(4); pred_pose_viz[:3, :3] = pred_rotation_matrix_viz; pred_pose_viz[:3, 3] = pred_translation_viz

    # Get original image and ground truth for visualization
    original_image = cv2.imread(val_dataset.data_paths[random_index]['rgb_path'])
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    intrinsics = np.array(val_dataset.data_paths[random_index]['camera_intrinsics'])
    true_pose_viz_6d = data['rotation'].numpy(); true_pose_viz_trans = data['translation'].numpy()
    true_rot_mat_viz_reshaped = true_pose_viz_6d.reshape(3, 2); r1_true_viz, r2_true_viz = true_rot_mat_viz_reshaped[:, 0], true_rot_mat_viz_reshaped[:, 1]; r3_true_viz = np.cross(r1_true_viz, r2_true_viz); full_rot_true_viz = np.stack((r1_true_viz, r2_true_viz, r3_true_viz), axis=1)
    true_pose_viz = np.eye(4); true_pose_viz[:3, :3] = full_rot_true_viz; true_pose_viz[:3, 3] = true_pose_viz_trans
    
    class_id_viz = val_dataset.data_paths[random_index]['objects'][0]['class_id']
    model_name_viz = class_id_to_name.get(class_id_viz)
    obj_model_path_viz = os.path.join(project_root, 'datasets', 'models', model_name_viz, 'textured.obj')

    # Draw boxes
    image_with_boxes = draw_3d_bounding_box(original_image.copy(), true_pose_viz, intrinsics, obj_model_path_viz, color=(0, 255, 0)) # Green
    image_with_boxes = draw_3d_bounding_box(image_with_boxes, pred_pose_viz, intrinsics, obj_model_path_viz, color=(255, 0, 0)) # Red

    # Save and show the final image
    plt.figure(figsize=(12, 8)); plt.imshow(image_with_boxes); plt.title("Model Prediction (Red) vs. Ground Truth (Green)"); plt.axis('off')
    plt.savefig("evaluation_result.png")
    print("Saved visualization to evaluation_result.png")
    # plt.show() # Commented out for non-interactive script execution

if __name__ == '__main__':
    evaluate_model()