import os
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import random
from scipy.io import loadmat

# Import all your custom modules
from src.components.model_trainer import PoseNet
from src.utils import draw_3d_bounding_box, class_id_to_name
from torchvision import transforms

def test_model(num_tests=5):
    """
    Loads the trained model and runs inference on a few random validation samples,
    saving the visual results.
    """
    print("--- Starting Model Test ---")

    # --- 1. SETUP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = 'posenet_model.pth'
    validation_folder_path = 'validation_data'
    output_folder = 'test_results'
    os.makedirs(output_folder, exist_ok=True) # Create output folder if it doesn't exist

    # --- 2. LOAD THE TRAINED MODEL ---
    model = PoseNet(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # --- 3. PREPARE DATA ---
    # Get a list of all available validation frames
    frame_files = os.listdir(validation_folder_path)
    frame_ids = sorted(list(set([f.split('-')[0] for f in frame_files])))
    
    # Select a few random frames to test
    random_frame_ids = random.sample(frame_ids, min(num_tests, len(frame_ids)))
    print(f"Selected {len(random_frame_ids)} random frames for testing.")

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- 4. RUN INFERENCE AND VISUALIZE ---
    for i, frame_id in enumerate(random_frame_ids):
        print(f"\n--- Testing sample {i+1}/{len(random_frame_ids)} (Frame: {frame_id}) ---")
        
        # Load data for the current frame
        rgb_path = os.path.join(validation_folder_path, f"{frame_id}-color.jpg")
        meta_path = os.path.join(validation_folder_path, f"{frame_id}-meta.mat")
        
        original_image = cv2.imread(rgb_path)
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        meta_data = loadmat(meta_path)
        
        # Prepare image for the model
        image_tensor = image_transform(original_image_rgb).unsqueeze(0).to(device).float()

        # Run inference
        with torch.no_grad():
            pred_translation, pred_rotation_6d = model(image_tensor)

        # --- Process prediction (convert 6D rotation to 4x4 pose matrix) ---
        r1 = pred_rotation_6d[:, 0:3]; r2 = pred_rotation_6d[:, 3:6]
        r1 = torch.nn.functional.normalize(r1, dim=1)
        r2 = r2 - (r1 * r2).sum(dim=1, keepdim=True) * r1
        r2 = torch.nn.functional.normalize(r2, dim=1)
        r3 = torch.cross(r1, r2, dim=1)
        pred_rotation_matrix = torch.stack((r1, r2, r3), dim=2).cpu().numpy()[0]
        pred_translation = pred_translation.cpu().numpy()[0]
        pred_pose = np.eye(4); pred_pose[:3, :3] = pred_rotation_matrix; pred_pose[:3, 3] = pred_translation

        # --- Process ground truth ---
        true_pose_data = meta_data['poses'][:, :, 0] # Assuming first object
        true_pose = np.eye(4); true_pose[:3, :] = true_pose_data

        # --- Get 3D model info for visualization ---
        class_id = int(meta_data['cls_indexes'].flatten()[0])
        model_name = class_id_to_name.get(class_id)
        project_root = os.path.abspath(os.path.dirname(__file__))
        obj_model_path = os.path.join(project_root, 'datasets', 'models', model_name, 'textured.obj')
        
        intrinsics = meta_data['intrinsic_matrix']

        # --- Draw bounding boxes ---
        image_with_boxes = draw_3d_bounding_box(original_image_rgb.copy(), true_pose, intrinsics, obj_model_path, color=(0, 255, 0)) # Green
        image_with_boxes = draw_3d_bounding_box(image_with_boxes, pred_pose, intrinsics, obj_model_path, color=(255, 0, 0)) # Red

        # --- Save the result ---
        output_path = os.path.join(output_folder, f"test_result_{i+1}_{frame_id}.png")
        plt.figure(figsize=(12, 8))
        plt.imshow(image_with_boxes)
        plt.title(f"Prediction (Red) vs. Ground Truth (Green) - Frame {frame_id}")
        plt.axis('off')
        plt.savefig(output_path)
        plt.close() # Close the plot to save memory
        print(f"Saved visualization to {output_path}")

    print("\n--- Testing complete! ---")

if __name__ == '__main__':
    # You need your validation_data folder and posenet_model.pth file
    # in the same directory to run this script.
    # You also need the 'datasets/models' folder for visualization.
    test_model()
