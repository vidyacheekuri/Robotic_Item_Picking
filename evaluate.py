import os
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import all your custom modules
from src.components.data_ingestion import DataIngestion
from src.components.dataset import YCBVideoDataset
from src.components.model_trainer import PoseNet
from src.utils import draw_3d_bounding_box, class_id_to_name, get_add_s_score
from torchvision import transforms

def evaluate_model():
    print("--- Starting Final Model Evaluation ---")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and subset the data
    ingestion = DataIngestion()
    all_data_paths = ingestion.get_data_paths()
    data_paths = all_data_paths[:5000] # Using the small subset for speed

    _, val_paths = train_test_split(data_paths, test_size=0.8, random_state=42) # Get a small slice
    val_paths = val_paths[:100] # Make sure it's exactly 100 samples

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = YCBVideoDataset(data_paths=val_paths, transform=image_transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Load the trained model
    model = PoseNet(pretrained=False)
    model.load_state_dict(torch.load('posenet_model.pth', map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    project_root = os.path.abspath(os.path.dirname(__file__))
    
    all_scores = []
    threshold = 0.02 # 2cm threshold for a correct pose

    print("--- Calculating ADD-S Metric on Validation Set ---")
    for i, data in enumerate(tqdm(val_loader)):
        image_tensor = data['image'].to(device).float()
        true_rotation_6d = data['rotation']
        true_translation = data['translation']

        with torch.no_grad():
            pred_translation, pred_rotation_6d = model(image_tensor)
    
        # Convert predicted 6D to 3x3 matrix
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
    print(f"ADD-S Accuracy ({threshold*100}cm threshold): {accuracy:.2f}%")

if __name__ == '__main__':
    evaluate_model()