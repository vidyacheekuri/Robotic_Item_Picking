import os
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Import all your custom modules
from src.components.data_ingestion import DataIngestion
from src.components.dataset import YCBVideoDataset
from src.components.model_trainer import PoseNet
from src.utils import draw_3d_bounding_box, class_id_to_name
from torchvision import transforms

def evaluate_model():
    print("--- Starting Model Evaluation ---")
    
    # 1. Set up the device (MPS for Apple Silicon)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: mps (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")

    # 2. Load and subset the data paths
    ingestion = DataIngestion()
    all_data_paths = ingestion.get_data_paths()
    data_paths = all_data_paths[:5000] # Using the small subset
    print(f"Using a subset of {len(data_paths)} samples for evaluation.")

    # 3. Create validation set
    _, val_paths = train_test_split(data_paths, test_size=0.2, random_state=42)

    # 4. Define transforms and create DataLoader
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = YCBVideoDataset(data_paths=val_paths, transform=image_transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0)

    # 5. Load the trained model
    model = PoseNet(pretrained=False) # Set pretrained=False as we are loading our own weights
    model_path = 'posenet_model.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # 6. Get one sample and visualize the prediction
    print("Visualizing a prediction...")
    data = next(iter(val_loader))
    image_tensor = data['image'].to(device).float()
    true_rotation = data['rotation']
    true_translation = data['translation']

    # We need the original, untransformed image for visualization
    # The DataLoader's dataset object holds the path info
    sample_index = val_loader.dataset.data_paths.index(val_loader.sampler.data_source.data_paths[0])
    original_image_path = val_loader.dataset.data_paths[sample_index]['rgb_path']
    original_image = cv2.imread(original_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    intrinsics = np.array(val_loader.dataset.data_paths[sample_index]['camera_intrinsics'])
    
    with torch.no_grad():
        pred_translation, pred_rotation_6d = model(image_tensor)
    
        # Convert predicted 6D rotation back to a 3x3 matrix
        b, _ = pred_rotation_6d.shape
        r1 = pred_rotation_6d[:, 0:3]
        r2 = pred_rotation_6d[:, 3:6]
        r1 = torch.nn.functional.normalize(r1, dim=1)
        r2 = r2 - (r1 * r2).sum(dim=1, keepdim=True) * r1
        r2 = torch.nn.functional.normalize(r2, dim=1)
        r3 = torch.cross(r1, r2, dim=1)
        pred_rotation_matrix = torch.stack((r1, r2, r3), dim=2).cpu().numpy()[0]
        
        pred_translation = pred_translation.cpu().numpy()[0]
        
        pred_pose = np.eye(4)
        pred_pose[:3, :3] = pred_rotation_matrix
        pred_pose[:3, 3] = pred_translation
        
        # Recreate the ground truth pose matrix
        true_rotation_matrix = true_rotation.numpy()[0].reshape(3, 2)
        r1_true, r2_true = true_rotation_matrix[:, 0], true_rotation_matrix[:, 1]
        r3_true = np.cross(r1_true, r2_true)
        full_rot_true = np.stack((r1_true, r2_true, r3_true), axis=1)
        
        true_pose = np.eye(4)
        true_pose[:3, :3] = full_rot_true
        true_pose[:3, 3] = true_translation.numpy()[0]

    # Get the object model path
    class_id = val_loader.dataset.data_paths[sample_index]['objects'][0]['class_id']
    model_name = class_id_to_name.get(class_id)
    project_root = os.path.abspath(os.path.dirname(__file__))
    obj_model_path = os.path.join(project_root, 'datasets', 'models', model_name, 'textured.obj')

    # Draw bounding boxes
    image_with_boxes = draw_3d_bounding_box(original_image.copy(), true_pose, intrinsics, obj_model_path, color=(0, 255, 0)) # Green
    image_with_boxes = draw_3d_bounding_box(image_with_boxes, pred_pose, intrinsics, obj_model_path, color=(255, 0, 0)) # Red

    # Display the final image
    plt.figure(figsize=(12, 8))
    plt.imshow(image_with_boxes)
    plt.title("Model Prediction (Red) vs. Ground Truth (Green)")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    evaluate_model()