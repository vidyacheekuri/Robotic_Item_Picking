import torch
import cv2
import numpy as np
import torch.nn as nn
import os
from scipy.io import loadmat
import boto3

from src.components.model_trainer import PoseNet
from src.utils import draw_3d_bounding_box, class_id_to_name
from torchvision import transforms

# --- 1. SETUP ---
DEVICE = torch.device("cpu") # Use CPU for deployment
MODEL_PATH = 'posenet_model.pth'
S3_BUCKET_NAME = 'sree-vidya-posenet-model'  
S3_MODEL_KEY = 'posenet_model.pth'

# --- 2. DOWNLOAD & LOAD THE MODEL ---
# This block will run when the application starts.
# It checks if the model file exists, and if not, downloads it from S3.
if not os.path.exists(MODEL_PATH):
    print(f"--- Model not found locally. Downloading from S3 bucket: {S3_BUCKET_NAME} ---")
    try:
        s3 = boto3.client('s3')
        s3.download_file(S3_BUCKET_NAME, S3_MODEL_KEY, MODEL_PATH)
        print("--- Model downloaded successfully. ---")
    except Exception as e:
        print(f"!!! ERROR: Failed to download model from S3. {e} !!!")

# Instantiate the model architecture
model = PoseNet(pretrained=False)
# Load the saved weights from the (now downloaded) file
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
# Move the model to the selected device
model.to(DEVICE)
# Set the model to evaluation mode
model.eval()
print("--- Model loaded successfully and ready for predictions ---")


def predict_and_visualize(image_path: str) -> np.ndarray:
    """
    Takes an image path, predicts the 6D pose, and draws the result on the image.

    Args:
        image_path (str): The file path of the input image.

    Returns:
        np.ndarray: The image with the predicted bounding box drawn on it (as a numpy array).
    """
    # --- 3. PREPARE THE INPUT IMAGE ---
    original_image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = image_transform(image_rgb).unsqueeze(0).to(DEVICE).float()

    # --- 4. RUN INFERENCE ---
    with torch.no_grad():
        pred_translation, pred_rotation_6d = model(image_tensor)

    # --- 5. POST-PROCESS THE OUTPUT ---
    r1 = pred_rotation_6d[:, 0:3]; r2 = pred_rotation_6d[:, 3:6]
    r1 = nn.functional.normalize(r1, dim=1)
    r2 = r2 - (r1 * r2).sum(dim=1, keepdim=True) * r1
    r2 = nn.functional.normalize(r2, dim=1)
    r3 = torch.cross(r1, r2, dim=1)
    pred_rotation_matrix = torch.stack((r1, r2, r3), dim=2).cpu().numpy()[0]
    pred_translation = pred_translation.cpu().numpy()[0]
    pred_pose = np.eye(4); pred_pose[:3, :3] = pred_rotation_matrix; pred_pose[:3, 3] = pred_translation
    
    # --- 6. DRAW THE VISUALIZATION ---
    # To draw, we need the 3D model and camera intrinsics.
    # This part is a simplification. For a real app, you'd need to know which object is in the image.
    # We'll just use a default object (the cracker box) and standard intrinsics for this demo.
    
    # Get a default object model to draw
    model_name = class_id_to_name.get(2) # Default to class_id 2 (cracker box)
    project_root = os.path.abspath(os.path.dirname(__file__))
    obj_model_path = os.path.join(project_root, 'datasets', 'models', model_name, 'textured.obj')
    
    # Use standard YCB camera intrinsics
    intrinsics = np.array([[1066.778, 0.0, 312.9869], [0.0, 1067.487, 241.3109], [0.0, 0.0, 1.0]])

    if os.path.exists(obj_model_path):
        image_with_box = draw_3d_bounding_box(image_rgb.copy(), pred_pose, intrinsics, obj_model_path, color=(255, 0, 0)) # Draw red box
        return cv2.cvtColor(image_with_box, cv2.COLOR_RGB2BGR) # Convert back to BGR for saving
    else:
        print(f"Warning: 3D model not found at {obj_model_path}. Returning original image.")
        return original_image