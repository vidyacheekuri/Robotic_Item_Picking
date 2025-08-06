import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your custom modules
from src.components.data_ingestion import DataIngestion
from src.components.dataset import YCBVideoDataset
from src.components.model_trainer import PoseNet
from torchvision import transforms
from sklearn.model_selection import train_test_split

def calculate_final_metrics():
    print("--- Starting Final Model Evaluation ---")
    
    # 1. Set up the device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load and prepare validation data
    ingestion = DataIngestion()
    all_data_paths = ingestion.get_data_paths()
    _, val_paths = train_test_split(all_data_paths, test_size=0.2, random_state=42)
    
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = YCBVideoDataset(data_paths=val_paths, transform=image_transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    print(f"Loaded {len(val_dataset)} validation samples.")

    # 3. Load the trained model
    model = PoseNet(pretrained=False)
    model.load_state_dict(torch.load('posenet_model.pth', map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # 4. Calculate final loss (MSE) on the validation set
    loss_function = nn.MSELoss()
    total_trans_loss = 0.0
    total_rot_loss = 0.0

    print("--- Calculating final validation loss (MSE) ---")
    with torch.no_grad():
        for data in tqdm(val_loader):
            images = data['image'].to(device).float()
            true_translations = data['translation'].to(device)
            true_rotations = data['rotation'].to(device)

            pred_translations, pred_rotations = model(images)
            
            total_trans_loss += loss_function(pred_translations, true_translations).item()
            total_rot_loss += loss_function(pred_rotations, true_rotations).item()

    avg_trans_loss = total_trans_loss / len(val_loader)
    avg_rot_loss = total_rot_loss / len(val_loader)
    
    print("\n--- Evaluation Complete ---")
    print(f"Final Validation Loss (Translation MSE): {avg_trans_loss:.6f}")
    print(f"Final Validation Loss (Rotation MSE):    {avg_rot_loss:.6f}")

if __name__ == '__main__':
    calculate_final_metrics()