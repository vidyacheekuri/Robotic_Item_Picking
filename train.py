import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Import all your custom modules
from src.components.data_ingestion import DataIngestion
from src.components.dataset import YCBVideoDataset
from src.components.model_trainer import PoseNet
from torchvision import transforms

def train_model():
    print("--- Starting Model Training ---")
    
    # 1. Set up the device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load and subset the data
    ingestion = DataIngestion()
    all_data_paths = ingestion.get_data_paths()
    data_paths = all_data_paths
    print(f"Using {len(data_paths)} samples for training.")

    # 3. Split data and create training DataLoader
    train_paths, _ = train_test_split(data_paths, test_size=0.2, random_state=42)
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = YCBVideoDataset(data_paths=train_paths, transform=image_transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)

    # 4. Initialize Model, Loss, and Optimizer
    model = PoseNet(pretrained=True).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # 5. Run Training for 20 Epochs
    model.train()
    num_epochs = 20
    print(f"Training for {num_epochs} epoch...")

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            images = data['image'].to(device).float()
            true_translations = data['translation'].to(device)
            true_rotations = data['rotation'].to(device)

            optimizer.zero_grad()
            pred_translations, pred_rotations = model(images)
            loss_trans = loss_function(pred_translations, true_translations)
            loss_rot = loss_function(pred_rotations, true_rotations)
            total_loss = loss_trans + loss_rot
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            if (i + 1) % 100 == 0:
                print(f'Step [{i+1}/{len(train_loader)}], Loss: {total_loss.item():.4f}')
        
        avg_epoch_loss = running_loss / len(train_loader)
        print(f'--- End of Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f} ---')

    # 6. Save the Model
    save_path = 'posenet_model.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved successfully to {save_path}")

if __name__ == '__main__':
    train_model()