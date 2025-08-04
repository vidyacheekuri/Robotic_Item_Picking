import torch
import torch.nn as nn
import torchvision.models as models

class PoseNet(nn.Module):
    def __init__(self, pretrained=True):
        super(PoseNet, self).__init__()
        
        # 1. Load a pre-trained ResNet50 model using the new 'weights' API
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
        else:
            weights = None
        
        self.backbone = models.resnet50(weights=weights)
        
        # 2. Isolate the feature-extracting layers
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # 3. Define the regression head
        self.head = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU()
        )

        # 4. Define the two output layers
        self.fc_translation = nn.Linear(512, 3)
        self.fc_rotation = nn.Linear(512, 6) 

    def forward(self, x):
        features = self.backbone(x)
        head_output = self.head(features)
        translation = self.fc_translation(head_output)
        rotation_6d = self.fc_rotation(head_output)
        return translation, rotation_6d