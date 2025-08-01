import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import sys


class YCBVideoDataset(Dataset):
    def __init__(self, data_paths, transform=None):
        """
        Args:
            data_paths (list): List of dicts, where each dict contains info for one frame.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_paths = data_paths
        self.transform = transform

    def __len__(self):
        # The total number of samples in the dataset
        return len(self.data_paths)

    def __getitem__(self, idx):
        # Get the data path dictionary for the given index
        frame_data = self.data_paths[idx]
        
        # Load the RGB image
        image = cv2.imread(frame_data['rgb_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # For simplicity, we will start by training on the first object in each frame.
        # A more advanced implementation would handle multiple objects.
        if not frame_data['objects']:
            # Handle cases with no objects if they exist
            # For now, we can try to return the next valid item
            return self.__getitem__((idx + 1) % len(self))

        target_object = frame_data['objects'][0]
        
        # Extract the pose (rotation and translation)
        pose_matrix = np.array(target_object['pose'])
        
        # The pose is a 3x4 matrix [R | t]
        # R is the 3x3 rotation matrix, t is the 3x1 translation vector
        rotation_matrix = pose_matrix[:3, :3]
        translation = pose_matrix[:3, 3]
        
        # Convert the 3x3 ground truth rotation matrix to the 6D representation
        # This is done by taking the first two columns of the matrix
        rotation_6d = rotation_matrix[:, :2].T.flatten()
        
        # Create the sample dictionary
        sample = {
            'image': image, 
            'rotation': torch.from_numpy(rotation_6d).float(), 
            'translation': torch.from_numpy(translation).float()
        }
        
        # Apply transformations if any
        if self.transform:
            sample['image'] = self.transform(sample['image'])
            
        return sample