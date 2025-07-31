import os
import sys
from scipy.io import loadmat
import numpy as np
from src.exception import CustomException
from src.logger import logging

class DataIngestion:
    def __init__(self):
        # Get the absolute path to the project's root directory
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        
        # Define absolute paths to the data directories
        self.data_dir = os.path.join(self.project_root, 'datasets', 'data')
        self.models_dir = os.path.join(self.project_root, 'datasets', 'models')
        
    def get_data_paths(self):
        """
        Scans the data directory to get structured paths for images, depth maps, and labels.
        """
        logging.info("Starting to scan for data paths...")
        structured_paths = []
        
        sequences = sorted(os.listdir(self.data_dir))

        for seq in sequences:
            seq_path = os.path.join(self.data_dir, seq)
            if not os.path.isdir(seq_path):
                continue

            frame_files = sorted(os.listdir(seq_path))
            frame_ids = sorted(list(set([f.split('-')[0] for f in frame_files])))

            for frame_id in frame_ids:
                try:
                    # --- CHANGE 1: Look for .jpg and .mat files ---
                    rgb_path = os.path.join(seq_path, f"{frame_id}-color.jpg")
                    depth_path = os.path.join(seq_path, f"{frame_id}-depth.png")
                    meta_path = os.path.join(seq_path, f"{frame_id}-meta.mat") # Changed from .yml to .mat

                    if os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(meta_path):
                        
                        # --- CHANGE 2: Read the .mat file using scipy ---
                        meta_data = loadmat(meta_path)

                        frame_data = {
                            'rgb_path': rgb_path,
                            'depth_path': depth_path,
                            'camera_intrinsics': meta_data['intrinsic_matrix'].tolist(),
                            'objects': []
                        }

                        # Add pose info for each object in the frame
                        for i, obj_id in enumerate(meta_data['cls_indexes'].flatten()):
                            obj_info = {
                                'class_id': int(obj_id),
                                'pose': meta_data['poses'][:, :, i].tolist()
                            }
                            frame_data['objects'].append(obj_info)
                        
                        structured_paths.append(frame_data)

                except Exception as e:
                    logging.warning(f"Could not process frame {frame_id} in sequence {seq}: {e}")

        logging.info(f"Successfully found {len(structured_paths)} frames with complete data.")
        return structured_paths

# Example usage:
if __name__ == "__main__":
    try:
        ingestion = DataIngestion()
        data_paths = ingestion.get_data_paths()
        
        if data_paths:
            print("Successfully processed data paths.")
            print("Example of the first data entry:")
            print(data_paths[0])
        else:
            print("No data paths were found. Check your data directory structure.")
            
    except Exception as e:
        raise CustomException(e, sys)