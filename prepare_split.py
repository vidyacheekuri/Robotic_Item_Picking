import os
import shutil
from sklearn.model_selection import train_test_split
from src.components.data_ingestion import DataIngestion
from tqdm import tqdm

def create_validation_split():
    print("--- Preparing Validation Data Split ---")
    
    # 1. Load all data paths
    ingestion = DataIngestion()
    all_data_paths = ingestion.get_data_paths()
    print(f"Found {len(all_data_paths)} total data points.")

    # 2. Split into training and validation sets
    _, val_paths = train_test_split(all_data_paths, test_size=0.2, random_state=42)
    
    # Let's use a subset of 200 samples for a robust validation score
    val_subset = val_paths[:200]
    print(f"Selected a subset of {len(val_subset)} samples for the validation folder.")

    # 3. Create the destination folder
    output_folder = "validation_data"
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder) # Remove old folder to ensure a clean start
    os.makedirs(output_folder)
    print(f"Created a clean directory at: ./{output_folder}")

    # 4. Copy the files
    print("Copying validation files...")
    for item in tqdm(val_subset):
        # Copy the color image, depth image, and metadata file
        try:
            shutil.copy(item['rgb_path'], output_folder)
            shutil.copy(item['depth_path'], output_folder)
            
            # The metadata file needs to be found for the copy
            base_filename = os.path.basename(item['rgb_path']).split('-')[0]
            meta_filename = f"{base_filename}-meta.mat"
            meta_filepath = os.path.join(os.path.dirname(item['rgb_path']), meta_filename)
            if os.path.exists(meta_filepath):
                shutil.copy(meta_filepath, output_folder)

        except FileNotFoundError as e:
            print(f"Warning: Could not find a file to copy: {e}")
            continue

    print("\n--- Validation data folder created successfully! ---")
    print(f"Please zip the '{output_folder}' folder and upload it to Colab.")


if __name__ == "__main__":
    create_validation_split()