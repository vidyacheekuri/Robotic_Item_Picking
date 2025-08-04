import time
from src.components.data_ingestion import DataIngestion

print("--- Starting standalone data ingestion test ---")

start_time = time.time()

ingestion = DataIngestion()
data_paths = ingestion.get_data_paths()

end_time = time.time()

print(f"Finished!")
print(f"Found {len(data_paths)} data points.")
print(f"Total time taken: {end_time - start_time:.2f} seconds")