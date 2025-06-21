import kagglehub
import shutil
import os

# Download the dataset
path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")

# Define the destination path
destination_path = "gtzan_dataset"

# Create the destination directory if it doesn't exist
os.makedirs(destination_path, exist_ok=True)

# Move the dataset to the desired location
shutil.move(path, destination_path)

print("Dataset moved to:", destination_path)