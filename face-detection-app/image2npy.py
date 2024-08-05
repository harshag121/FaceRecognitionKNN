import cv2
import numpy as np
import os

# Path to the dataset directory
dataset_path = "face-recognition-dataset/"
npy_save_path = "face-recognition-npy/"

# Ensure the npy save path exists
os.makedirs(npy_save_path, exist_ok=True)

# Function to process images and save as .npy
def convert_images_to_npy(dataset_path, npy_save_path):
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_folder):
            images = []
            for img_name in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_name)
                if img_path.endswith(".jpg") or img_path.endswith(".jpeg") or img_path.endswith(".png"):
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Resize image if necessary (example: 100x100)
                        img = cv2.resize(img, (100, 100))
                        # Convert to grayscale if necessary
                        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        images.append(img)
            if images:
                images_np = np.array(images)
                npy_filename = os.path.join(npy_save_path, f"{person_name}.npy")
                np.save(npy_filename, images_np)
                print(f"Saved {npy_filename} with shape {images_np.shape}")

# Convert the images
convert_images_to_npy(dataset_path, npy_save_path)
