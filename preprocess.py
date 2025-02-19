import os
import cv2
import numpy as np

# Set dataset path
DATASET_PATH = "dataset/"
PROCESSED_PATH = "processed_dataset/"

# Create processed dataset directory if it doesn't exist
os.makedirs(PROCESSED_PATH, exist_ok=True)

# Target size
IMG_SIZE = (224, 224)

def preprocess_image(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        return False  # Skip if image is unreadable

    img = cv2.resize(img, IMG_SIZE)  # Resize to 224x224
    img = img / 255.0  # Normalize to [0,1]

    # Convert .bmp to .jpg for consistency
    if image_path.lower().endswith(".bmp"):
        output_path = output_path.replace(".bmp", ".jpg")

    cv2.imwrite(output_path, (img * 255).astype(np.uint8))  # Save as uint8
    return True

# Process all images
for filename in os.listdir(DATASET_PATH):
    if filename.lower().endswith((".jpg", ".bmp", ".png")):  # Accept only valid formats
        input_path = os.path.join(DATASET_PATH, filename)
        output_path = os.path.join(PROCESSED_PATH, filename)
        preprocess_image(input_path, output_path)

print(f"Preprocessing complete. Processed images saved in '{PROCESSED_PATH}'")
