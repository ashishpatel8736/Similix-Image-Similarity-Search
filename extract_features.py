# The code snippet above demonstrates how to extract image features using a pre-trained ResNet50 model. The ResNet50 model is loaded without the classification layer, and images are preprocessed using the `preprocess_input` function from `tensorflow.keras.applications.resnet50`. The extracted features are then saved in a dictionary and stored in a pickle file for later use. The extracted features can be used for tasks such as image similarity, image retrieval, and image classification.

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
import pickle

# Paths
DATASET_PATH = "processed_dataset/"
FEATURES_PATH = "features/"
os.makedirs(FEATURES_PATH, exist_ok=True)

# Load Pre-trained Model (ResNet50 without classification layer)
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

def extract_features(img_path):
    """Extracts feature vector from an image using ResNet50."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess for ResNet50

    # Extract features
    features = model.predict(img_array)
    return features.flatten()  # Flatten to 1D vector

# Dictionary to store image features
features_dict = {}

# Process all images and extract features
for filename in tqdm(os.listdir(DATASET_PATH), desc="Extracting Features"):
    img_path = os.path.join(DATASET_PATH, filename)
    feature_vector = extract_features(img_path)
    features_dict[filename] = feature_vector

# Save extracted features using pickle
with open(os.path.join(FEATURES_PATH, "image_features.pkl"), "wb") as f:
    pickle.dump(features_dict, f)

print(f"Feature extraction complete. Features saved in '{FEATURES_PATH}image_features.pkl'")