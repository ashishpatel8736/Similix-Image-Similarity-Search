import os
import faiss
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Paths
FEATURES_PATH = "features/"
INDEX_PATH = "features/faiss_index.idx"

# Load FAISS index
faiss_index = faiss.read_index(INDEX_PATH)
print(f"Loaded FAISS index type: {type(faiss_index)}")  # Debugging step

# Load filenames
with open(os.path.join(FEATURES_PATH, "filenames.pkl"), "rb") as f:
    filenames = pickle.load(f)

# Load Pre-trained Model (ResNet50)
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

def extract_features(img_path):
    """Extract feature vector from an image."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    features = model.predict(img_array)
    return features.flatten()

def search_similar_images(query_image_path, top_k=5):
    """Finds the most similar images using FAISS."""
    query_vector = extract_features(query_image_path).reshape(1, -1)
    
    # Search FAISS index
    distances, indices = faiss_index.search(query_vector, top_k)

    
    # Get matched image filenames
    similar_images = [filenames[i] for i in indices[0]]
    
    # Display results
    fig, axes = plt.subplots(1, top_k + 1, figsize=(15, 5))
    
    # Show query image
    axes[0].imshow(image.load_img(query_image_path))
    axes[0].set_title("Query Image")
    axes[0].axis("off")
    
    # Show similar images
    for i, img_name in enumerate(similar_images):
        img_path = os.path.join("processed_dataset", img_name)
        axes[i + 1].imshow(image.load_img(img_path))
        axes[i + 1].set_title(f"Match {i+1}")
        axes[i + 1].axis("off")
    
    plt.show()

# Test with a sample image
query_image = "D:\\Python Projects\\data\\flowers\\0030.png"  # Replace with the actual image path
search_similar_images(query_image)

print(dir())  # Lists all defined variables
