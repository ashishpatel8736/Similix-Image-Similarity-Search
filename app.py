import os
import faiss
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import streamlit as st
from PIL import Image

# Paths
FEATURES_PATH = "features/"
INDEX_PATH = "features/faiss_index.idx"

# Load FAISS index
index = faiss.read_index(INDEX_PATH)

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
    distances, indices = index.search(query_vector, top_k)
    similar_images = [filenames[i] for i in indices[0]]
    return similar_images

# Streamlit UI
st.title("Image Similarity SearchSimilix App")
uploaded_file = st.file_uploader("Choose an image...")

if uploaded_file is not None:
    query_image_path = f"temp_image.png"
    with open(query_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image(query_image_path, caption="Query Image", use_container_width=True)
    st.write("Searching for similar images...")
    similar_images = search_similar_images(query_image_path)
    st.write("## Similar Images")
    cols = st.columns(len(similar_images))
    for i, (col, img_name) in enumerate(zip(cols, similar_images)):
        img_path = os.path.join("processed_dataset", img_name)
        col.image(img_path, caption=f"Match {i+1}", use_container_width=True)


# Footer
# Displaying main message
st.markdown("<h2 style='text-align: center;'>Developed with ❤️ using Streamlit</h2>", unsafe_allow_html=True)

# Adding a separator
st.markdown("---")

# Footer with developer and technology stack
st.markdown("""
<div style='text-align: center;'>
    <p><strong>Developed by <a href='https://github.com/ashishpatel8736' target='_blank'>Ashish Patel</a></strong></p>
    <p>Powered by <strong>Streamlit</strong></p>
</div>
""", unsafe_allow_html=True)