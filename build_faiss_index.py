import os
import pickle
import faiss
import numpy as np

# Paths
FEATURES_PATH = "features/image_features.pkl"
INDEX_PATH = "features/faiss_index.idx"

# Load extracted features
with open(FEATURES_PATH, "rb") as f:
    features_dict = pickle.load(f)

# Convert dictionary to a list of feature vectors and filenames
filenames = list(features_dict.keys())
feature_vectors = np.array(list(features_dict.values()), dtype=np.float32)

# Create FAISS index
d = feature_vectors.shape[1]  # Dimension of feature vectors
index = faiss.IndexFlatL2(d)  # L2 (Euclidean) distance index
index.add(feature_vectors)  # Add feature vectors to the index

# Save index and filenames
faiss.write_index(index, INDEX_PATH)

# Save filenames for reference
with open("features/filenames.pkl", "wb") as f:
    pickle.dump(filenames, f)

print(f"FAISS index created and saved at '{INDEX_PATH}'")
