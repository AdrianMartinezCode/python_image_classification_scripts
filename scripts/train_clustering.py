import os
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans
import joblib
import cv2
import argparse
import random

# Preprocess Images
def preprocess_image(image_path, target_size=(448, 448)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = preprocess_input(np.expand_dims(img, axis=0))
    return img

# Extract Features
def extract_features(image_path, model):
    processed_image = preprocess_image(image_path)
    return model.predict(processed_image).flatten()

def train_clustering(input_dir, model_path, n_clusters=9, max_files=1000):
    # Load Pre-trained Model
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

    # Collect All Images
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, filename))

    # Limit the Number of Files to Use
    if len(image_paths) > max_files:
        image_paths = random.sample(image_paths, max_files)

    print(f"Using {len(image_paths)} images for training.")

    # Extract Features
    features = []
    for img_path in image_paths:
        features.append(extract_features(img_path, feature_extractor))

    # Ensure Features are Numpy Array with dtype=float64
    features = np.array(features, dtype=np.float64)

    # Train K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features)

    # Save the Clustering Model
    joblib.dump(kmeans, model_path)
    print(f"Clustering model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train K-Means clustering model on images.")
    parser.add_argument("input_dir", help="Path to the folder containing images.")
    parser.add_argument("model_path", help="Path to save the clustering model.")
    parser.add_argument("--n_clusters", type=int, default=9, help="Number of clusters/classes (default: 9).")
    parser.add_argument("--max_files", type=int, default=1000, help="Maximum number of files to use for training (default: 1000).")
    args = parser.parse_args()

    train_clustering(args.input_dir, args.model_path, n_clusters=args.n_clusters, max_files=args.max_files)
