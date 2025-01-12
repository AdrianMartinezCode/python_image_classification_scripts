import os
import shutil
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
import joblib
import cv2
import argparse

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

# Recursive Function to Classify and Move Images
def move_images_to_clusters(input_dir, output_dir, model_path):
    # Load Pre-trained Model
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

    # Load Saved Clustering Model
    kmeans = joblib.load(model_path)

    # Create Output Folders for Clusters
    n_clusters = kmeans.n_clusters
    for cluster_id in range(n_clusters):
        cluster_dir = os.path.join(output_dir, f"class_{cluster_id}")
        os.makedirs(cluster_dir, exist_ok=True)

    # Classify Images and Move to Corresponding Clusters
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, filename)
                features = extract_features(img_path, feature_extractor)

                # Ensure the features are 2D for KMeans
                features = np.array(features, dtype=np.float64).reshape(1, -1)
                
                cluster_id = kmeans.predict(features)[0]

                # Move to the corresponding cluster folder
                dest_path = os.path.join(output_dir, f"class_{cluster_id}", os.path.basename(img_path))
                shutil.copy(img_path, dest_path)
                print(f"Copied {img_path} to {dest_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify images using K-Means clustering model and move to cluster folders.")
    parser.add_argument("input_dir", help="Path to the folder containing images.")
    parser.add_argument("output_dir", help="Path to the folder where clustered images will be saved.")
    parser.add_argument("model_path", help="Path to the saved clustering model.")
    args = parser.parse_args()

    move_images_to_clusters(args.input_dir, args.output_dir, args.model_path)