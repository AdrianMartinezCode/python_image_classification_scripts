import os
import shutil
import numpy as np
import joblib
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
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

# Recursive Function to Compare and Move Images
def process_images(input_dir, output_dirs, feature_extractor, classifier):
    num_files = sum([len(files) for _, _, files in os.walk(input_dir)])
    copied_files = 0
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, filename)
                features = extract_features(file_path, feature_extractor)
                prediction = classifier.predict([features])[0]

                # Move to the corresponding folder (flattening folder structure)
                dest_path = os.path.join(output_dirs[prediction], filename)
                shutil.move(file_path, dest_path)
                copied_files += 1
                print(f"Moved: {file_path} -> {dest_path} , File {copied_files}/{num_files}")

def main(input_dir, not_similar_dir, similar_dir, neutral_dir, model_path):
    # Load Pre-trained Model
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

    # Load Trained Classifier
    classifier = joblib.load(model_path)

    # Create output directories if not exist
    os.makedirs(not_similar_dir, exist_ok=True)
    os.makedirs(similar_dir, exist_ok=True)
    os.makedirs(neutral_dir, exist_ok=True)

    output_dirs = {
        0: not_similar_dir,
        1: similar_dir,
        2: neutral_dir
    }

    process_images(input_dir, output_dirs, feature_extractor, classifier)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare images and classify into three classes.")
    parser.add_argument("input_dir", help="Path to the folder to search for images (including subfolders).")
    parser.add_argument("not_similar_dir", help="Path to the folder for 'Not Similar' images (flattened structure).")
    parser.add_argument("similar_dir", help="Path to the folder for 'Similar' images (flattened structure).")
    parser.add_argument("neutral_dir", help="Path to the folder for 'Neutral' images (flattened structure).")
    parser.add_argument("model_path", help="Path to the trained model.")
    args = parser.parse_args()

    main(args.input_dir, args.not_similar_dir, args.similar_dir, args.neutral_dir, args.model_path)