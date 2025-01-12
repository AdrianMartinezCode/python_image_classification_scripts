import os
import shutil
import numpy as np
import joblib
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
import cv2

# Paths
model_path = 'models/trained_model.pkl'



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
def process_images(input_dir, output_dir, model_path):
    # Load Pre-trained Model
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

    # Load Trained Classifier
    classifier = joblib.load(model_path)
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, filename)
                features = extract_features(file_path, feature_extractor)
                prediction = classifier.predict([features])

                if prediction[0] == 1:  # If classified as "similar"
                    relative_path = os.path.relpath(root, input_dir)
                    dest_dir = os.path.join(output_dir, relative_path)
                    os.makedirs(dest_dir, exist_ok=True)
                    dest_path = os.path.join(dest_dir, filename)
                    shutil.move(file_path, dest_path)
                    print(f"Moved: {file_path} -> {dest_path}")

# Main Function
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare images and move matches.")
    parser.add_argument("input_dir", help="Path to the folder to search for images.")
    parser.add_argument("output_dir", help="Path to the folder where similar images will be moved.")
    parser.add_argument("model_path", default=model_path, help="Path to the trained model.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    process_images(args.input_dir, args.output_dir, args.model_path)