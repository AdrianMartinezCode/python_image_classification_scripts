import os
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.svm import SVC
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

def main(similar_dir, neutral_dir, not_similar_dir, model_path):
    # Load Pre-trained Model
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

    # Extract Features for All Images
    features = []
    labels = []

    # Class 0: Not Similar
    for filename in os.listdir(not_similar_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(not_similar_dir, filename)
            features.append(extract_features(img_path, feature_extractor))
            labels.append(0)

    # Class 1: Similar
    for filename in os.listdir(similar_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(similar_dir, filename)
            features.append(extract_features(img_path, feature_extractor))
            labels.append(1)

    # Class 2: Neutral
    for filename in os.listdir(neutral_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(neutral_dir, filename)
            features.append(extract_features(img_path, feature_extractor))
            labels.append(2)

    # Train a Multi-class Classifier
    features = np.array(features)
    labels = np.array(labels)

    if len(np.unique(labels)) < 3:
        raise ValueError("The number of classes has to be at least three; ensure all three classes are included.")

    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(features, labels)

    # Save the Model
    joblib.dump(classifier, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a multi-class classifier for image similarity.")
    parser.add_argument("similar_dir", help="Path to the folder for 'Similar' images.")
    parser.add_argument("neutral_dir", help="Path to the folder for 'Neutral' images.")
    parser.add_argument("not_similar_dir", help="Path to the folder for 'Not Similar' images.")
    parser.add_argument("model_path", help="Path to save the trained model.")
    args = parser.parse_args()

    main(args.similar_dir, args.neutral_dir, args.not_similar_dir, args.model_path)