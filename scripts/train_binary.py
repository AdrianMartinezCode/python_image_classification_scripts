import os
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.svm import SVC
import joblib
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

def train(input_folder_similar, input_folder_not_similar, model_path):
    # Load Pre-trained Model
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

    # Extract Features for All Images
    features = []
    labels = []

    # Positive Samples (Similar)
    for filename in os.listdir(input_folder_similar):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder_similar, filename)
            features.append(extract_features(img_path, feature_extractor))
            labels.append(1)  # Label as "similar"

    # Negative Samples (Not Similar)
    for filename in os.listdir(input_folder_not_similar):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder_not_similar, filename)
            features.append(extract_features(img_path, feature_extractor))
            labels.append(0)  # Label as "not similar"

    # Train a Classifier
    features = np.array(features)
    labels = np.array(labels)

    if len(np.unique(labels)) < 2:
        raise ValueError("The number of classes has to be greater than one; ensure both 'similar' and 'not similar' samples are included.")

    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(features, labels)

    # Save the Model
    joblib.dump(classifier, model_path)
    print(f"Model saved to {model_path}")

# Main Function
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument("input_folder_similar", help="Path to the folder to search for images.")
    parser.add_argument("input_folder_not_similar", help="Path to the folder to search for images.")
    parser.add_argument("model_path", default=model_path, help="Path to save the trained model.")
    args = parser.parse_args()

    train(args.input_folder_similar, args.input_folder_not_similar, args.model_path)
