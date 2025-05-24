import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model_path = 'fine_tuned_spoof_model_600x600.h5'  # Path to your saved model
spoof_model = load_model(model_path)

# Function to preprocess image for prediction
def preprocess_image(img_path, target_size=(600, 600)):
    """
    Preprocess the input image to match the model's input requirements.
    Args:
        img_path (str): Path to the input image.
        target_size (tuple): Desired resolution of the image (width, height).
    Returns:
        np.array: Preprocessed image ready for prediction.
    """
    img = cv2.imread(img_path)  # Load the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, target_size)  # Resize to target size
    img_array = np.array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
    return img_array

# Function to predict if the face is real or fake
def predict_image(img_path):
    """
    Predict if the input image is real or fake.
    Args:
        img_path (str): Path to the input image.
    Returns:
        tuple: Probabilities for real and fake.
    """
    img_array = preprocess_image(img_path)  # Preprocess the image
    prediction = spoof_model.predict(img_array)  # Get the prediction
    real_prob = prediction[0][0]  # Probability of being real
    fake_prob = 1 - real_prob  # Probability of being fake
    return real_prob, fake_prob

# Function to process and display results for all images in a folder
def process_folder(folder_path):
    """
    Process all images in a specified folder and display predictions.
    Args:
        folder_path (str): Path to the folder containing images.
    """
    # Get a list of all image files in the folder
    valid_extensions = (".jpg", ".jpeg", ".png")
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print("No valid image files found in the folder.")
        return

    for img_path in image_files:
        print(f"Processing: {img_path}")
        real_prob, fake_prob = predict_image(img_path)

        print(f"Real Probability: {real_prob * 100:.2f}%")
        print(f"Fake Probability: {fake_prob * 100:.2f}%\n")

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow('Input Image', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # Show the image
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    folder_path = r"C:\Users\Aadya\OneDrive\Desktop\final\final\train\real" # Replace with the path to your folder containing images
    process_folder(folder_path)

