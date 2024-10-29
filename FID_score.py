'''# Necessary imports
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer, Dropout
from tensorflow.keras.utils import to_categorical
from scipy.linalg import sqrtm  # For FID calculation

# Function to load images (grayscale, 7x10) from folder and convert to numpy array
def load_images_from_folder(folder, image_size=(7, 10)):
    images = []
    
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img = img.resize(image_size)
            img_array = np.array(img)
            images.append(img_array)

    return np.array(images)

# Load the real images
real_images_folder = '/Users/annafabia/Desktop/NIDS_deep_images/classifier/class_train_images'
X_train= load_images_from_folder(real_images_folder)
X_train = X_train / 255.0  # Normalize real images
X_train = np.expand_dims(X_train, axis=-1)  # Shape (samples, 7, 10, 1)

# Load the generated images
generated_images_folder = '/Users/annafabia/Desktop/NIDS_deep_images/conditional_GAN/delivery/generated_images'
X_generated= load_images_from_folder(generated_images_folder)
X_generated = X_generated / 255.0  # Normalize generated images
X_generated = np.expand_dims(X_generated, axis=-1)  # Shape (samples, 7, 10, 1)

# Load the trained feature extractor model
model = tf.keras.models.load_model('/Users/annafabia/Desktop/NIDS_deep_images/conditional_GAN/custom_feature_extractor_model.h5')

dummy_input = np.zeros((1, 7, 10, 1))  # Single dummy image with correct shape
model.predict(dummy_input)  # Call to build the model

model.summary()

# Remove the last layer(s) to use as a feature extractor
# Change the index if needed based on the model summary
feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-4].output)

# Remove the last layer to use the model as a feature extractor
# Ensure you're using the correct layer index or layer name
feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)

# Extract features for real and generated images
real_features = feature_extractor.predict(X_train)
generated_features = feature_extractor.predict(X_generated)

# Function to calculate FID between real and generated features
def calculate_fid(real_features, generated_features):
    # Calculate the mean and covariance of real images
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    # Calculate the mean and covariance of generated images
    mu_generated = np.mean(generated_features, axis=0)
    sigma_generated = np.cov(generated_features, rowvar=False)
    
    # Calculate mean difference
    diff = mu_real - mu_generated
    
    # Compute square root of product of covariances
    covmean = sqrtm(sigma_real.dot(sigma_generated))
    
    # Check and correct imaginary numbers from sqrtm
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Calculate FID score
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_generated - 2 * covmean)
    return fid

# Calculate FID
fid_score = calculate_fid(real_features, generated_features)
print(f"FID Score: {fid_score}")'''


import os
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from scipy.linalg import sqrtm
from skimage.transform import resize

# Function to load images from a local folder
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = Image.open(os.path.join(folder_path, filename))
        if img is not None:
            img = np.array(img)  # Convert to numpy array
            images.append(img)
    return np.array(images)

# Preprocess images for the InceptionV3 model
def preprocess_image(image, target_size=(299, 299)):
    # If the image is grayscale, repeat the single channel to make it 3-channel (RGB)
    if len(image.shape) == 2 or image.shape[-1] == 1:  # Assuming grayscale
        image = np.repeat(image[:, :, np.newaxis], 3, axis=-1)  # Convert to RGB
    
    # Resize image to target size
    image_resized = resize(image, target_size, anti_aliasing=True)
    
    # Ensure pixel values are in the range [0, 255]
    image_resized = (image_resized * 255).astype(np.float32)
    
    return image_resized

# Load pre-trained Inception model from TensorFlow Hub
inception_model = hub.load('https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4')

# Extract features from the Inception model
def get_inception_features(images):
    # Preprocess each image
    images_preprocessed = np.array([preprocess_image(img) for img in images])
    
    # Ensure values are scaled between -1 and 1 as required by InceptionV3
    images_preprocessed = (images_preprocessed / 127.5) - 1.0
    
    # Extract features from InceptionV3 model
    features = inception_model(images_preprocessed)
    
    return features

# Compute FID score using the Fréchet distance formula
def calculate_fid(real_features, generated_features):
    # Calculate mean and covariance of real and generated features
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    mu_gen = np.mean(generated_features, axis=0)
    sigma_gen = np.cov(generated_features, rowvar=False)
    
    # Calculate the FID score
    mu_diff = mu_real - mu_gen
    cov_mean, _ = sqrtm(sigma_real.dot(sigma_gen), disp=False)
    
    # Numerical error can produce a small imaginary component
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    
    fid_score = np.sum(mu_diff**2) + np.trace(sigma_real + sigma_gen - 2 * cov_mean)
    
    return fid_score

# Main function to compute FID score between real and generated images
def compute_fid(real_images_path, generated_images_path):
    # Load real and generated images from respective folders
    real_images = load_images_from_folder(real_images_path)
    generated_images = load_images_from_folder(generated_images_path)
    
    # Extract features for real and generated images
    real_features = get_inception_features(real_images)
    generated_features = get_inception_features(generated_images)
    
    # Calculate and return the FID score
    fid_score = calculate_fid(real_features, generated_features)
    return fid_score

# Example usage
real_images_path = '/Users/annafabia/Desktop/NIDS_deep_images/conditional_GAN/class_train_images_subset_20000'  # Replace with the actual path
generated_images_path = '/Users/annafabia/Desktop/NIDS_deep_images/conditional_GAN/subset_generated_20000'  # Replace with the actual path

fid_score = compute_fid(real_images_path, generated_images_path)
print(f"FID Score: {fid_score}")

