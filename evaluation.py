import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_addons as tfa
import os

# Load the InceptionV3 model from TensorFlow Hub (used for calculating Inception Score)
inception_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
inception_model = tf.keras.Model(inputs=inception_model.input, outputs=inception_model.layers[-1].output)

# Define a function to preprocess and resize images
def preprocess_image(image):
    image = tf.image.resize(image, (299, 299))
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image

# Load and preprocess a reference dataset for FID calculation (e.g., real avatars)
# Ensure that the reference dataset is preprocessed and saved as a numpy array
reference_data = np.load('reference_dataset.npy')  # Replace with the path to your reference dataset

# Calculate Inception Score for the generated avatars
def calculate_inception_score(images, model, batch_size=32):
    # Preprocess images and compute activations
    images = np.array([preprocess_image(img) for img in images])
    activations = model.predict(images)

    # Calculate the mean and covariance of the activations
    p_yx = np.exp(activations) / np.exp(activations).sum(axis=1, keepdims=True)
    KL_divs = p_yx * (np.log(p_yx) - np.log(p_yx.mean(axis=0, keepdims=True)))
    KL_divergence = KL_divs.sum(axis=1)
    inception_score = np.exp(KL_divergence.mean())

    return inception_score

inception_scores = calculate_inception_score(generated_images, inception_model)
print(f"Inception Scores: {inception_scores}")

# Calculate FID (Frechet Inception Distance) between the generated avatars and the reference dataset
def calculate_fid(real_data, generated_data, model, batch_size=32):
    real_activations = model.predict(real_data)
    generated_activations = model.predict(generated_data)

    m1 = real_activations.mean(axis=0)
    m2 = generated_activations.mean(axis=0)

    s1 = np.cov(real_activations, rowvar=False)
    s2 = np.cov(generated_activations, rowvar=False)

    diff = m1 - m2
    covmean, _ = tfa.metrics.FID._sqrtm(s1.dot(s2), eps=1e-6)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_score = np.sum(diff**2) + np.trace(s1 + s2 - 2*covmean)

    return fid_score

fid_scores = calculate_fid(reference_data, generated_images, inception_model)
print(f"FID Scores: {fid_scores}")
