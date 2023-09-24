import tensorflow as tf
import numpy as np
import os

# Load the trained generator model
generator = tf.keras.models.load_model('generator_model.h5')  # Replace with the actual path to your generator model

# Generate avatars from random noise vectors
num_samples = 10  # Specify the number of avatars to generate
latent_dim = 100  # Specify the dimension of the noise vector

for i in range(num_samples):
    noise = np.random.normal(0, 1, (1, latent_dim))  # Generate random noise vector
    generated_image = generator.predict(noise)[0]    # Generate an avatar
    generated_image = (generated_image + 1) / 2.0    # Convert values to the range [0, 1]

    # Save the generated avatar as an image file
    save_path = f'generated_avatar_{i}.png'
    tf.keras.preprocessing.image.save_img(save_path, generated_image)

print("Avatars generated and saved successfully!")
