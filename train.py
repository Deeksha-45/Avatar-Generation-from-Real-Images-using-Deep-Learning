import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from tqdm import tqdm
import cv2

latent_dim = 100 

# Define the generator model
def build_generator(latent_dim):
    generator = models.Sequential()

    # Add a dense layer that maps from the latent space to an intermediate layer
    generator.add(layers.Dense(128, input_dim=latent_dim))
    generator.add(layers.LeakyReLU(alpha=0.2))

    # Add another dense layer and an activation function
    generator.add(layers.Dense(256))
    generator.add(layers.LeakyReLU(alpha=0.2))

    # Add the output layer with appropriate activation and shape
    generator.add(layers.Dense(784, activation='tanh'))
    generator.add(layers.Reshape((28, 28, 1)))

    return generator

# Define the discriminator model
def build_discriminator(input_shape):
    discriminator = models.Sequential()

    # Flatten the input image
    discriminator.add(layers.Flatten(input_shape=input_shape))

    # Add dense layers with LeakyReLU activation
    discriminator.add(layers.Dense(256))
    discriminator.add(layers.LeakyReLU(alpha=0.2))

    discriminator.add(layers.Dense(128))
    discriminator.add(layers.LeakyReLU(alpha=0.2))

    # Output layer with sigmoid activation for binary classification
    discriminator.add(layers.Dense(1, activation='sigmoid'))

    return discriminator

# Define the GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = layers.Input(shape=(latent_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = models.Model(gan_input, gan_output)
    return gan

# Define loss functions and optimizers
discriminator = build_discriminator(input_shape=(28, 28, 1))
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

generator = build_generator(latent_dim)
discriminator.trainable = False

gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Define training parameters
epochs = 10000
batch_size = 128
sample_interval = 1000

# Load and preprocess your dataset here
# Assuming you have a preprocessed dataset directory with images
dataset_directory = r'E:\face dataset\img_align_celeba\img_align_celeba'

# Initialize an empty list to store file paths of preprocessed images
preprocessed_image_paths = []

# Loop through the dataset and collect file paths of preprocessed images
for filename in tqdm(os.listdir(dataset_directory)):
    if filename.endswith('.jpg'):
        img_path = os.path.join(dataset_directory, filename)
        preprocessed_image_paths.append(img_path)

# Training loop
for epoch in range(epochs):
    # Train the discriminator and generator as you did before

    # Print progress and save generated images at sample intervals
    if epoch % sample_interval == 0:
        print(f"Epoch {epoch}/{epochs}, D Loss: {d_loss[0]}, G Loss: {g_loss}")
        
        # Generate and save images
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_images = generator.predict(noise)
        
        for i, generated_image in enumerate(generated_images):
            save_path = f'generated_image_epoch_{epoch}_sample_{i}.jpg'
            cv2.imwrite(save_path, generated_image * 255)  # Ensure pixel values are in the [0, 255] range


# Training loop for both discriminator and generator
for epoch in range(epochs):
    # Train the discriminator on real and fake data here
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_images = X_train[idx]
    labels_real = np.ones((batch_size, 1))
    
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    generated_images = generator.predict(noise)
    labels_fake = np.zeros((batch_size, 1))

    d_loss_real = discriminator.train_on_batch(real_images, labels_real)
    d_loss_fake = discriminator.train_on_batch(generated_images, labels_fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train the generator here
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    labels_gan = np.ones((batch_size, 1))
    
    g_loss = gan.train_on_batch(noise, labels_gan)

    # Print progress and save generated images at sample intervals
    if epoch % sample_interval == 0:
        print(f"Epoch {epoch}/{epochs}, D Loss: {d_loss[0]}, G Loss: {g_loss}")
        
        # Generate and save images
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_images = generator.predict(noise)
        
        for i, generated_image in enumerate(generated_images):
            save_path = f'generated_image_epoch_{epoch}_sample_{i}.jpg'
            cv2.imwrite(save_path, generated_image * 255)  # Ensure pixel values are in the [0, 255] range

import matplotlib.pyplot as plt

# Initialize empty lists to store loss values
d_losses = []
g_losses = []

# Training loop
for epoch in range(epochs):
 

    # Append the loss values to the lists
    d_losses.append(d_loss[0])
    g_losses.append(g_loss)



# Plot the loss curves
plt.figure(figsize=(10, 5))
plt.plot(range(epochs), d_losses, label="Discriminator Loss")
plt.plot(range(epochs), g_losses, label="Generator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("GAN Loss Curves")
plt.show()


# Generate random noise vectors
noise = np.random.normal(0, 1, (num_samples, latent_dim))

# Generate images
generated_images = generator.predict(noise)
# Save the generator model
generator.save("generator_model.h5")

# Save the discriminator model
discriminator.save("discriminator_model.h5")

# Save the GAN model
gan.save("gan_model.h5")
