# Avatar Generation from CelebA Dataset

Create unique avatars using deep learning techniques with the CelebA dataset.

## Dataset
- Download the CelebA dataset from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset).

## Project Overview
This project aims to generate unique avatars using deep learning techniques. We use the CelebA dataset, a collection of celebrity face images, to train a Generative Adversarial Network (GAN) model. The GAN model learns to generate avatars that resemble real faces from the dataset.

## How it Works
1. **Data Collection**:
   - Download the CelebA dataset from the provided Kaggle link.

2. **Data Preprocessing**:
   - Extract the dataset and organize it into folders.
   - Resize and preprocess the images.

3. **Model Selection**:
   - Choose a GAN architecture (e.g., DCGAN).

4. **Model Training**:
   - Define the generator and discriminator models.
   - Train the GAN model on the CelebA dataset.

5. **Evaluation**:
   - Calculate Inception Score and FID to evaluate the quality of generated avatars.

6. **Avatar Generation**:
   - Use the trained GAN model to generate avatars from random noise vectors.

7. **Save and Showcase**:
   - Save the generated avatars as image files.
   - Create a user-friendly interface for users to generate avatars interactively.

## Usage
1. Download the CelebA dataset from the provided Kaggle link.

2. Preprocess the dataset using the provided code.

3. Train the GAN model with your preprocessed dataset.

4. After training, use the model to generate avatars from random noise vectors.

5. Showcase the generated avatars and allow users to create their own avatars.

## Evaluation
- You can evaluate the quality of generated avatars using Inception Score and FID. Refer to the code for implementation details.

## Results 
- You can see sample of generated avatars once you run the "generate avatar.py" file in your vscode.

## Future Enhancements

1. **Customization**: Enhance user experience with attribute customization options.
2. **Artistic Styles**: Add artistic style choices for unique avatars.
3. **Interactive Interface**: Develop an intuitive web interface for user convenience.
4. **Resolution Variety**: Expand avatar generation to offer different resolutions.
5. **Animation**: Explore the creation of animated avatars with expressive motions.
6. **Privacy and Security**: Ensure user data privacy and secure handling.

## Acknowledgments
- CelebA Dataset: [Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

## License
This project is licensed under the [MIT License](LICENSE.md).



