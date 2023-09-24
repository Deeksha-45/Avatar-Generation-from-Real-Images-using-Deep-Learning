import os
import cv2
import numpy as np
from tqdm import tqdm

# Define paths
data_dir = r"E:\face dataset\img_align_celeba\img_align_celeba"  # Update with your dataset path
output_dir = 'preprocessed_celeba'  # Create a new directory for preprocessed data

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load CelebA attribute file to filter faces
attr_file = os.path.join(data_dir, "E:\face dataset\list_attr_celeba.csv")

# Function to check if an image has desired attributes
def has_desired_attributes(attr_values, attr_names, desired_attributes):
    return all(attr_values[attr_names.index(attr)] == '1' for attr in desired_attributes)

# Read attribute names from the attribute file
with open(attr_file, 'r') as f:
    attr_names = f.readline().strip().split(',')[1:]

# Desired attributes with proper capitalization and underscores
desired_attributes = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
    'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
    'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
    'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
    'Wearing_Necktie', 'Young'
]

# Loop through the dataset and preprocess images
for filename in tqdm(os.listdir(data_dir)):
    if filename.endswith('.jpg'):
        img_path = os.path.join(data_dir, filename)
        img = cv2.imread(img_path)
        
        # Perform face detection (you may need to adjust the parameters)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        # Check if the image has desired attributes
        with open(attr_file, 'r') as f:
            attr_line = f.readline()
            while not attr_line.startswith(filename):
                attr_line = f.readline()
            attr_values = attr_line.strip().split(',')[1:]
            if not has_desired_attributes(attr_values, attr_names, desired_attributes):
                continue
        
        # Save detected faces
        for i, (x, y, w, h) in enumerate(faces):
            face = img[y:y+h, x:x+w]
            face_filename = os.path.join(output_dir, f'{filename.split(".")[0]}_face_{i}.jpg')
            cv2.imwrite(face_filename, face)


