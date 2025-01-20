# yolo_training/
# │
# ├── data/
# │   ├── train/
# │   │   ├── images/           # Training images
# │   │   └── labels/           # Training labels (annotations)
# │   │
# │   └── test/
# │       ├── images/           # Testing images
# │       └── labels/


import os
import shutil
from sklearn.model_selection import train_test_split

# Define directories
image_dir = "/net/travail/rramesh/TRDP v2/yolo_annot/images"  # Directory containing images
label_dir = "/net/travail/rramesh/TRDP v2/yolo_annot/labels"  # Directory containing labels
output_dir = "/net/travail/rramesh/TRDP v2/yolo_annot/"  # Output directory for train/test sets

# Proportion of the dataset to include in the test split
test_size = 0.2

# Create output directories
train_images_dir = os.path.join(output_dir, "train", "images")
train_labels_dir = os.path.join(output_dir, "train", "labels")
test_images_dir = os.path.join(output_dir, "test", "images")
test_labels_dir = os.path.join(output_dir, "test", "labels")

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)

# Group files by class based on filename prefixes
class_files = {}

for filename in os.listdir(image_dir):
    if filename.endswith((".jpg", ".png")):  # Check for image files
        class_name = filename.split("Place")[0]  # Extract the class name (prefix)
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt").replace(".png", ".txt"))
        
        if os.path.exists(label_path):  # Ensure corresponding label exists
            if class_name not in class_files:
                class_files[class_name] = []
            class_files[class_name].append((image_path, label_path))

# Split each class into train and test sets
for class_name, files in class_files.items():
    train_files, test_files = train_test_split(files, test_size=test_size, random_state=42)
    
    # Copy training files
    for image_path, label_path in train_files:
        shutil.copy(image_path, train_images_dir)
        shutil.copy(label_path, train_labels_dir)
    
    # Copy testing files
    for image_path, label_path in test_files:
        shutil.copy(image_path, test_images_dir)
        shutil.copy(label_path, test_labels_dir)

print("YOLO dataset split into training and testing sets with balanced class distribution successfully.")









































