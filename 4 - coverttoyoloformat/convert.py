"""
This script generates Converts GITW data to yolo format
Made by Rohin Andy Ramesh and abubakr qahir on 17/04/24
"""

import os
import cv2
import argparse
from utils import normalize_bbox

def convert_to_yolo_format(objectpath, imagesdata, annotationdata, class_id=3):
    """
    Converts dataset from the specified objectpath to YOLO format and saves 
    images and labels in the respective directories.

    Args:
        objectpath (str): Path to the dataset containing object sequences.
        imagesdata (str): Path to save the processed images.
        annotationdata (str): Path to save the YOLO format annotations.
        class_id (int): Class ID for YOLO format (default is 3).
    """
    # Ensure output directories exist
    os.makedirs(imagesdata, exist_ok=True)
    os.makedirs(annotationdata, exist_ok=True)

    for seqname in sorted(os.listdir(objectpath)):
        # Paths to bounding box text and images
        bboxpath = os.path.join(objectpath, seqname, "bounding_box_finetune.txt")
        imagespath = os.path.join(objectpath, seqname, "Frames")

        # Read each line in the bounding box file
        with open(bboxpath, 'r') as file:
            lines = file.readlines()

        for line in lines:
            parts = line.split()
            frame_name = parts[0].replace(".png", ".jpg")
            bbox = list(map(int, parts[1:]))

            # Load the corresponding image
            image_path = os.path.join(imagespath, frame_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image: {image_path}")
                continue

            # Normalize bounding box
            img_height, img_width = image.shape[:2]
            x, y, w, h = bbox
            x_center, y_center, norm_w, norm_h = normalize_bbox(x, y, w, h, img_width, img_height)

            # Save the image in the images directory
            save_image_path = os.path.join(imagesdata, seqname + frame_name)
            cv2.imwrite(save_image_path, image)

            # Write the bounding box to a text file in YOLO format
            label_file_name = seqname + os.path.splitext(frame_name)[0] + ".txt"
            save_label_path = os.path.join(annotationdata, label_file_name)

            with open(save_label_path, 'w') as label_file:
                # YOLO format: <class_id> <x_center> <y_center> <width> <height>
                label_file.write(f"{class_id} {x_center} {y_center} {norm_w} {norm_h}\n")

def main():
    parser = argparse.ArgumentParser(description="Convert dataset to YOLO format.")
    parser.add_argument("--objectpath", type=str, required=True, help="Path to the dataset containing object sequences.")
    parser.add_argument("--imagesdata", type=str, required=True, help="Path to save the processed images.")
    parser.add_argument("--annotationdata", type=str, required=True, help="Path to save the YOLO format annotations.")
    parser.add_argument("--class_id", type=int, default=3, help="Class ID for YOLO format (default: 3).")
    args = parser.parse_args()

    convert_to_yolo_format(args.objectpath, args.imagesdata, args.annotationdata, args.class_id)

if __name__ == "__main__":
    main()
