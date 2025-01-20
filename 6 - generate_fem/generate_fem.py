"""
This script generates Fem Visualization for YOLOv8 model
Made by Rohin Andy Ramesh on 05/10/24
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from ultralytics import YOLO
from Fem import Feature_Explanation_method

def show_cam_on_image(img: np.ndarray, mask: np.ndarray, use_rgb: bool = False, colormap: int = cv2.COLORMAP_JET, image_weight: float = 0.5) -> np.ndarray:
    """ Overlay the CAM mask on the image as a heatmap.

    Args:
        img: Base image in RGB or BGR format.
        mask: The CAM mask.
        use_rgb: Use RGB or BGR heatmap.
        colormap: OpenCV colormap.
        image_weight: Weight for blending the image and mask.

    Returns:
        The image with the CAM overlay.
    """
    if np.isnan(mask).any() or np.isinf(mask).any():
        print("Mask contains NaN or inf values")

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise ValueError("The input image should be np.float32 in the range [0, 1].")

    if not (0 <= image_weight <= 1):
        raise ValueError(f"image_weight should be in the range [0, 1]. Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def main(args):
    # Load the YOLO model
    model = YOLO(args.model_path)

    input_images_path = args.input_images_path
    images = os.listdir(input_images_path)

    # Define target layers
    targetlayers_s1 = [model.model.model[22].cv3[0][2]]
    targetlayers_s2 = [model.model.model[22].cv3[1][2]]
    targetlayers_s3 = [model.model.model[-4]]

    for idx, framename in enumerate(images):
        frame = cv2.imread(os.path.join(input_images_path, framename))
        seqname_alter, ext = os.path.splitext(framename)

        # Perform object detection
        results = model(frame)
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Define the target layers for each stage
        target_layers = [targetlayers_s1, targetlayers_s2, targetlayers_s3]
        visualizations = []

        # Generate saliency maps for each target layer set (s1, s2, s3)
        for i, target_layer in enumerate(target_layers):
            fem = Feature_Explanation_method(model, target_layer)
            saliency = fem(frame)

            # Normalize and overlay the saliency map on the image
            input_image = frame.astype(np.float32) / 255.0
            visualization = show_cam_on_image(input_image, saliency, use_rgb=True)
            visualizations.append(visualization)

        # Plot all 4 subplots: the 3 CAM visualizations and the annotated frame
        plt.figure(figsize=(16, 8))
        plt.suptitle(f'FRAME NAME: {seqname_alter}', fontsize=16, fontweight='bold')

        # Subplot for CAM from targetlayers_s1
        plt.subplot(1, 4, 1)
        plt.imshow(visualizations[0])
        plt.title('Scale 80')
        plt.axis('off')

        # Subplot for CAM from targetlayers_s2
        plt.subplot(1, 4, 2)
        plt.imshow(visualizations[1])
        plt.title('Scale 40')
        plt.axis('off')

        # Subplot for CAM from targetlayers_s3
        plt.subplot(1, 4, 3)
        plt.imshow(visualizations[2])
        plt.title('Scale 20')
        plt.axis('off')

        # Subplot for the annotated frame
        plt.subplot(1, 4, 4)
        plt.imshow(annotated_frame)
        plt.title('Annotated Frame')
        plt.axis('off')

        # Show the figure
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize CAM overlays on images using YOLO.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the YOLO model file.")
    parser.add_argument("--input_images_path", type=str, required=True, help="Path to the folder containing input images.")

    args = parser.parse_args()
    main(args)