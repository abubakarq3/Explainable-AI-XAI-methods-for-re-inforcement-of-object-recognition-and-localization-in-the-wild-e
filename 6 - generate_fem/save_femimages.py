import cv2
import torch
import numpy as np
import os
from Fem import Feature_Explanation_method
from ultralytics import YOLO

def save_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      output_path: str,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5):
    """
    Save the cam overlay on the image as a heatmap.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param output_path: Path to save the resulting image.
    :param use_rgb: Whether to use an RGB or BGR heatmap.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    """
    if np.isnan(mask).any() or np.isinf(mask).any():
        print("mask contains NaN or inf values")

    # Reverse the colormap scaling to ensure red for high intensity
    mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))  # Normalize to [0, 1]
    heatmap = cv2.applyColorMap(np.uint8(255 * (1 - mask)), colormap)

    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise ValueError("The input image should be np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise ValueError("image_weight should be in the range [0, 1]")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    cam_output = np.uint8(255 * cam)
    cv2.imwrite(output_path, cam_output)

# Load the YOLO model
model = YOLO("/net/travail/rramesh/TRDP v2/runs/detect/train6/weights/best.pt")
input_images_path = "/net/travail/rramesh/TRDP v2/yoloformat/test/images"
output_path = "/net/travail/rramesh/TRDP v2/FEM_M1_s3_results/"  # Specify the path to save FEM results

if not os.path.exists(output_path):
    os.makedirs(output_path)

images = os.listdir(input_images_path)
#targetlayers_s2 = [model.model.model[22].cv3[1][2]]  # Target layer for s2
targetlayers_s2 = [model.model.model[-4]]

for idx, framename in enumerate(images):
    frame = cv2.imread(os.path.join(input_images_path, framename))
    seqname_alter, ext = os.path.splitext(framename)

    output_image_path = os.path.join(output_path, f"{seqname_alter}_s2.png")

    if os.path.exists(output_image_path):
        print("the image already exist")
    
    else:
        # Perform object detection
        results = model(frame)

        # Generate saliency map for target layer s2
        fem = Feature_Explanation_method(model, targetlayers_s2)
        saliency = fem(frame)

        # Normalize and save the saliency map overlaid on the input image
        input_image = frame.astype(np.float32) / 255.0
        
        save_cam_on_image(input_image, saliency, output_image_path, use_rgb=True, colormap=cv2.COLORMAP_JET)

        print(f"FEM results saved to {output_path}")
