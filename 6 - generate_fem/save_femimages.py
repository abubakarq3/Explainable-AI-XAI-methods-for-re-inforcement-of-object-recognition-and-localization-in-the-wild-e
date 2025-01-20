import cv2
import os
import numpy as np
import argparse
from ultralytics import YOLO
from Fem import Feature_Explanation_method

def save_cam_on_image(img: np.ndarray, mask: np.ndarray, output_path: str, use_rgb: bool = False, colormap: int = cv2.COLORMAP_JET, image_weight: float = 0.5):
    """
    Save the CAM overlay on the image as a heatmap.

    Args:
        img: Base image in RGB or BGR format.
        mask: CAM mask.
        output_path: Path to save the resulting image.
        use_rgb: Use RGB or BGR heatmap.
        colormap: OpenCV colormap to use.
        image_weight: Blending weight for the image and CAM.
    """
    if np.isnan(mask).any() or np.isinf(mask).any():
        print("Mask contains NaN or inf values")

    # Normalize mask to [0, 1]
    mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
    heatmap = cv2.applyColorMap(np.uint8(255 * (1 - mask)), colormap)

    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise ValueError("The input image should be np.float32 in the range [0, 1]")

    if not (0 <= image_weight <= 1):
        raise ValueError("image_weight should be in the range [0, 1]")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    cam_output = np.uint8(255 * cam)
    cv2.imwrite(output_path, cam_output)

def main(args):
    # Load the YOLO model
    model = YOLO(args.model_path)

    input_images_path = args.input_images_path
    output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    images = os.listdir(input_images_path)
    targetlayers_s2 = [model.model.model[-4]]  # Target layer for saliency generation

    for idx, framename in enumerate(images):
        frame = cv2.imread(os.path.join(input_images_path, framename))
        seqname_alter, ext = os.path.splitext(framename)

        output_image_path = os.path.join(output_path, f"{seqname_alter}_s2.png")

        if os.path.exists(output_image_path):
            print(f"The image {output_image_path} already exists.")
        else:
            # Perform object detection
            results = model(frame)

            # Generate saliency map for target layer s2
            fem = Feature_Explanation_method(model, targetlayers_s2)
            saliency = fem(frame)

            # Normalize and save the saliency map overlaid on the input image
            input_image = frame.astype(np.float32) / 255.0
            save_cam_on_image(input_image, saliency, output_image_path, use_rgb=True, colormap=cv2.COLORMAP_JET)

            print(f"FEM results saved to {output_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save FEM images with CAM overlays.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the YOLO model file.")
    parser.add_argument("--input_images_path", type=str, required=True, help="Path to the folder containing input images.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save FEM results.")

    args = parser.parse_args()
    main(args)
