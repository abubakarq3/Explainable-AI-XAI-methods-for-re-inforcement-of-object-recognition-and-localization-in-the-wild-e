import cv2
import torch
import torchvision.transforms as transforms
from Fem import Feature_Explanation_method
from ultralytics import YOLO
import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
    #mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0) 
    if np.isnan(mask).any() or np.isinf(mask).any():
        print("mask contains NaN or inf values")
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    print(f"heatmap size is {heatmap.shape}")
    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

# Load the YOLO model
model = YOLO("/net/travail/rramesh/TRDP v2/runs/detect/train6/weights/best.pt")

input_images_path =  "/net/travail/rramesh/TRDP v2/yoloformat/test/images"
images = os.listdir(input_images_path)
# targetlayers = [model.model.model[22].dfl.conv]
targetlayers_s1 = [model.model.model[22].cv3[0][2]]
targetlayers_s2 = [model.model.model[22].cv3[1][2]]
targetlayers_s3 = [model.model.model[-4]]



for idx, framename in enumerate(images):
    frame = cv2.imread(os.path.join(input_images_path, framename))
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #frame = cv2.resize(frame, (640, 640))
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
    plt.title('scale 80')
    plt.axis('off')

    # Subplot for CAM from targetlayers_s2
    plt.subplot(1, 4, 2)
    plt.imshow(visualizations[1])
    plt.title('scale 40')
    plt.axis('off')

    # Subplot for CAM from targetlayers_s3
    plt.subplot(1, 4, 3)
    plt.imshow(visualizations[2])
    plt.title('scale 20')
    plt.axis('off')

    # Subplot for the annotated frame
    plt.subplot(1, 4, 4)
    plt.imshow(annotated_frame)
    plt.title('Annotated Frame')
    plt.axis('off')

    # Show the figure
    plt.tight_layout()
    plt.show()

    