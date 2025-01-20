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
model = YOLO("/net/travail/rramesh/TRDP v2/runs/detect/train/weights/best.pt")

input_images_path =  "/net/travail/rramesh/TRDP v2/yolo_annot/test/images"
images = os.listdir(input_images_path)
# targetlayers = [model.model.model[22].dfl.conv]
targetlayers = [model.model.model[22].cv3[0][2]]

print(targetlayers)


for idx , framename in enumerate(images):
    frame = cv2.imread(os.path.join(input_images_path, framename))
    seqname_alter, ext = os.path.splitext(framename)
    # Convert the OpenCV image (BGR) to RGB

    # Perform object detection with the model on the tensor
    results = model(frame)
    annotated_frame = results[0].plot()

    # Specify the target layers (this part depends on your model's architecture)
    #targetlayers = [model.model.model[22].cv3[2][1].conv]
    # targetlayers = [model.model.model[-4]]

    #targetlayers = [model.model.model[22].dfl.conv]

    # Call your Feature Explanation Method (passing the tensor instead of raw image)
    fem = Feature_Explanation_method(model, targetlayers)

    # Now pass the tensor (img_tensor) instead of the raw image
    saliency = fem(frame)
    
    # Load and preprocess the input image (ensure it's of shape (224, 224, 3))
    input_image = np.array(frame)
    #input_image = cv2.resize(input_image, (224, 224))
    input_image = input_image.astype(np.float32) / 255.0  # Normalize the image to range [0, 1]
    # Overlay the heatmap on the input image
    visualization = show_cam_on_image(input_image, saliency, use_rgb=True)
    
    plt.figure(figsize=(14, 7))

    # Subplot 1 - visualization_gcam
    plt.subplot(1, 2, 1)
    plt.imshow(visualization)
    plt.title(f'Fem {idx+1}')
    plt.axis('off')

    # Subplot 2 - visualization_fem
    plt.subplot(1, 2, 2)
    plt.imshow(annotated_frame)
    plt.title(f'prediction {idx+1}')
    plt.axis('off')


    # output_image_path = "/net/travail/rramesh/TRDP v2/Femresults_indiv/"

    # # Check if the directory exists
    # if not os.path.exists(output_image_path):
    #     os.makedirs(output_image_path)  # Create the directory if it doesn't exist

    # # #save the subplot
    # # output_file = os.path.join(output_image_path, f"{seqname_alter}.png")  # Change the extension as needed
    # # plt.savefig(output_file)
    
    # cv2.imwrite(f'{output_image_path}{seqname_alter}.jpg', visualization - np.max(visualization))
    # print("Image saved!")

    