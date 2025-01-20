import cv2
import os
from ultralytics import YOLO
import numpy as np
import torch

path = "/net/cremi/rramesh/espaces/travail/TRDP v2/Eigencam_M1_results"
print(len(os.listdir(path)))

# Load the YOLO model
# model = YOLO("/net/travail/rramesh/TRDP v2/runs/detect/train6/weights/best.pt")

# imagesdata = "/net/cremi/rramesh/espaces/travail/TRDP v2/yoloformat/test/images"

# save_path = "/net/travail/rramesh/TRDP v2/prediction_yolofromat"
# os.makedirs(save_path, exist_ok=True)

# # Loop through all images in the images directory
# for image_name in sorted(os.listdir(imagesdata)):
#     # Construct paths for the image and its corresponding label file
#     image_path = os.path.join(imagesdata, image_name)
    
#     frame =cv2.imread(image_path)
#     # Run YOLO inference on the frame
#     results = model(frame)

#     # Visualize the results on the frame
#     annotated_frame = results[0].plot()

#     output_path = os.path.join(save_path,image_name) # Specify the desired output file name and path
#     cv2.imwrite(output_path, annotated_frame)