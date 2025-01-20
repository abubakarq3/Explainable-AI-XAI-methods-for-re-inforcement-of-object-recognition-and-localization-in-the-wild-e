import cv2
import os
import numpy as np
from utils import normalize_bbox

def read_yolo_annotations(label_path, img_width, img_height):
    bboxes = []
    bboxes_nn = []
    with open(label_path, 'r') as file:
        for line in file.readlines():
            parts = line.strip().split()
            class_id, x_center, y_center, width, height = map(float, parts)

            std_width = np.std(width)
            std_height = np.std(height)
            
            alpha = 0.1
            print(f"the standard deviation of width is {std_width}")
            print(f"The standard deviation of height is {std_height}")

            xcent = x_center * img_width
            ycent = y_center * img_height 
            extended_width = (width + alpha * std_width) * img_width
            extended_height = (height + alpha * std_height) * img_height
             
            x1_nn = int((xcent - extended_width / 2))
            y1_nn = int((ycent - extended_height / 2))
            x2_nn = int((xcent + extended_width / 2))
            y2_nn = int((ycent + extended_height / 2))
            bboxes_nn.append((x1_nn, y1_nn, x2_nn, y2_nn))

            print(img_height,img_width)
            print(x_center, y_center,width, height)
            print(x_center*img_width, y_center*img_height,width*img_width, height*img_height)
            # Convert YOLO format to top-left corner and width/height
            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)
            bboxes.append((class_id, x1, y1, x2, y2))
    return bboxes,bboxes_nn

# Paths to the image and label files
image_path = "/net/travail/rramesh/TRDP v2/yoloformat/test/images/CanOfCocaColaPlace2Subject1Frame_4600.jpg"
label_path = "/net/travail/rramesh/TRDP v2/yoloformat/test/labels/CanOfCocaColaPlace2Subject1Frame_4600.txt"

# Read the image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")
img_height, img_width = image.shape[:2]

# Read bounding boxes
boxes, boxes_nn = read_yolo_annotations(label_path, img_width, img_height)

print(f"the boxes are {boxes}")
print(f"the boxe_nn are {boxes_nn}")

# Draw bounding boxes on the image
for box in boxes:
    _, x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green boxes

for x1_nn, y1_nn, x2_nn, y2_nn in boxes_nn:
    cv2.rectangle(image, (x1_nn, y1_nn), (x2_nn, y2_nn), (0, 0, 255), 2)  # Red extended boxes

# Display the image with bounding boxes
cv2.imshow("Image with Bounding Boxes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()