import cv2
import os
import numpy as np
from utils import normalize_bbox

def read_yolo_annotations(label_path, img_width, img_height):
    bboxes = []
    with open(label_path, 'r') as file:
        for line in file.readlines():
            parts = line.strip().split()
            class_id, x_center, y_center, width, height = map(float, parts)
            
            print(img_height,img_width)
            print(x_center, y_center,width, height)
            print(x_center*img_width, y_center*img_height,width*img_width, height*img_height)
            # Convert YOLO format to top-left corner and width/height
            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)
            bboxes.append((class_id, x1, y1, x2, y2))
    return bboxes


saveimagesdata = "/net/travail/rramesh/TRDP v2/yoloformat_cropped_v2/test/images"
saveannotationdata = "/net/travail/rramesh/TRDP v2/yoloformat_cropped_v2/test/Labels"

os.makedirs(saveimagesdata, exist_ok=True)
os.makedirs(saveannotationdata, exist_ok=True)



# Paths
imagesdata = "/net/travail/rramesh/TRDP v2/yoloformat/test/images"
annotationdata = "/net/travail/rramesh/TRDP v2/yoloformat/test/labels"

seq_images_name = os.listdir(imagesdata)
  
for seq_name in seq_images_name:

    image = cv2.imread(os.path.join(imagesdata,seq_name))
             
    label_path = os.path.join(annotationdata, os.path.splitext(seq_name)[0] + ".txt")
    print(label_path)
    img_height, img_width = image.shape[:2]

    boxes = read_yolo_annotations(label_path, img_width, img_height)
    print(boxes)
    x_center = abs(int((boxes[0][3] + boxes[0][1])/2))
    y_center = abs(int((boxes[0][2] + boxes[0][4])/2 ))
    mean_width = np.abs(np.mean(boxes[0][3] - boxes[0][1]))
    mean_height = np.abs(np.mean(boxes[0][2] - boxes[0][4]))
    std_width = np.std(boxes[0][3] + boxes[0][1]) + 40
    std_height = np.std(boxes[0][2] + boxes[0][4]) + 40
    print(f"The mean width is {boxes[0][3] - boxes[0][1]} and the {mean_width}")
    print(f"The xcenter is {x_center} and the ycenter {y_center}")

    alpha = 3
    print(f"The mean is {mean_width} and {mean_height}")
    print(f"The standard deviation of the{std_width} and {std_height}")


    # Extend dimensions
    extended_width = mean_width + alpha * std_width
    extended_height = mean_height + alpha * std_height

    print(f"The extendednt hwidth and height is {extended_height} and  {extended_width}")
    x1 = int((x_center - extended_width / 2))
    y1 = int((y_center - extended_height / 2))
    x2 = int((x_center + extended_width / 2))
    y2 = int((y_center + extended_height / 2))



    cropped_image = image[y1:y2, x1:x2]
    crop_width , crop_height ,_= cropped_image.shape
    print(f"The shape of the cropped image is {cropped_image.shape}")
    for class_id, x1_original , y1_original , x2_original , y2_original in boxes:
        new_x1 = max(0, x1_original - x1)
        new_y1 = max(0, y1_original - y1)
        new_x2 = min(x2 - x1, x2_original - x1)
        new_y2 = min(y2 - y1, y2_original - y1)

        # If the bounding box is completely outside the crop, set it to (0, 0, 0, 0)
        if new_x2 <= new_x1 or new_y2 <= new_y1:
            new_x1, new_y1, new_x2, new_y2 = 0, 0, 0, 0
        
        if new_x1 != 0 and crop_width !=0 and crop_height!=0:
            x_center, y_center, norm_w, norm_h = normalize_bbox(new_x1, new_y1,(new_x2 - new_x1) , (new_y2-new_y1), crop_width, crop_height)
        
            # Save the image in the images directory
            save_image_path = os.path.join(saveimagesdata, seq_name)
            cv2.imwrite(save_image_path, cropped_image)

            # Write the bounding box to a text file in YOLO format
            label_file_name = saveannotationdata+"/" + os.path.splitext(seq_name)[0] + ".txt"
            save_label_path = os.path.join(annotationdata, label_file_name)

            with open(save_label_path, 'w') as label_file:
                # YOLO format: <class_id> <x_center> <y_center> <width> <height>
                # Modify if you have multiple classes
                label_file.write(f"{int(class_id)} {x_center} {y_center} {norm_w} {norm_h}\n")

        else:
            print("The bounding box dont exist")

    #     # Draw rectangle
    #     cv2.rectangle(cropped_image, (new_x1, new_y1), (new_x2, new_y2), (0, 255, 0), 2)    # Draw class ID text
    #     cv2.putText(cropped_image, f"Class {int(class_id)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # # Display the image with bounding boxes
    # if len(cropped_image) > 0 :
    #     cv2.imshow("YOLO Bounding Box Visualization", cropped_image)
    #     # Wait for a key press to move to the next image or close the window
    #     cv2.waitKey(0)

    # else:
    #     print("I think the bounding box is not detcted Pleas echeck if the image is cropped or not")
    
