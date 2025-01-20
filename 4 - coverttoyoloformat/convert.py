import cv2
import os
from utils import normalize_bbox

objectpath = "/net/travail/rramesh/TRDP v2/Dataset/MilkBottle"
class_id = 3

imagesdata = "/net/travail/rramesh/TRDP v2/yoloformat/images"
annotationdata = "/net/travail/rramesh/TRDP v2/yoloformat/Labels"

os.makedirs(imagesdata, exist_ok=True)
os.makedirs(annotationdata, exist_ok=True)

for seqname in sorted(os.listdir(objectpath)):
    #To read the bounding box text and the images
    bboxpath = os.path.join(objectpath,seqname,"bounding_box_finetune.txt") 
    imagespath = os.path.join(objectpath,seqname,"Frames")
    
    #Read each line in the file
    with open(bboxpath, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.split()
        frame_name = parts[0]
        frame_name = frame_name.replace(".png", ".jpg")
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

        print(save_label_path)

        print(save_label_path)
        with open(save_label_path, 'w') as label_file:
            # YOLO format: <class_id> <x_center> <y_center> <width> <height>
              # Modify if you have multiple classes
            label_file.write(f"{class_id} {x_center} {y_center} {norm_w} {norm_h}\n")

            