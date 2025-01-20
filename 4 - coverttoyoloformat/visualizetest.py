import cv2
import os

# Function to read YOLO annotation and convert to bounding box coordinates
def read_yolo_annotations(label_path, img_width, img_height):
    bboxes = []
    with open(label_path, 'r') as file:
        for line in file.readlines():
            parts = line.strip().split()
            class_id, x_center, y_center, width, height = map(float, parts)
            # Convert YOLO format to top-left corner and width/height
            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)
            bboxes.append((class_id, x1, y1, x2, y2))
    return bboxes

# Paths
imagesdata = "/net/cremi/rramesh/espaces/travail/TRDP v2/yoloformat/test/images"
annotationdata = "/net/cremi/rramesh/espaces/travail/TRDP v2/yoloformat/test/labels"

save_path = "/net/cremi/rramesh/espaces/travail/TRDP v2/groundtruth_yoloformat"
os.makedirs(save_path , exist_ok=True)

# Loop through all images in the images directory
for image_name in sorted(os.listdir(imagesdata)):
    # Construct paths for the image and its corresponding label file
    image_path = os.path.join(imagesdata, image_name)
    label_path = os.path.join(annotationdata, os.path.splitext(image_name)[0] + ".txt")
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        continue

    img_height, img_width = image.shape[:2]

    # Check if annotation file exists
    if not os.path.exists(label_path):
        print(f"No annotation file for {image_name}. Skipping visualization.")
        continue

    # Read annotations and draw bounding boxes
    bboxes = read_yolo_annotations(label_path, img_width, img_height)
    for class_id, x1, y1, x2, y2 in bboxes:
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw class ID text
        cv2.putText(image, f"Class {int(class_id)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # # Display the image with bounding boxes
    # cv2.imshow("YOLO Bounding Box Visualization", image)
    output_path = os.path.join(save_path,image_name)  # Specify the desired output file name and path
    cv2.imwrite(output_path, image)
    
#     # Wait for a key press to move to the next image or close the window
#     key = cv2.waitKey(0)
#     if key == 27:  # Press 'Esc' to exit
#         break

# # Close all OpenCV windows
# cv2.destroyAllWindows()

# testimages = os.listdir("/net/travail/rramesh/TRDP v2/yolo_annot/test/images")
# trainimages = os.listdir("/net/travail/rramesh/TRDP v2/yolo_annot/train/images")

# classdist = {}

# # Dictionary to store class countsyoloformat
# for filename in trainimages:
#     class_name = filename.split("Place")[0]
#     if class_name not in classdist:
#         classdist[class_name] = {"train": 0, "test": 0}
#     classdist[class_name]["train"] += 1

# # Count class occurrences in the testing set
# for filename in testimages:
#     class_name = filename.split("Place")[0]
#     if class_name not in classdist:
#         classdist[class_name] = {"train": 0, "test": 0}
#     classdist[class_name]["test"] += 1

# # Perform length test for 80-20 distribution
# for class_name, counts in classdist.items():
#     train_count = counts["train"]
#     test_count = counts["test"]
#     total_count = train_count + test_count
#     train_ratio = train_count / total_count
#     test_ratio = test_count / total_count

#     print(f"Class: {class_name}")
#     print(f"  Train: {train_count} ({train_ratio:.2%})")
#     print(f"  Test: {test_count} ({test_ratio:.2%})")

#     # Check if distribution is approximately 80-20
#     if abs(train_ratio - 0.8) > 0.05 or abs(test_ratio - 0.2) > 0.05:
#         print(f"  WARNING: Class '{class_name}' does not meet the 80-20 split requirement!")
