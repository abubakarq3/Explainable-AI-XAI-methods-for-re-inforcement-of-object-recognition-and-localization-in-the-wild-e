import cv2
import os
from ultralytics import YOLO
import argparse

def run_yolo_inference(model_path, imagesdata, save_path):
    """
    Perform inference using a YOLO model on images in the specified directory
    and save the annotated results.

    Args:
        model_path (str): Path to the YOLO model weights file.
        imagesdata (str): Directory containing input images.
        save_path (str): Directory to save annotated images.
    """
    # Load the YOLO model
    model = YOLO(model_path)

    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Loop through all images in the images directory
    for image_name in sorted(os.listdir(imagesdata)):
        # Construct the image path
        image_path = os.path.join(imagesdata, image_name)

        # Read the image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Could not read image: {image_path}")
            continue

        # Run YOLO inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Save the annotated frame
        output_path = os.path.join(save_path, image_name)
        cv2.imwrite(output_path, annotated_frame)

        print(f"Annotated image saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run YOLO inference on a set of images.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the YOLO model weights file (e.g., 'best.pt').")
    parser.add_argument("--imagesdata", type=str, required=True, help="Path to the directory containing input images.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the annotated images.")
    args = parser.parse_args()

    run_yolo_inference(
        model_path=args.model_path,
        imagesdata=args.imagesdata,
        save_path=args.save_path
    )

if __name__ == "__main__":
    main()
