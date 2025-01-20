"""
Made by Rohin Andy Ramesh and abubakr qahir
"""

import argparse
from YOLOv8_Explainer import yolov8_heatmap, display_images

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate heatmaps using YOLOv8 and EigenCAM/EigenGradCAM.")

    parser.add_argument(
        "--weight",
        type=str,
        required=True,
        help="Path to the YOLOv8 model weights file (e.g., best.pt).",
    )

    parser.add_argument(
        "--img_path",
        type=str,
        required=True,
        help="Path to the directory containing images for heatmap generation.",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to the directory where the generated heatmaps will be saved.",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="EigenCAM",
        choices=["EigenCAM", "EigenGradCAM"],
        help="The method to use for generating heatmaps. Default is EigenCAM.",
    )

    args = parser.parse_args()

    # Load the model with the specified heatmap method
    model = yolov8_heatmap(weight=args.weight, method=args.method)

    # Generate heatmaps for the images in the specified directory
    imagelist = model(img_path=args.img_path, save_path=args.save_path)

    # Optionally display images (commented out by default for batch processing)
    # display_images(imagelist)

    print(f"Heatmaps generated and saved to: {args.save_path}")

if __name__ == "__main__":
    main()