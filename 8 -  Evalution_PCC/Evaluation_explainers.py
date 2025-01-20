"""
This script generates Evaluation for Evaluation and Explainers
Made by Rohin Andy Ramesh  and abubakr qahir on 01/01/25
"""
import os
import numpy as np
from PIL import Image
from scipy.stats import pearsonr
from tqdm import tqdm
import csv
import argparse

def preprocess_image(image, target_shape):
    """
    Resize and normalize image to range [0, 1].

    Args:
        image (numpy.ndarray): Input image.
        target_shape (tuple): Target shape (height, width).

    Returns:
        numpy.ndarray: Preprocessed image.
    """
    image = Image.fromarray(image)
    image = image.resize(target_shape, Image.ANTIALIAS)
    image = np.array(image) / 255.0
    return image

def compute_pcc(image1, image2):
    """
    Compute Pearson Correlation Coefficient (PCC) between two images.
    """
    # Flatten the images to 1D arrays
    img1_flat = image1.flatten()
    img2_flat = image2.flatten()

    # Compute PCC
    pcc, _ = pearsonr(img1_flat, img2_flat)
    return pcc

def load_images_from_folder(folder, target_shape):
    """
    Load images from a folder and return a dictionary with filenames as keys and preprocessed images as values.
    """
    images = {}
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(folder, filename)
            image = np.array(Image.open(filepath).convert('L'))  # Convert to grayscale
            image = preprocess_image(image, target_shape)
            images[filename] = image
    return images

def evaluate_pcc(saliency_folder, explainer_folder, target_shape=(384, 640)):
    """
    Evaluate PCC for all matching images between two folders.
    """
    print("Loading saliency images...")
    saliency_images = load_images_from_folder(saliency_folder, target_shape)
    print("Loading explainer images...")
    explainer_images = load_images_from_folder(explainer_folder, target_shape)

    # Compute PCC for matching filenames
    results = []
    for filename in tqdm(saliency_images, desc="Computing PCC"):
        exp_filename = filename.replace("_Frame", "Frame")
        exp_filename = exp_filename.replace(".png", ".jpg")

        if exp_filename in explainer_images:
            saliency_image = saliency_images[filename]
            explainer_image = explainer_images[exp_filename]
            pcc = compute_pcc(saliency_image, explainer_image)
            results.append({"Filename": filename, "PCC": pcc})

    # Calculate mean PCC
    mean_pcc = np.mean([result["PCC"] for result in results])
    results.append({"Filename": "Mean", "PCC": mean_pcc})

    return results

def save_results_to_csv(results, output_path):
    """
    Save the PCC results to a CSV file.
    """
    with open(output_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["Filename", "PCC"])
        writer.writeheader()
        for result in results:
            writer.writerow(result)

def main():
    parser = argparse.ArgumentParser(description="Compute PCC between saliency and explainer images.")
    parser.add_argument("--saliency_folder", type=str, required=True, help="Path to the saliency maps folder.")
    parser.add_argument("--explainer_folder", type=str, required=True, help="Path to the explainer maps folder.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the results CSV file.")
    parser.add_argument("--target_shape", type=int, nargs=2, default=(384, 640), help="Target shape for resizing images (height, width).")

    args = parser.parse_args()

    saliency_folder = args.saliency_folder
    explainer_folder = args.explainer_folder
    output_csv = args.output_csv
    target_shape = tuple(args.target_shape)

    pcc_results = evaluate_pcc(saliency_folder, explainer_folder, target_shape)

    print("PCC Results:")
    for result in pcc_results:
        print(f"{result['Filename']}: {result['PCC']:.4f}")

    save_results_to_csv(pcc_results, output_csv)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    main()
