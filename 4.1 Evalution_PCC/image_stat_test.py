import numpy as np
from PIL import Image

def get_image_stats(image_path):
    """
    Analyze image properties: shape, minimum, and maximum intensity values.

    Args:
        image_path (str): Path to the image file.

    Returns:
        dict: A dictionary with image statistics.
    """
    image = np.array(Image.open(image_path))

    # Collect image statistics
    image_stats = {
        "Filename": image_path,
        "Shape": image.shape,
        "Min Intensity": image.min(),
        "Max Intensity": image.max()
    }
    return image_stats

if __name__ == "__main__":
    image_path1 = "/net/travail/rramesh/TRDP v2/saliencyinYoloimage/CanOfCocaColaPlace2Subject1_Frame_8080.png"  # Replace with the path to the first image
    image_path2 = "/net/travail/rramesh/TRDP v2/Eigengradcam_M1_results/CanOfCocaColaPlace2Subject1Frame_8080.jpg"  # Replace with the path to the second image

    stats1 = get_image_stats(image_path1)
    stats2 = get_image_stats(image_path2)

    print("Image Statistics:")
    print(stats1)
    print(stats2)
