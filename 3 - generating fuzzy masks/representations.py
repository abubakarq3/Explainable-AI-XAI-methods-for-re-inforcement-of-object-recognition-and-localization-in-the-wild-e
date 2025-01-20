from PIL import Image, ImageChops
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from utils import normalise


def represent_heatmap(saliency, cmap: str = 'gray'):
    """"
    Inputs:
    Saliency map as image (either unsigned or signed),
    Colormap to be used

    Ouput:
    Heatmapped saliency map as RGBImage
    """
    # if np.max(saliency) > 1:
    saliency = normalise(saliency)

    # Setting colormap to grey if there is no input colour map and saliency is unsigned (0-1)
    if cmap == 'gray' and np.min(saliency) >= 0:
        cmap = 'gray'

    # if saliency type i.e np.min(saliency) < 0, then error handle and have user specify colormap

    colormap = plt.get_cmap(cmap)
    heatmapped_saliency = (colormap(saliency) * 255).astype(np.uint8)

    heatmapped_saliency_image = Image.fromarray(heatmapped_saliency)
    # Heatmapped_saliency_image = RGBImage(heatmapped_saliency_image)
    return heatmapped_saliency_image


def represent_heatmap_overlaid(saliency, image, cmap: str):
    """
    Inputs:
    Saliency map as image (either unsigned or signed),
    RGB Image,
    Colormap

    Output:
    RGBImage overlaid with saliency map
    """
    # print(saliency.size)
    # saliency = saliency.T
    # Calling represent_heatmap to generate heatmapped saliency using specified
    heatmapped_saliency = represent_heatmap(saliency, cmap)

    # print(heatmapped_saliency.size)
    # print(image.size)

    # Blending Input Image with saliency heatmap image
    blended_image = Image.blend(image.convert(
        'RGBA'), heatmapped_saliency.convert('RGBA'),  alpha=0.5)

    # Blended_image = RGBImage(blended_image)
    return blended_image


def represent_isolines(saliency,  cmap: str):
    """
    Inputs:
    Saliency map as image (either unsigned or signed),
    Colormap to be used

    Ouput:
    Heatmapped Isolines as RGBImage
    """
    # Specifying range of neighboring pixels to search for contiguous pixels
    contour_levels = np.linspace(0, 1, 11)
    level_list = []  # list having contours / rings for each specified range / level

    w, h = saliency.shape
    isoline = np.zeros((w, h))

    saliency = normalise(saliency)  # normalizing saliency map

    # To find contours / rings in each range and add them to a list
    # where each addition is a numpy array having the various contours / rings found,
    # stored as tuples of the x and y coordinates
    for level in contour_levels:
        contours_list = measure.find_contours(saliency, level)
        level_list.append(contours_list)

    # Representing the the pixel values for each position in the isoline object by its
    # value at the same position in the saliency map
    for level in level_list:
        for contour in level:
            for x, y in contour:
                isoline[int(x), int(y)] = saliency[int(x), int(y)]

    # Applying a heatmap to the obtained isoline
    isoline_heatmapped = represent_heatmap(isoline, cmap)

    return isoline_heatmapped


def represent_isolines_superimposed(saliency, image, cmap: str):
    """
    Inputs:
    Saliency map as image (either unsigned or signed),
    RGB Image,
    Colormap to be used

    Output:
    Heatmapped Isolines overlaid on Input Image as RGBImage
    """
    saliency = saliency.T
    # To get heatmapped Isolines using the saliency map and colormap
    isolines_heatmapped = represent_isolines(saliency,  cmap)

    # Blending Input Image with saliency heatmap image
    blended_image = Image.blend(image.convert(
        'RGBA'), isolines_heatmapped.convert('RGBA'),  alpha=0.5)

    # Blended_image = RGBImage(blended_image)
    return blended_image

    # Blended_image = RGBImage(blended_image)
    # return Blended_image


# def represent_hard_selection(saliency, image, threshold: int):
#     """
#     Inputs:
#     Saliency map as image (either unsigned or signed),
#     RGB Image,
#     threshold value as integer

#     Output:
#     Input image overlaid by Binary mask (obtained by thresholding values in saliency map less than specified threhold value)
#     """

#     w, h = saliency.shape

#     # Generating Image of zeros
#     blank = np.zeros((w, h))
#     blank = Image.fromarray(blank)
#     blank = blank.convert('RGBA')

#     # Obtaining binary mask as Image using threshold value
#     mask = (saliency >= threshold).astype(np.uint8)
#     mask = Image.fromarray(mask * 255)

#     # stacking the images on each other
#     hard_image = Image.composite(image, blank, mask)

#     Hard_image = RGBImage(hard_image)
#     return Hard_image.image


# def represent_soft_selection(saliency: Saliency, image: RGBImage) -> RGBImage:
#     """
#     Inputs:
#     Saliency map as image (either unsigned or signed),
#     RGB Image,

#     Output:
#     Input image masked using saliency map 
#     """

#     saliency = Image.fromarray(saliency)
#     saliency = saliency.convert('RGB')
#     image = image.convert('RGB')
#     soft_image = ImageChops.multiply(image, saliency)

#     Soft_image = RGBImage(soft_image)
#     return Soft_image.image
