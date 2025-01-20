"""
Utility functions for computing focal areas and foveation
"""

import os
import numpy as np
import math

import numpy as np

def compute_sigma_mm(z, alpha):
    """
    Computes the standard deviation (sigma) in millimeters based on the depth (z) and angle (alpha).

    Parameters:
    - z (float): Depth or distance.
    - alpha (float): Angle in degrees.

    Returns:
    - float: Standard deviation in millimeters.
    """
    alpha = np.deg2rad(alpha)
    return z * np.tan(alpha)


def compute_fov_mm(z, cam_fov_x, cam_fov_y):
    """
    Computes the field of view in millimeters based on depth (z) and camera's horizontal and vertical field of view.

    Parameters:
    - z (float): Depth or distance.
    - cam_fov_x (float): Camera's horizontal field of view in degrees.
    - cam_fov_y (float): Camera's vertical field of view in degrees.

    Returns:
    - tuple: Field of view in millimeters (fov_x_mm, fov_y_mm).
    """
    fov_x = np.deg2rad(cam_fov_x)
    fov_y = np.deg2rad(cam_fov_y)

    fov_x_mm = 2 * z * np.tan(fov_x / 2)
    fov_y_mm = 2 * z * np.tan(fov_y / 2)

    return fov_x_mm, fov_y_mm


def compute_sigma_pixels(z, alpha, width, height, cam_fov_x, cam_fov_y):
    """
    Computes the standard deviation in pixels based on depth (z), angle (alpha), image dimensions, and camera's field of view.

    Parameters:
    - z (float): Depth or distance.
    - alpha (float): Angle in degrees.
    - width (int): Image width in pixels.
    - height (int): Image height in pixels.
    - cam_fov_x (float): Camera's horizontal field of view in degrees.
    - cam_fov_y (float): Camera's vertical field of view in degrees.

    Returns:
    - tuple: Standard deviation in pixels (sigma_x, sigma_y).
    """
    fov_x_mm, fov_y_mm = compute_fov_mm(z, cam_fov_x, cam_fov_y)

    # print(f'fov_x_mm: {fov_x_mm}, fov_y_mm: {fov_y_mm}')

    sigma_x_mm = compute_sigma_mm(z, alpha)
    sigma_y_mm = compute_sigma_mm(z, alpha)

    pix_x = width / fov_x_mm
    pix_y = height / fov_y_mm

    # print(f'pix_x: {pix_x}, pix_y: {pix_y}')

    sigma_x = sigma_x_mm * pix_x
    sigma_y = sigma_y_mm * pix_y

    return sigma_x, sigma_y


def compute_saliency_map(image, fixation_point, sigma_x, sigma_y, A=1):
    """
    Computes the saliency map based on the input image, fixation point, and standard deviations in x and y.

    Parameters:
    - image (numpy.ndarray): Input image.
    - fixation_point (tuple): Coordinates of the fixation point (x, y).
    - sigma_x (float): Standard deviation in the x-direction.
    - sigma_y (float): Standard deviation in the y-direction.
    - A (float): Amplitude parameter (default=1).

    Returns:
    - numpy.ndarray: Saliency map.
    """
    h, w, _ = image.shape

    x_coords, y_coords = np.mgrid[0:w, 0:h]
    coords = np.vstack([x_coords.ravel(), y_coords.ravel()])

    saliency_map = A * np.exp(-((coords[0] - fixation_point[0])**2 / (2 * sigma_x**2)) - ((coords[1] - fixation_point[1])**2 / (2 * sigma_y**2)))
    saliency_map = saliency_map.reshape(w, h)

    return saliency_map
