"""
Utilities for the Fuzzy mask computation
"""

import numpy as np
import math
from utils import normalise

def get_bbox_center(bbox):
    """
    Calculate the center coordinates of a bounding box.

    Parameters:
    - x: Top-left x-coordinate of the bounding box
    - y: Top-left y-coordinate of the bounding box
    - width: Width of the bounding box
    - height: Height of the bounding box

    Returns:
    - Tuple of (center_x, center_y)
    """
    x, y, width, height = bbox
    center_x = x + width / 2
    center_y = y + height / 2
    return center_x, center_y

def distance_to_edge(point, bounding_box):
    x, y = point
    x_bb, y_bb, w_bb, h_bb = bounding_box
    
    # Calculate distances to each edge
    left_distance = x - x_bb
    right_distance = x_bb + w_bb - x
    top_distance = y - y_bb
    bottom_distance = y_bb + h_bb - y
    
    return min(left_distance, right_distance, top_distance, bottom_distance)

def distance_to_centroid(point, centroid):
    x_centroid, y_centroid = centroid
    x, y = point
    distance = math.sqrt((x - x_centroid)**2 + (y - y_centroid)**2)
    return distance

def assign_weights(points, bounding_box, centroid, max_weight, use_bbox = True, decay_factor=0.5):
    weights = []
    
    for point in points:
        if use_bbox:
            distance = distance_to_edge(point, bounding_box)
            weight = exponential_decay_bbox(distance, decay_factor, max_weight) # Adding a small constant to avoid division by zero

        else:
            distance = distance_to_centroid(point, centroid)
            weight = exponential_decay_centroid(distance, decay_factor, max_weight) # Adding a small constant to avoid division by zero
        
        # Inverse proportionality with distance
        # weight = exponential_decay(distance, decay_factor, 0.8) + 0.001  # Adding a small constant to avoid division by zero
        weights.append(weight)
    
    return weights


def exponential_decay_centroid(distance, decay_factor, max_weight):
    return max_weight * (1 - math.exp(-decay_factor * distance))

def exponential_decay_bbox(distance, decay_factor, max_weight):
    return max_weight * math.exp(-decay_factor * distance)

def linear_decay(distance, max_distance):
    return max(0, distance / max_distance)


def sample_n_points(bbox, num_points, scaling_factor=0.2):
    """
    Sample points from a Gaussian distribution within a bounding box.

    Parameters:
    - bbox: Tuple or list representing the bounding box (x, y, width, height)
    - num_points: Number of points to generate
    - scaling_factor: Scaling factor for variance (default is 0.2)

    Returns:
    - List of tuples representing the sampled points [(x1, y1), (x2, y2), ...]
    """
    cx, cy = get_bbox_center(bbox)  # Center coordinates
    width, height = bbox[2], bbox[3]  # Bbox size

    # print(f'Center: {cx, cy}, Width: {width}, Height: {height}')
    # print(f'actual width: {width}, actual height: {height}')
    # print(f'width * scaling_factor: {scaling_factor * width}, height * scaling_factor: {scaling_factor * height}')

    # Calculate mean and variance
    mean_x, mean_y = cx, cy
    variance_x, variance_y = scaling_factor * width, scaling_factor * height

    sampled_points_x = np.random.normal(mean_x, variance_x, num_points)
    sampled_points_y = np.random.normal(mean_y, variance_y, num_points)

    # Clip the points to be within the bounding box
    sampled_points_x = np.clip(sampled_points_x, bbox[0], bbox[0] + width)
    sampled_points_y = np.clip(sampled_points_y, bbox[1], bbox[1] + height)

    return list(zip(map(int, sampled_points_x), map(int, sampled_points_y)))




def compute_partial_fuzzy_mask(image, fixation_point, sigma, A=1.0):
    """
    Compute a partial fuzzy mask around a fixation point in an image.

    Parameters:
    - image: The input image.
    - fixation_point (tuple): The coordinates (x, y) of the fixation point.
    - sigma (float): The standard deviation for the Gaussian function.
    - A (float): Amplitude parameter for the Gaussian function. Default is 1.0.

    Returns:
    - mesh_grid (numpy.ndarray): 2D array representing the partial fuzzy mask.
    """
    h, w, _ = image.shape
    x = np.arange(w)
    y = np.arange(h)

    X, Y = np.meshgrid(x, y)

    mesh_grid = np.dstack((X, Y))
    # mesh_grid = (mesh_grid - fixation_point)**2 / (2 * sigma**2)
    mesh_grid = ((mesh_grid - fixation_point[0])**2 / (2 * sigma[0]**2)) + ((mesh_grid - fixation_point[1])**2 / (2 * sigma[1]**2))

    mesh_grid = A * np.exp(-np.sum(mesh_grid, axis=2))
    return mesh_grid

def compute_fuzzy_mask(image, fixation_points):
    """
    Compute a combined fuzzy mask based on multiple fixation points in an image.

    Parameters:
    - image: The input image.
    - fixation_points (list of tuples): List of fixation points' coordinates (x, y).
    - sigma (float): The standard deviation for the Gaussian function.

    Returns:
    - saliency_map (numpy.ndarray): 2D array representing the combined fuzzy mask.
    """
    saliency_map = 0.0

    sigma = compute_sigma(fixation_points)
    A = 1 / (2 * np.pi * sigma[0] * sigma[1])
    for fixation_point in fixation_points:
        saliency_map += compute_partial_fuzzy_mask(image, fixation_point, sigma, A)

    saliency_map = saliency_map / len(fixation_points)
    saliency_map = normalise(saliency_map)
    return saliency_map

def compute_sigma(fixation_points):
    """
    Compute the standard deviation for the Gaussian function based on the fixation points.

    Parameters:
    - fixation_points (list of tuples): List of fixation points' coordinates (x, y).

    Returns:
    - sigma (float): The standard deviation for the Gaussian function.
    """

    N = len(fixation_points)

    sig_x = np.std([point[0] for point in fixation_points])
    sig_y = np.std([point[1] for point in fixation_points])

    # Calculating bandwidth using Silverman's rule of thumb
    w_x = sig_x / (2 * N**(1/6))
    w_y = sig_y / (2 * N**(1/6))

    return w_x, w_y
    

# fuzzy_mask = compute_fuzzy_mask(current_frame_rgb, largest_cluster)
# fuzzy_mask = fuzzy_mask.T