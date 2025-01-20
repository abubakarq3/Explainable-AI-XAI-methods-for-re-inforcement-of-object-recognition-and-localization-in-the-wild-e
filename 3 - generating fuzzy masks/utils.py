"""
General utility functions for the project
"""
import torch
import torch.nn as nn

import os
import numpy as np
import csv
from scipy.stats import multivariate_normal, gaussian_kde
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import ast


def extract_frame_values_from_file(file_path, frame):
    try:
        # Open the file and read its content
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Loop through each line
        for line in lines:
            # Check if the line starts with the frame you are looking for
            if line.startswith(frame):
                # Return the values after the frame identifier
                return line[len(frame):].strip()
        
        # Return None if the frame was not found
        return None
    
    except FileNotFoundError:
        return f"File not found: {file_path}"



def get_bbox_for_frame(file_path, target_frame):
    with open(file_path, 'r') as file:
        for line in file:
            if target_frame in line:
                # Split the line and extract bounding box coordinates
                parts = line.split()
                x, y, w, h = map(int, parts[1:])
                return (x, y, w, h)
    
    # Return None if the frame is not found in the file
    return None

def get_fixation_for_frame(file_path, target_frame):
    with open(file_path, 'r') as file:
        for line in file:
            if target_frame in line:
                # Split the line and extract bounding box coordinates
                parts = line.split()
                _, x, y, z = map(int, parts[1:])
                return (x, y, z)
    
    # Return None if the frame is not found in the file
    return None

def compute_square_bbox(center_x, center_y, side_length):
    half_side = side_length // 2
    x = center_x - half_side
    y = center_y - half_side
    width = side_length
    height = side_length
    return int(x), int(y), int(width), int(height)

def gaussian_2d(x, y, vX, vY, sigma_x, sigma_y):
    #Reference: https://www.originlab.com/doc/Origin-Help/Create-2D-Kernel-Density
    n_x = len(vX)
    n_y = len(vY)

    # Calculating bandwidth using Silverman's rule of thumb
    w_x = sigma_x / (2 * n_x**(1/6))
    w_y = sigma_y / (2 * n_y**(1/6))
    
    result = np.zeros((len(vX), len(vY)))
    temp = 0

    for i in vX:
        for j in vY:
            exponent = -((x - i)**2 / (2 * w_x**2) + (y - j)**2 / (2 * w_y**2))

            result[i,j] = np.exp(exponent) / (2 * np.pi * w_x * w_y)

    return result

def calculate_fuzzy_mask(centroid, image):

    h, w, _ = image.shape

    # Calculating the PDF values for each pixel coordinate using the Gaussian kernel
    pdf_values = gaussian_2d(centroid[0], centroid[1], np.arange(0, w, 1), np.arange(0, h, 1), np.std(np.arange(0, w, 1)), np.std(np.arange(0, h, 1)))

    # print(f'PDF Values shape: {pdf_values.shape}')
    # #Normalizing PDF values to range [0, 1]
    # pdf_values = normalise(pdf_values)
    # # Normalizing and scaling the PDF values to range [0, 255]
    # pdf_values = norm_and_scale(pdf_values)

    return pdf_values

def normalise(image):
    # Normalizing the pixel values to the range [0, 1]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    return image

def norm_and_scale(image):
    # Normalizing the pixel values to the range [0, 1]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # Scaling the pixel values to the range [0, 255]
    image = image * 255
    image = np.clip(image, 0, 255)

    return image

def clustering(x, eps, min_samples=2):
    """ Using DBSCAN clustering on the gaze points to remove the outliers
    @param x: raw gaze points
    @param eps: the maximum distance between the gaze points in a cluster
    @return: list of the maximum cluster of gaze points
    """

    clustering_data = DBSCAN(eps=eps, min_samples=min_samples).fit(x)
    lst = clustering_data.labels_.tolist()
    # print(f'Number of clusters: {len(set(lst))}')
    b = max(set(lst), key=lst.count)

    return x[clustering_data.labels_ == b]


def clustering_all_clusters(x, eps, min_samples=2):
    """ Using DBSCAN clustering on the gaze points to remove the outliers
    @param x: raw gaze points
    @param eps: the maximum distance between the gaze points in a cluster
    @return: list of the maximum cluster of gaze points
    """

    clustering_data = DBSCAN(eps=eps, min_samples=min_samples).fit(x)

    return clustering_data
    # lst = clustering_data.labels_.tolist()
    # # print(f'Number of clusters: {len(set(lst))}')
    # b = max(set(lst), key=lst.count)

    # return x[clustering_data.labels_ == b]


def create_bounding_box(x,y, sigma_x, sigma_y, k_x, k_y):
    top_left = (int(x - (k_x*sigma_x)), int(y - (k_y*sigma_y)))
    bottom_right = (int(x + (k_x*sigma_x)), int(y + (k_y*sigma_y)))

    return top_left, bottom_right

def create_output_directories(baseDir):
    framesWithGazeDir = baseDir + '/frames_with_gp'
    if os.path.exists(framesWithGazeDir)==0: 
        os.makedirs(framesWithGazeDir)
    
    eachClusterDir = baseDir + '/frames_with_each_cluster'
    if os.path.exists(eachClusterDir)==0: 
        os.makedirs(eachClusterDir)

    largestClusterDir = baseDir + '/frames_with_largest_cluster'
    if os.path.exists(largestClusterDir)==0: 
        os.makedirs(largestClusterDir)


    fuzzyMaskDir = baseDir + '/frames_with_fuzzy_mask'
    if os.path.exists(fuzzyMaskDir)==0: 
        os.makedirs(fuzzyMaskDir)

    blendedFuzzyMaskDir = baseDir + '/blended_fuzzy_mask'
    if os.path.exists(blendedFuzzyMaskDir)==0: 
        os.makedirs(blendedFuzzyMaskDir)

    blendedGpClusterDir = baseDir + '/blended_gp_cluster'
    if os.path.exists(blendedGpClusterDir)==0: 
        os.makedirs(blendedGpClusterDir)

    return framesWithGazeDir, eachClusterDir, largestClusterDir, fuzzyMaskDir, blendedFuzzyMaskDir, blendedGpClusterDir

def create_eval_directories(object_name, video_name):
        # Create output directories
    samMaskDir = object_name + '/' + video_name + '/sam_mask'
    if os.path.exists(samMaskDir)==0: 
        os.makedirs(samMaskDir)
    
    samBboxDir = object_name + '/' + video_name + '/sam_bbox'
    if os.path.exists(samBboxDir)==0: 
        os.makedirs(samBboxDir)

    samBlendedDir = object_name + '/' + video_name + '/sam_blended'
    if os.path.exists(samBlendedDir)==0: 
        os.makedirs(samBlendedDir)

    return samMaskDir, samBboxDir, samBlendedDir

def create_mask_eval_directories(object_name, video_name):
        # Create output directories
    samMaskDir = object_name + '/' + video_name + '/sam_mask'
    if os.path.exists(samMaskDir)==0: 
        os.makedirs(samMaskDir)
    
    gtBlendedDir = object_name + '/' + video_name + '/gt_blended'
    if os.path.exists(gtBlendedDir)==0: 
        os.makedirs(gtBlendedDir)

    samBlendedDir = object_name + '/' + video_name + '/sam_blended'
    if os.path.exists(samBlendedDir)==0: 
        os.makedirs(samBlendedDir)

    return samMaskDir, gtBlendedDir, samBlendedDir


def read_annotation(fileDir):
    l_fix = [] 
    l_last = [] 
    with open(fileDir + "/annotation.txt", 'r') as fichier:  
        stage = 0
        for line in fichier:
            elements = line.split()
            if stage == 0 and elements[1] == '1':
                stage = 1
            if stage == 1 and elements[1] == '0':
                stage = 2
            if stage == 1:
                l_fix.append(elements)
            elif stage == 2:
                l_last.append(elements)
    fichier.close()

    return l_fix

def read_annotation_with_z(fileDir):
    l_fix = [] 
    l_last = [] 
    with open(fileDir + "/annotation_with_z.txt", 'r') as fichier:  
        stage = 0
        for line in fichier:
            elements = line.split()
            if stage == 0 and elements[1] == '1':
                stage = 1
            if stage == 1 and elements[1] == '0':
                stage = 2
            if stage == 1:
                l_fix.append(elements)
            elif stage == 2:
                l_last.append(elements)
    fichier.close()

    return l_fix

def read_all_annotation(fileDir):
    l_fix = [] 
    with open(fileDir + "/annotation.txt", 'r') as fichier:  
        for line in fichier:
            elements = line.split()

            l_fix.append(elements)

    fichier.close()

    return l_fix

def read_all_annotation_with_z(fileDir):
    l_fix = [] 
    with open(fileDir + "/annotation_with_z.txt", 'r') as fichier:  
        for line in fichier:
            elements = line.split()

            l_fix.append(elements)

    fichier.close()

    return l_fix

def get_min_max(centroid, size):
    x, y = centroid
    x_min = x - size / 2
    x_max = x + size / 2
    y_min = y - size / 2
    y_max = y + size / 2

    return int(x_min), int(x_max), int(y_min), int(y_max)


def extract_values_for_frame(input_frame, data):
    # Convert input_frame to match the format (e.g., Frame_1)
    frame_number = int(input_frame.split('_')[1])  # Extract number from Frame_1, Frame_40 etc.
    
    # Split the data into lines
    lines = data.strip().split("\n")
    
    # Iterate over each line to find the frame
    for line in lines:
        if line.startswith(f"Frame_{frame_number}"):
            # Split the line by spaces and return the values (ignoring 'Frame_<number>')
            parts = line.split()
            return tuple(map(int, parts[1:]))  # Convert to integers and return (x, y, z)

    return None  # Return None if frame not found


def expanded_join(path: str, *paths: str) -> str:
    """Path concatenation utility function.
    Automatically handle slash and backslash depending on the OS but also relative path user.

    :param path: Most left path.
    :param paths: Most right parts of path to concatenate.
    :return: A string which contains the absolute path.
    """
    return os.path.expanduser(os.path.join(path, *paths))

class Patchify(nn.Module):
    def __init__(self, patch_size=56):
        super().__init__()
        self.p = patch_size
        self.unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x -> B c h w
        bs, c, h, w = x.shape
        
        x = self.unfold(x)
        # x -> B (c*p*p) L
        
        # Reshaping into the shape we want
        a = x.view(bs, c, self.p, self.p, -1).permute(0, 4, 1, 2, 3)
        # a -> ( B no.of patches c p p )
        return a
    


import csv

def create_nested_dict_from_csv(file_path):
    """
    Create a nested dictionary from a CSV file.

    The CSV file should have 4 columns. The first column should contain strings
    called object names, the second column should contain strings called sequences,
    the third column should contain start numbers, and the fourth column should
    contain end numbers. The function creates a nested dictionary where each key
    is a string from the first column, and the value is another dictionary where
    each key is a string from the second column, and the value is a list of numbers
    from the start number to the end number with a step of 40.

    Parameters:
    file_path: The path to the CSV file.

    Returns:
    A nested dictionary created from the CSV file.
    """
    nested_dict = {}
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        for row in reader:
            object_name, sequence, start, end, _ = row
            if object_name not in nested_dict:
                nested_dict[object_name] = {}
            nested_dict[object_name][sequence] = list(range(int(start), int(end), 40))
    return nested_dict


def bb_intersection_over_union(bbox_gt, bbox_pred):
    # determine the (x, y)-coordinates of the intersection rectangle
    x_gt = max(bbox_gt[0], bbox_pred[0])
    y_gt = max(bbox_gt[1], bbox_pred[1])
    x_pred = min(bbox_gt[0] + bbox_gt[2], bbox_pred[0] + bbox_pred[2])
    y_pred = min(bbox_gt[1] + bbox_gt[3], bbox_pred[1] + bbox_pred[3])
    # compute the area of intersection rectangle
    interArea = max(0, x_gt) * max(0, y_gt - y_gt + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    bbox_gt_area = (bbox_gt[2]+ 1) * (bbox_gt[3] + 1)
    bbox_pred_area = (bbox_pred[2] + 1) * (bbox_pred[3] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(bbox_gt_area + bbox_pred_area - interArea)
    iou = max(0, iou)
	# return the intersection over union value
    return iou  

def new_bb_intersection_over_union(boxA, boxB):
    # Convert the bounding boxes to the (x_min, y_min, x_max, y_max) format
    boxA = [boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3]]
    boxB = [boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3]]

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def compute_iou(gt_mask, pred_mask):
    mask1_area = np.count_nonzero( gt_mask )
    mask2_area = np.count_nonzero( pred_mask )
    intersection = np.count_nonzero( np.logical_and( gt_mask, pred_mask ) )
    iou = intersection/(mask1_area+mask2_area-intersection)
    return iou

def compute_pixel_accuracy(gt_mask, pred_mask):
    """
    Calculate pixel accuracy between ground truth mask and predicted mask.

    Parameters:
        gt_mask (numpy array): Ground truth binary mask (binary array).
        pred_mask (numpy array): Predicted binary mask (binary array).

    Returns:
        float: Pixel accuracy score.
    """
    # Ensure the shapes of the masks match
    if gt_mask.shape != pred_mask.shape:
        raise ValueError("Ground truth and predicted masks must have the same shape.")

    # Calculate pixel accuracy
    correct_pixels = np.sum(gt_mask == pred_mask)
    total_pixels = gt_mask.size

    accuracy = correct_pixels / total_pixels
    return accuracy

def compute_dice_coefficient(gt_mask, pred_mask):
    """
    Calculate Dice coefficient between ground truth mask and predicted mask.

    Parameters:
        gt_mask (numpy array): Ground truth binary mask (binary array).
        pred_mask (numpy array): Predicted binary mask (binary array).

    Returns:
        float: Dice coefficient score.
    """
    # Ensure the shapes of the masks match
    if gt_mask.shape != pred_mask.shape:
        raise ValueError("Ground truth and predicted masks must have the same shape.")

    # Calculate Dice coefficient
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    dice_coeff = (2. * intersection) / (gt_mask.sum() + pred_mask.sum())
    return dice_coeff

def get_object_type(place_name, object_dict):
    for key, value in object_dict.items():
        if place_name in value:
            return key
    return None

def get_gp_for_frame(file_path, target_frame):
    with open(file_path, 'r') as file:
        for line in file:
            if target_frame in line:
                # Split the line and extract bounding box coordinates
                parts = line.split()
                x, y = map(int, parts[2:])
                return (x, y)
    
    # Return None if the frame is not found in the file
    return None

def get_fixation_for_frame_fine_tune(file_path, target_frame):
    with open(file_path, 'r') as file:
        for line in file:
            if target_frame in line:
                # Split the line and extract bounding box coordinates
                parts = line.split()
                x, y = map(int, parts[1:])
                return (x, y)
    
    # Return None if the frame is not found in the file
    return None

def get_projected_fixations_for_frame_fine_tune(file_path, target_frame):
    with open(file_path, 'r') as file:
        for line in file:
            if target_frame in line:
                # Split the line and extract bounding box coordinates
                parts = line.split()
                fixations  = parts[1:]
                # Join the list elements into a single string
                joined_data = ''.join(fixations)

                # Evaluate the string to convert it into a list of tuples
                tuples_list = ast.literal_eval(joined_data)

                # Convert the tuples to integers
                fixations = [(int(x), int(y)) for x, y in tuples_list]

                return fixations
    
    # Return None if the frame is not found in the file
    return None


# def new_bb_intersection_over_union(boxA, boxB):
#     # determine the (x, y)-coordinates of the intersection rectangle
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])

#     # compute the area of intersection rectangle
#     interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
#     if interArea == 0:
#         return 0
#     # compute the area of both the prediction and ground-truth
#     # rectangles
#     boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
#     boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

#     # compute the intersection over union by taking the intersection
#     # area and dividing it by the sum of prediction + ground-truth
#     # areas - the interesection area
#     iou = interArea / float(boxAArea + boxBArea - interArea)

#     # return the intersection over union value
#     return iou