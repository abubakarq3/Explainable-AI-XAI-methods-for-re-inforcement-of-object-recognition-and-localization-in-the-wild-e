"""
This script generates the fuzzy masks for fine-tunung SAM
#Produced by Bolutife Atoki 04/05/2024
"""

"""
It uses the following steps:
1. Loads a csv file containing the sequences to be considered for finetuning.
These sequences are manually selected based on the following criteria:
    - The object of interest is fixated on
    - There is little to no drifts of the gaze point on the object
2. For each sequence in the csv, corresponding start and stop frames (based on gaze point from appearing on object up until grasping process)
3. For each frame in the sequence, the following steps are performed:
    - Load the frame
    - Load the fixation
    - Compute the homography of frames in temporal window unto the current frame
    - Project the fixation of the previous frames to the current frame
    - Cluster these gaze points and select the largest cluster
    - Use Segment Anything to detect the object at these gaze points and obtain its bounding box
    - Add more points to largest cluster and weight them
    - Compute the fuzzy mask with largest cluster using the Gaussian KDE
    - Save Image, Fuzzy Mask and Superimposed Image
"""


import os
import matplotlib.pyplot as plt
import cv2
from scipy import stats
import numpy as np
from PIL import Image
import argparse

# Importing functions from other files
from utils import (read_all_annotation_with_z, normalise, get_projected_fixations_for_frame_fine_tune,
                   get_bbox_for_frame)
from utils_kde import sample_n_points, assign_weights
from representations import represent_heatmap_overlaid
from utils_foveation import compute_sigma_pixels


#-##########################################################################################################################
#                                                      Parameter Settings                                                  #
###########################################f#################################################################################
EPS_CLUSTERING = 1.4 #0.05 #0.005, 0.01,0.05, 0.1, 0.15, 0.20,0.25 #DBSCAN N -radius 
MIN_SAMPLE_CLUSTERING = 2 #DBSCAN min density

HOMOGRAPHY_RADIUS = 100 #Parameter for homography
delta_t_exp = int(250 / (1000/25)) #Scene exploration time in frames... number of frames at the be to skip due to eye movement

ALPHA = 2.0 # Gaussian propagation angle
CAM_FOV_X = 82 # Camera Horizontal Field of View in degrees
CAM_FOV_Y = 52 # Camera Vertical Field of View in degrees


# sam_checkpoint = "sam_vit_h_4b8939.pth" #Path to the SAM model checkpoint
# model_type = "vit_h" #Type of model to use for SAM
# device = "cuda" #Device to use for SAM
###########################################################################################################################

M = np.eye(3)
data = []
coord = []
projected_fixations = []
Hk_s = []
break_counter = 0
correct_predictions = 0
frame_counter = 0

# Arguments Parser Declaration
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_points", default=100, type=int,
                    help='Number of points to sample for the bounding box')
    parser.add_argument("--max_weight", default=0.8, type=float,
                    help='Maximum weight for the new points')

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    # window_size = args.window_size
    # object_name = args.object_name
    num_points = args.num_points
    max_weight = args.max_weight
    fixation_path = "/net/travail/rramesh/TRDP v2/Complete_Gaze_Fixations_projected.txt"

    binMaskDir = 'GITW Filtered Binary Masks'



    fuzzyMaskDir = 'Fuzzy Mask'
    if os.path.exists(fuzzyMaskDir)==0:
        os.makedirs(fuzzyMaskDir)

    initialFuzzyMaskDir = 'Initial Fuzzy Mask'
    if os.path.exists(initialFuzzyMaskDir)==0:
        os.makedirs(initialFuzzyMaskDir)

    blendedFuzzyMaskDir = 'Blended Fuzzy Mask'
    if os.path.exists(blendedFuzzyMaskDir)==0:
        os.makedirs(blendedFuzzyMaskDir)

    # object_list = ['CanOfCocaCola', 'Mug', 'Plate', 'Sponge', 'Sugar']
    # object_list = ['Lid', 'OilBottle', 'Rice', 'WashLiquid', 'Glass', 'FryingPan', 'VinegarBottle']
    object_pt = "/net/travail/rramesh/TRDP v2/Dataset"
    object_list = ['Jam']

    for object_name in object_list:

        #sequence_list = sorted([ f.name for f in os.scandir(object_name) if f.is_dir() ])
        sequence_list = sorted(os.listdir(os.path.join(object_pt,object_name)))
        print(sequence_list)

        for sequence_name in sequence_list:
            video_name = sequence_name
            print(f'----------------------------------Processing {video_name}----------------------------------------')

            fixations_list = read_all_annotation_with_z(os.path.join(object_pt,object_name) + '/' + video_name)


            for window_index, index in enumerate(range(len(fixations_list))): 
                # print(f'-------------Processing frame {fixations_list[index][0]}---------------------------')

                ############################################################################################################################
                #                         Step 2: Loading the frame and making copies to save the different required images                #
                ############################################################################################################################
                current_frame = cv2.imread(os.path.join(object_pt,object_name) + '/' + video_name + '/Frames/' +  fixations_list[index][0] + '.jpg', cv2.IMREAD_UNCHANGED)
                current_frame_copy = current_frame.copy()


                width = current_frame.shape[1]
                height = current_frame.shape[0]

                ############################################################################################################################
                #                                Step 3: Loading the fixation and computing sigma in pixels                                #
                ############################################################################################################################
                current_fixation = (float(fixations_list[index][2]), float(fixations_list[index][3]))
                current_fixation_z = float(fixations_list[index][4])

                sigma_x, sigma_y = compute_sigma_pixels(current_fixation_z, ALPHA, width, height, CAM_FOV_X, CAM_FOV_Y)
                
                target_frame_name = video_name + '_' + fixations_list[index][0]

                file_path = os.path.join(binMaskDir, os.path.join(object_pt,object_name), target_frame_name + '.jpg')

                if os.path.isfile(file_path):
                    fixation_points = get_projected_fixations_for_frame_fine_tune(fixation_path, target_frame_name)
                    fixation_points = np.array(fixation_points)
                    centroid = fixation_points.mean(axis=0)

                    ############################################################################################################################
                    #                                                Step 4: Get Estimated BBOX                                                #
                    ############################################################################################################################
                    bbox = get_bbox_for_frame(os.path.join(os.path.join(object_pt,object_name), video_name, 'bounding_box_finetune.txt'), fixations_list[index][0])


                    ############################################################################################################################
                    #                                 Step 5: Adding more points to largest cluster and weighting                              #
                    ############################################################################################################################y
                    new_points = sample_n_points(bbox, num_points, scaling_factor=0.2)


                    weights_new_points = assign_weights(new_points, bbox, centroid, max_weight, use_bbox = False, decay_factor=0.2)
                    weights_cluster_points = np.ones(len(fixation_points))
                    weights = np.concatenate((weights_cluster_points, weights_new_points), axis=0)
                    fixation_points = np.concatenate((fixation_points, new_points), axis=0)

                    ############################################################################################################################
                    #                   Step 6: Computing the fuzzy mask with largest cluster using the Gaussian KDE                           #
                    ############################################################################################################################
                    kernel = stats.gaussian_kde(fixation_points.T, bw_method='silverman', weights=weights)

                    h, w, _ = current_frame.shape
                    x_coords, y_coords = np.mgrid[0:w, 0:h]
                    coords = np.vstack([x_coords.ravel(), y_coords.ravel()])
                    fuzzy_mask = kernel.evaluate(coords)
                    fuzzy_mask = fuzzy_mask.reshape(w, h).T 
                    fuzzy_mask = normalise(fuzzy_mask) * 255

                    ############################################################################################################################
                    #                           Step 7: Saving Image, Fuzzy Mask and Superimposed Image                                       #
                    ############################################################################################################################
                    cv2.imwrite(fuzzyMaskDir + '/' + video_name + '_' + fixations_list[index][0] + '.jpg', fuzzy_mask)

                    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

                    blended_heatmap = represent_heatmap_overlaid(fuzzy_mask, Image.fromarray(current_frame), 'turbo')
                    blended_heatmap = blended_heatmap.convert("RGB")
                    blended_heatmap.save(blendedFuzzyMaskDir + '/' + video_name + '_' + fixations_list[index][0] + '.jpg')


    

