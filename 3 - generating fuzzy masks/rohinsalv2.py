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
from utils_foveation import compute_sigma_pixels,compute_saliency_map


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


object_pt = "/net/travail/rramesh/TRDP v2/Dataset"
fixation_path ="/net/travail/rramesh/TRDP v2/Complete_Gaze_Fixations_projected.txt"

object_list = ['Jam']

for object_name in object_list:

    #sequence_list = sorted([ f.name for f in os.scandir(object_name) if f.is_dir() ])
    sequence_list = sorted(os.listdir(os.path.join(object_pt,object_name)))
    print(sequence_list)

    for sequence_name in sequence_list:
        print(f'----------------------------------Processing {sequence_name}----------------------------------------')

        fixations_list = read_all_annotation_with_z(os.path.join(object_pt,object_name) + '/' + sequence_name)

        for window_index, index in enumerate(range(len(fixations_list))): 
            current_frame = cv2.imread(os.path.join(object_pt,object_name) + '/' + sequence_name + '/Frames/' +  fixations_list[index][0] + '.jpg', cv2.IMREAD_UNCHANGED)
            current_frame_copy = current_frame.copy()


            width = current_frame.shape[1]
            height = current_frame.shape[0]
             
            current_fixation = (float(fixations_list[index][2]), float(fixations_list[index][3]))
            current_fixation_z = float(fixations_list[index][4])

            sigma_x, sigma_y = compute_sigma_pixels(current_fixation_z, ALPHA, width, height, CAM_FOV_X, CAM_FOV_Y) 

            target_frame_name = sequence_name + '_' + fixations_list[index][0]

            fixation_points = get_projected_fixations_for_frame_fine_tune(fixation_path, target_frame_name)
            fixation_points = np.array(fixation_points)
            print(fixation_points)
            if fixation_points is not None:

                accumulated_saliency_map = np.zeros_like(current_frame, dtype=float)
                accumulated_saliency_map = np.zeros((current_frame.shape[1], current_frame.shape[0]), dtype=np.float64)


                for coord in fixation_points:
                    saliency_map = compute_saliency_map(current_frame, coord, sigma_x, sigma_y, 1)

                    print(f"the shape of the saliency map is {saliency_map.shape}")
                    # Add the current saliency map to the accumulated saliency map
                    accumulated_saliency_map += saliency_map



                accumulated_saliency_map_normalized = accumulated_saliency_map / accumulated_saliency_map.max()
                accumulated_saliency_map_normalized =accumulated_saliency_map_normalized.T
                print(f"the maximum of the acsal is {accumulated_saliency_map_normalized.max()} and min is {accumulated_saliency_map_normalized.min()}")
                print(f"the shape of the accumnorm saliency map is {accumulated_saliency_map_normalized.shape}")
                blended_heatmap = represent_heatmap_overlaid(accumulated_saliency_map_normalized, Image.fromarray(current_frame), 'turbo')

                plt.imshow(blended_heatmap)
                plt.axis('off')  # Turn off the axis labels and ticks
                plt.show()

            else: 
                print("the fixation point is None")