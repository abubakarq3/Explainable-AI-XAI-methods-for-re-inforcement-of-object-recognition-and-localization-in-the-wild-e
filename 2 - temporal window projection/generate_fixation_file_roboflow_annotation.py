"""
This script generates a txt file containing the gaze points for each corresponding frame.
Made by Bolutife Atoki on 15/04/24
"""


import os
import cv2
import numpy as np


# Importing functions from other files
from utils import (create_nested_dict_from_csv, get_gp_for_frame, read_all_annotation_with_z, clustering)
from utils_homography import homography, project_fixation
#from utils_sam import check_required_directories


saveFixationFile = open('Complete_Gaze_Fixations_projected.txt', "a+")
test_save_path = 'test_visualization'

window_size = 10
EPS_CLUSTERING = 1.4 #0.05 #0.005, 0.01,0.05, 0.1, 0.15, 0.20,0.25 #DBSCAN N -radius 
MIN_SAMPLE_CLUSTERING = 2 #DBSCAN min density
HOMOGRAPHY_RADIUS = 100 #Parameter for homography


M = np.eye(3)
data = []
coord = []
projected_fixations = []
Hk_s = []
break_counter = 0
iou_list = []
pixel_accuracy_list = []
dice_list = []
time_list = []
frame_counter = 0
no_masks = 0

# object_list = sorted(['Lid', 'OilBottle', 'Rice', 'WashLiquid', 'Glass', 'FryingPan', 'VinegarBottle'])
object_list = sorted(['FryingPan'])

full_path = "/net/travail/rramesh/TRDP v2/Dataset"

for obj_path in object_list:

    obj_path = os.path.join(full_path,obj_path)  
    sequence_list = os.listdir(obj_path)
    sequence_list = sorted(sequence_list)

    for sequence_name in sorted(sequence_list):
        # print(f'Processing {object_name}/{object_type}/{sequence_name}')
        sequence_path = os.path.join(obj_path, sequence_name)

        shortened_sequence_name = sequence_name[-14:]

        print(f'sequence_name: {sequence_name}')


        fixations_list = read_all_annotation_with_z(obj_path + '/' + sequence_name)

        for window_index, index in enumerate(range( (window_size - 1), len(fixations_list))):

            currentpath = test_save_path + '/' + sequence_name + fixations_list[index][0] + '.jpg'

            if not os.path.exists(currentpath):
                print(f'-------------Processing frame {fixations_list[index][0]}---------------------------')

                # print(f'window_index: {window_index}, index: {index}')

                ############################################################################################################################
                #                         Step 2: Loading the frame and making copies to save the different required images                #
                ############################################################################################################################
                current_frame = cv2.imread(obj_path + '/' + sequence_name + '/Frames/' +  fixations_list[index][0] + '.jpg', cv2.IMREAD_UNCHANGED)
                # print(f'current frame shape: {current_frame.shape}')
                current_frame_rgb_clusters = current_frame.copy()
                current_frame_rgb_largest_cluster = current_frame.copy()


                width = current_frame.shape[1]
                height = current_frame.shape[0]

                ############################################################################################################################
                #                                Step 3: Loading the fixation and computing sigma in pixels                                #
                ############################################################################################################################
                current_fixation = (float(fixations_list[index][2]), float(fixations_list[index][3]))
                current_fixation_z = float(fixations_list[index][4])

                ############################################################################################################################
                #                                  Step 4: Computing the homography for the current frame                                  #
                ############################################################################################################################
                # print('top of homography loop')
                for i in range(max(0, index - window_size + 1), index): #0 to 9, 10 to 19, 20 to 29, 30 to 39, 40 to 49
                    ref_frame = cv2.imread(obj_path + '/' + sequence_name + '/Frames/' +  fixations_list[i][0] + '.jpg', 0) 
                    ref_fixation = (float(fixations_list[i][2]), float(fixations_list[i][3])) 

                    target_frame = cv2.imread(obj_path + '/' + sequence_name + '/Frames/' +  fixations_list[i+1][0] + '.jpg', 0) 
                    target_fixation = (float(fixations_list[i+1][2]), float(fixations_list[i+1][3])) 

                    Mk = homography(ref_frame, target_frame, ref_fixation, target_fixation, HOMOGRAPHY_RADIUS)

                    if Mk is not None:
                        M = np.matmul(Mk, M)
                    else:
                        data.append('[]\n')
                        print(f'Homography not found between {fixations_list[i][0]} and {fixations_list[i+1][0]}')

                    Hk_s.insert(0, M) 
                ############################################################################################################################
                #                     Step 5: Projecting the fixation for each previous frame to current frame                             #
                ############################################################################################################################
                window_fixations = fixations_list[max(0, index - window_size + 1):index] 

                for window_item_idx in range(0,window_size-1): 
                    (x_res, y_res) = project_fixation(float(window_fixations[window_item_idx][2]), float(window_fixations[window_item_idx][3]), Hk_s[window_item_idx]) 
                    projected_fixations.append((x_res, y_res)) 
                current_frame = cv2.circle(current_frame, (int(current_fixation[0]), int(current_fixation[1])), 4, (255, 0, 0), -1) 

                projected_fixations.append((current_fixation[0], current_fixation[1]))        

                ############################################################################################################################
                #                                          Step 5: Clustering the gaze points                                              #
                ############################################################################################################################
                largest_cluster = np.array(projected_fixations)

                #Visualization
                for coord in largest_cluster: 
                    current_frame_rgb_clusters = cv2.circle(current_frame_rgb_clusters, (int(coord[0]), int(coord[1])), 4, (0, 255, 0), -1)
                    current_frame_rgb_largest_cluster = cv2.circle(current_frame_rgb_largest_cluster, (int(coord[0]), int(coord[1])), 4, (0, 255, 0), -1)

                centroid = largest_cluster.mean(axis=0)
                current_frame_rgb_largest_cluster = cv2.circle(current_frame_rgb_largest_cluster, (int(centroid[0]), int(centroid[1])), 4, (0, 0, 255), -1)

                cv2.imwrite(test_save_path + '/' + sequence_name + fixations_list[index][0] + '.jpg', current_frame_rgb_clusters) 

                saveFixationFile.write(f'{sequence_name}_{fixations_list[index][0]} {projected_fixations}\n')


                M = np.eye(3) 
                Hk_s = [] 
                projected_fixations = [] 
            else:
                print(f"This file {currentpath} already exist")