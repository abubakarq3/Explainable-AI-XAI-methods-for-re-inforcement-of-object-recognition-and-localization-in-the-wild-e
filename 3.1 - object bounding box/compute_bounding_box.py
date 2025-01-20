"""
This script loads the sequence, computes the bounding box around the object and saves both the frames with the bounding box and the bounding box in txt file
It works for all sequences for the specified object
#Produced by Bolutife Atoki on 07/03/2024

It uses the following steps:
1. It optionally creates an annotation file if none exists.
2. Loads the gaze points and Z coordinate value from the annotation_with_z file.
3. Loads the frames from the video.
4. Prints the gaze points on each frame.
5. Computes the Foveation area on the object using the z coordinate value and the camera field of view parameters
6. Generates a mask for the foveation area
7. Obtains the bbox of the object using k*sigma_x and k*sigma_y
8. Saves the bbox annotations to file 
9. Superimposes the mask on the frame
10. Saves the generated images.
"""

import os
import matplotlib.pyplot as plt
import cv2
from scipy import stats
import numpy as np
from PIL import Image
import argparse

# Importing functions from other files
from save_fixation_and_frame_from_video import generate_annotation_txt_with_z
from utils import read_all_annotation_with_z, create_bounding_box, get_projected_fixations_for_frame_fine_tune
from utils_foveation import compute_sigma_pixels



#-##########################################################################################################################
#                                                      Parameter Settings                                                  #
############################################################################################################################
ALPHA = 2.0 # Gaussian propagation angle
CAM_FOV_X = 82 # Camera Horizontal Field of View in degrees
CAM_FOV_Y = 52 # Camera Vertical Field of View in degrees
K_X = 4 # Number of standard deviations to consider for the bounding box
K_Y = 4 # Number of standard deviations to consider for the bounding box
############################################################################################################################
break_counter = 0

# Arguments Parser Declaration
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_name", default='MilkBottle', type=str,
                        help='Name of the object in the video file')
    parser.add_argument("--fixation_path", default='Complete_Gaze_Fixations_projected.txt', type=str,
                        help='Path to the fixation file')


    return parser.parse_args()

if __name__ == "__main__":

    #args = parse_args()
    fixation_path = "/net/travail/rramesh/TRDP v2/Complete_Gaze_Fixations_projected_4class.txt"

    object_name = "/net/travail/rramesh/TRDP v2/Dataset/FryingPan"

    sequence_list = [ f.name for f in os.scandir(object_name) if f.is_dir() ]
     
    print(f"the seqence list os the image place and objects are: {sequence_list}") 
     
    for video_name in sorted(sequence_list):
        # break_counter = 0

        print(f'----------------------------------Processing {video_name}----------------------------------------')

        # Create annotation file if none exists
        if os.path.exists(object_name + '/' + video_name + "/annotation_with_z.txt"): 
            print("Annotation file exists")  

        else:
            print('No annotation with z-coordinates file')
            generate_annotation_txt_with_z(object_name + '/' + video_name)


        # Creating save directory
        framesWithObjectFoveationDir = object_name + '/' + video_name + '/Frame_with_foveation_area'
        if os.path.exists(framesWithObjectFoveationDir)==0: 
            os.makedirs(framesWithObjectFoveationDir)
        framesWithBboxDir = object_name + '/' + video_name + '/Frame_with_bbox'
        if os.path.exists(framesWithBboxDir)==0: 
            os.makedirs(framesWithBboxDir)

        # Read the fixations and frame numbers from annotation.txt
        fixations_list = read_all_annotation_with_z(object_name + '/' + video_name)


        bboxFile = os.path.join(object_name + '/' + video_name, 'bounding_box_finetune.txt')
        bf = open(bboxFile, "w")


        for index in (range(len(fixations_list))): 
            print(f'-------------Processing frame {fixations_list[index][0]}---------------------------')

            current_frame= cv2.imread(object_name + '/' + video_name + '/Frames/' +  fixations_list[index][0] + '.jpg', cv2.IMREAD_UNCHANGED)
            current_frame_bbox = current_frame.copy()
            current_frame_pil = Image.fromarray(current_frame)

            width = current_frame.shape[1]
            height = current_frame.shape[0]

            current_fixation = (float(fixations_list[index][2]), float(fixations_list[index][3]))
            current_fixation_z = float(fixations_list[index][4])


            target_frame_name = video_name + '_' + fixations_list[index][0]

            fixation_points = get_projected_fixations_for_frame_fine_tune(fixation_path, target_frame_name)

            if fixation_points:
                mean_fixation = np.mean(fixation_points, axis=0)

                ############################################################################################################################
                #                               Step 1: Computing the Sigma in pixels for x and y axis                                     #
                ############################################################################################################################

                sigma_x, sigma_y = compute_sigma_pixels(current_fixation_z, ALPHA, width, height, CAM_FOV_X, CAM_FOV_Y)

                ############################################################################################################################
                #                                            Step 2: Creating BBOX around object                                           #
                ############################################################################################################################
                top_left, bottom_right = create_bounding_box(int(mean_fixation[0]), int(mean_fixation[1]), sigma_x, sigma_y, K_X, K_Y)

                bbox = f"{fixations_list[index][0] + '.png'} {top_left[0]} {top_left[1]} {bottom_right[0] - top_left[0]  } {bottom_right[1] - top_left[1]}\n"
                bf.write(bbox)

                current_frame_bbox = cv2.rectangle(current_frame_bbox, top_left, bottom_right, (0,255,0), 4)

                cv2.imwrite(framesWithBboxDir + '/' + fixations_list[index][0] + '.jpg', current_frame_bbox) 
            else:
                print(f"The file {target_frame_name} is not in the fixation {fixation_path} main path")

        bf.close()
        print(f"Bounding Box file for {video_name} generated")