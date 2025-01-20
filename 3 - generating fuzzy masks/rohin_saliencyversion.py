from utils_foveation import compute_sigma_pixels, compute_saliency_map
from utils import (read_all_annotation_with_z, normalise, get_projected_fixations_for_frame_fine_tune,
                   extract_values_for_frame, extract_frame_values_from_file)
import numpy as np
import cv2
from representations import represent_heatmap_overlaid
from PIL import Image
import matplotlib.pyplot as plt
import os

ALPHA = 2.0 # Gaussian propagation angle
CAM_FOV_X = 82 # Camera Horizontal Field of View in degrees
CAM_FOV_Y = 52 # Camera Vertical Field of View in degrees


maindatapath = "/net/travail/rramesh/TRDP v2/Dataset"
fixation_path ="/net/travail/rramesh/TRDP v2/Complete_Gaze_Fixations_projected.txt"
readimages_path =  "/net/travail/rramesh/TRDP v2/yoloformat/test/images/"

paths = os.listdir(readimages_path)

for seqname in paths:
    image = cv2.imread(os.path.join(readimages_path,seqname))
    
    width = image.shape[1]
    height = image.shape[0]

    seqname_alter = seqname.replace("Frame", "_Frame")
    seqname_alter, ext = os.path.splitext(seqname_alter)

    name_before_place2 = seqname_alter.split("Place")[0]

    projected_points = get_projected_fixations_for_frame_fine_tune(fixation_path, seqname_alter)
    projected_points = np.array(projected_points)

    # Check if the array is empty
    print(f"the projected points are {projected_points}")
    # print(f"the projected points are {projected_points}")
    #get the z value
    name_before_Frame = seqname_alter.split("_Frame")[0]
    pathforannoz = os.path.join(maindatapath,name_before_place2,name_before_Frame)
     
    frame_index = seqname_alter.split("_")[1] + "_" + seqname_alter.split("_")[2]
    fixations_list = extract_frame_values_from_file(pathforannoz + "/annotation_with_z.txt", frame_index)
    z_fixation  = fixations_list.split()[-1]

    accumulated_saliency_map = np.zeros_like(image, dtype=float)
    accumulated_saliency_map = np.zeros((image.shape[1], image.shape[0]), dtype=np.float64)


    for coord in projected_points:
        sigma_x, sigma_y = compute_sigma_pixels(int(z_fixation), ALPHA, width, height, CAM_FOV_X, CAM_FOV_Y)
        saliency_map = compute_saliency_map(image, coord, sigma_x, sigma_y, 1)

        print(f"the shape of the saliency map is {saliency_map.shape}")
        # Add the current saliency map to the accumulated saliency map
        accumulated_saliency_map += saliency_map



    accumulated_saliency_map_normalized = accumulated_saliency_map / accumulated_saliency_map.max()
    accumulated_saliency_map_normalized =accumulated_saliency_map_normalized.T
    print(f"the maximum of the acsal is {accumulated_saliency_map_normalized.max()} and min is {accumulated_saliency_map_normalized.min()}")
    print(f"the shape of the accumnorm saliency map is {accumulated_saliency_map_normalized.shape}")
    blended_heatmap = represent_heatmap_overlaid(accumulated_saliency_map_normalized, Image.fromarray(image), 'turbo')
    
    # Specify the output directory path
    output_image_path = "/net/travail/rramesh/TRDP v2/saliencyinYoloimage/"

    # Check if the directory exists
    if not os.path.exists(output_image_path):
        os.makedirs(output_image_path)  # Create the directory if it doesn't exist

    # Specify the complete file path including the image name
    file_name = seqname_alter + ".png"  # You can change the file name and extension as needed
    complete_image_path = os.path.join(output_image_path, file_name)

    # Save the blended heatmap
    blended_heatmap.save(complete_image_path)

    print(f"Image saved at {complete_image_path}")



    
    

# fixation_points = get_projected_fixations_for_frame_fine_tune(fixation_path, "CanOfCocaColaPlace3Subject4_Frame_160")
# fixation_points = np.array(fixation_points)
# z= 1183

# print(fixation_points)

# image = cv2.imread(r"C:\Users\rohin\Desktop\New folder (2)\trdp\dataset\Bowl\Bowl\BowlPlace1Subject1\extractimages\frame0.jpg") 
# height ,width,  _ = image.shape
# print(f"the shaoe of the image is {image.shape}")

# #accumulated_saliency_map = np.zeros_like(image, dtype=float)
# accumulated_saliency_map = np.zeros((image.shape[1], image.shape[0]), dtype=np.float64)


# for coord in fixation_points:
#     sigma_x, sigma_y = compute_sigma_pixels(z, ALPHA, width, height, CAM_FOV_X, CAM_FOV_Y)
#     saliency_map = compute_saliency_map(image, coord, sigma_x, sigma_y, 1)

#     print(f"the shape of the saliency map is {saliency_map.shape}")
#     # Add the current saliency map to the accumulated saliency map
#     accumulated_saliency_map += saliency_map



# accumulated_saliency_map_normalized = accumulated_saliency_map / accumulated_saliency_map.max()
# accumulated_saliency_map_normalized =accumulated_saliency_map_normalized.T
# print(f"the maximum of the acsal is {accumulated_saliency_map_normalized.max()} and min is {accumulated_saliency_map_normalized.min()}")
# print(f"the shape of the accumnorm saliency map is {accumulated_saliency_map_normalized.shape}")
# blended_heatmap = represent_heatmap_overlaid(accumulated_saliency_map_normalized, Image.fromarray(image), 'turbo')



# # Plot the blended heatmap using matplotlib
# plt.figure(figsize=(10, 8))  # Set figure size
# plt.imshow(blended_heatmap)  # Display the image
# plt.axis('off')  # Hide the axes for better visualization
# plt.title("Blended Heatmap")  # Optional: Add a title
# plt.show()