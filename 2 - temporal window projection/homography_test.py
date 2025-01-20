from utils_homography import homography, project_fixation
import cv2

HOMOGRAPHY_RADIUS = 100

ref_frame = cv2.imread("/net/travail/rramesh/TRDP v2/Dataset/Jam/JamPlace4Subject4/Frames/Frame_5680.jpg", 0) 
ref_fixation = (float(1168), float(470)) 

target_frame = cv2.imread("/net/travail/rramesh/TRDP v2/Dataset/Jam/JamPlace4Subject4/Frames/Frame_5720.jpg", 0) 
target_fixation = (float(1191), float(425)) 

Mk = homography(ref_frame, target_frame, ref_fixation, target_fixation, HOMOGRAPHY_RADIUS)


