""" 
Run this code for generating the frame extraction and saliency map with z value 
"""


# import utils_foveation 
import os
from save_fixation_and_frame_from_video import generate_annotation_txt_with_z

pathfordata =  "/net/travail/rramesh/TRDP v2/Dataset/CanOfCocaCola"

subdata = os.listdir(pathfordata)

for subpath in subdata:
    annotfile = os.path.join(pathfordata ,subpath + "/annotation_with_z.txt")

    print(annotfile)
    if os.path.exists(annotfile):
        print("file alredy exisit")
        # utils_foveation.main(os.path.join(pathfordata ,subpath))
         
    else:
        print(f"File does not exist: {annotfile}")
        generate_annotation_txt_with_z(os.path.join(pathfordata ,subpath), scale_reduction=1.0)
    