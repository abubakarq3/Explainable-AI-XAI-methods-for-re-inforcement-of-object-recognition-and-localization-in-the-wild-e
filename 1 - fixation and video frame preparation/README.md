# This is the Readme file for the gaze fixation and video frame generation parts of the project:


## save_fixation_and_frame_from_video.py
- This script contains the functions required to generate and save both the gaze fixation information, and video frames. It contains two functions: *generate_annotation_txt* and *generate_annotation_txt_with_z*, which both generate the annotation file and video frames, but the latter includes the depth coordinate of each fixation point. 

Their inputs are:
- Path to the Object dataset (String)

Their outputs are both saved to the provided path and include:
- A frames folder for each sequence, containing the video frames
- A txt file for each sequence econtaining the frame number, and the coordinates of the gaze fixation point for each frame.


The dataset directory for all objects on the aivcalc4 labri server is "/data/batoki/GITW/Masking", so each object name should be added after *Masking*