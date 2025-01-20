# This is the Readme file for the part of the project that estimates the bounding box of the object of interest. This bounding box is used to generate the fuzzy mask, as additional points are sampled inside its borders.

## compute_bounding_box.py
- This script generates for all frames of an object class, the estimated bounding box coordinates, which are saved into a txt file per sequence (with the frame names), and also plots the estimated bounding box.

Their inputs are:
- The path to the projected gaze fixation txt file (String)
- The name of the object of interest (String)


Their outputs are both saved to the provided path and include:
- A txt file for every frame of all objects specified, containing the bounding box coordinates
- The video frames, with the bounding box plotted.

## utils_foveation.py
- This script contains the utility functions used to compute the bounding box.

## utils.py
- This script contains the general utility functions used in the project.