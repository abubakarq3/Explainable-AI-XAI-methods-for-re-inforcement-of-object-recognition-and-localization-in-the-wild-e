# This is the Readme file for the part of the project that computes and saves the fuzzy mask of the object of interest.

## main_generate_fuzzy_masks.py
- This script generates for all object classes specified, the fuzzy masks of all frames for which binary masks were annotated.

Its inputs are:
- The path to the projected gaze fixation txt file (String)
- The path to the folder containing the binary masks (String)
- A list containing the names of the object classes (List of String)
- The number of points to be sampled and used for estimating the fuzzy masks (Int)
- The maximum weight of sampled points. (0-1) (Float)


Their outputs are both saved to the provided path and include:
- A txt file for every frame of all objects specified, containing the bounding box coordinates
- The video frames, with the bounding box plotted.

## utils_foveation.py
- This script contains the utility functions used to compute the bounding box.

## utils.py
- This script contains the general utility functions used in the project.

## utils_kde
- This script contains utility functions to sample points in the bounding box and weight them.

## representations.py
- This script contains utility functions to blend the obtained fuzzy mask with the image, using specified colour map and opacity level.