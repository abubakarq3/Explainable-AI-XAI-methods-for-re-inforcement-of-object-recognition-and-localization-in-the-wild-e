# This is the Readme file for the temporal window projection part of the project:

## generate_fixation_file_roboflow_annotation.py
- This script generates a txt file, containing current and projected gaze fixation points for each frame, by estimating the homography matrices between consecutive frames in the window, and projecting fixation points from previous frames, unto the last frame in the window... For each frame.

Its inputs are:
- Length of temporal window (Integer)
- Names of objects to compute projection for (List of Strings)
- Clustering and homography parameters (Float and Integers)
- Name of txt file to store projections.

Its outputs are both saved to the provided path and include:
- A txt file for every frame of all objects specified, containing current and projected gaze fixation point coordinates

## utils_homography.py
- This script contains the utility functions required to compute the homography matrix, between two frames.


It contains the following functions: 
- Function: homography
- Description: Compute homography transformation matrix between two images using SIFT keypoints and descriptors.
- Input:
- src: Source image (numpy array).
  - dst: Destination image (numpy array).
   - center_src: Center coordinates (x, y) of the region of interest in the source image.
   - center_dst: Center coordinates (x, y) of the corresponding region in the destination image.
   - r: Radius of the circular mask for keypoint extraction. If None, no mask is applied.
- Process:
   - Detect SIFT keypoints and descriptors for both source and destination images.
   - Apply circular mask if radius (r) is provided, focusing on a specific region.
   - Use FLANN (Fast Library for Approximate Nearest Neighbors) to find keypoint matches.
   - Apply Lowe's ratio test to filter out good matches.
   - Use RANSAC to compute the homography transformation matrix.
   - Retry with an increased radius if the number of keypoints is below a threshold.
- Output:
   - M: Homography transformation matrix (3x3) if successful, else None.
- Function: compute_norm_fixation
- Description: Compute normalized fixation coordinates after applying a homography transformation.
- Input:
   - x: X-coordinate of the original fixation point.
   - y: Y-coordinate of the original fixation point.
   - M: Homography transformation matrix (3x3).
- Process:
   - Create a homogeneous coordinate vector [x, y, 1].
   - Apply the homography transformation to obtain a new homogeneous coordinate vector.
   - Normalize the coordinates by dividing by the third element of the new homogeneous coordinate vector.
- Output:
   - Tuple containing the normalized Cartesian coordinates (x, y) of the fixation point.

## utils.py
- This script contains major utility functions, used across the segmentation aspect of the project.