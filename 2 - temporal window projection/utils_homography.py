"""
Utilities for the homography computation
"""

import time
import numpy as np
import cv2
from matplotlib import pyplot as plt
import statistics as st

# Variable initialization
kp1_time_array = []
kp2_time_array = []
radius_array = []
matcher_array = []
homography_array = []
kp1_extracted_array = []
kp2_extracted_array = []
gp_extracted_array = []

###############################################################################################################################################
# Function: homography
# Description: Compute homography transformation matrix between two images using SIFT keypoints and descriptors.
# Input:
#   - src: Source image (numpy array).
#   - dst: Destination image (numpy array).
#   - center_src: Center coordinates (x, y) of the region of interest in the source image.
#   - center_dst: Center coordinates (x, y) of the corresponding region in the destination image.
#   - r: Radius of the circular mask for keypoint extraction. If None, no mask is applied.
# Process:
#   - Detect SIFT keypoints and descriptors for both source and destination images.
#   - Apply circular mask if radius (r) is provided, focusing on a specific region.
#   - Use FLANN (Fast Library for Approximate Nearest Neighbors) to find keypoint matches.
#   - Apply Lowe's ratio test to filter out good matches.
#   - Use RANSAC to compute the homography transformation matrix.
#   - Retry with an increased radius if the number of keypoints is below a threshold.
# Output:
#   - M: Homography transformation matrix (3x3) if successful, else None.
###############################################################################################################################################

def homography(src, dst, center_src, center_dst, r):
    height, width = src.shape[:2]

    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    radius_array.append(r)

    # find the keypoints and descriptors with SIFT
    if r is None:
        mine_start = time.time()
        kp1, des1 = sift.detectAndCompute(src, mask=None)
        mine_end = time.time()
        kp1_time_array.append((mine_end - mine_start) * 1000)
        mine_start = time.time()
        kp2, des2 = sift.detectAndCompute(dst, mask=None)
        mine_end = time.time()
        kp2_time_array.append((mine_end - mine_start) * 1000)

        # # Draw keypoints on the image
        # img_with_keypoints = cv2.drawKeypoints(src, kp1, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # img_with_keypoints2 = cv2.drawKeypoints(dst, kp2, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
        # # Plot the images with keypoints side-by-side
        # plt.figure(figsize=(15, 7))

        # # Plot the first image with keypoints
        # plt.subplot(1, 2, 1)
        # plt.imshow(img_with_keypoints, cmap='gray')
        # plt.title('Source Image Keypoints')
        # plt.axis('off')  # Hide axes for cleaner visualization

        # # Plot the second image with keypoints
        # plt.subplot(1, 2, 2)
        # plt.imshow(img_with_keypoints2, cmap='gray')
        # plt.title('Destination Image Keypoints')
        # plt.axis('off')  # Hide axes for cleaner visualization

        # plt.tight_layout()
        # plt.show()

    else:
        # compute the mask
        (x1, y1) = center_src
        M1 = np.zeros((height, width))
        cv2.circle(M1, (int(x1), int(y1)), r, (255, 255, 255), -1)
        M1 = np.uint8(M1)

        (x2, y2) = center_dst
        M2 = np.zeros((height, width))
        cv2.circle(M2, (int(x2), int(y2)), r, (255, 255, 255), -1)
        M2 = np.uint8(M2)

        mine_start = time.time()
        kp1, des1 = sift.detectAndCompute(src, mask=M1)
        mine_end = time.time()
        kp1_time_array.append((mine_end - mine_start) * 1000)
        mine_start = time.time()
        kp2, des2 = sift.detectAndCompute(dst, mask=M2)
        mine_end = time.time()
        kp2_time_array.append((mine_end - mine_start) * 1000)

        # Draw keypoints on both images with the mask applied
        img_with_keypoints1 = cv2.drawKeypoints(src, kp1, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img_with_keypoints2 = cv2.drawKeypoints(dst, kp2, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # # Plotting the images in a 2x2 grid
        # plt.figure(figsize=(12, 12))

        # # Plot the mask for the source image
        # plt.subplot(2, 2, 1)
        # plt.imshow(M1, cmap='gray')
        # plt.title('Source Image Mask')
        # plt.axis('off')

        # # Plot the keypoints on the source image
        # plt.subplot(2, 2, 2)
        # plt.imshow(img_with_keypoints1, cmap='gray')
        # plt.title('Source Image Keypoints with Mask')
        # plt.axis('off')

        # # Plot the mask for the destination image
        # plt.subplot(2, 2, 3)
        # plt.imshow(M2, cmap='gray')
        # plt.title('Destination Image Mask')
        # plt.axis('off')

        # # Plot the keypoints on the destination image
        # plt.subplot(2, 2, 4)
        # plt.imshow(img_with_keypoints2, cmap='gray')
        # plt.title('Destination Image Keypoints with Mask')
        # plt.axis('off')

        # plt.tight_layout()
        # plt.show()

    kp1_extracted_array.append(len(kp1))
    kp2_extracted_array.append(len(kp2))

    # print(f"The keypoints in source image {kp1_extracted_array}")
    # print(f"The keypoints in destination  image {kp2_extracted_array}")


    if len(kp1) < 100 or len(kp2) < 100:
        if r is None:
            return None
        if r < 300:
            return homography(src, dst, center_src, center_dst, r + 50)
        else:
            return homography(src, dst, center_src, center_dst, None)
    else:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        if des1 is None or des2 is None:
            if r is None:
                return None
            if r < 300:
                return homography(src, dst, center_src, center_dst, r + 50)
            else:
                return homography(src, dst, center_src, center_dst, None)
        else:
            mine_start = time.time()
            matches = flann.knnMatch(des1, des2, k=2)

            # store all the good matches as per Lowe's ratio test.
            good = []
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good.append(m)
            mine_end = time.time()
            matcher_array.append((mine_end - mine_start) * 1000)


            # Draw the good matches between the source and destination images
            img_matches = cv2.drawMatches(
                src, kp1, dst, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

            # # Plotting the matches
            # plt.figure(figsize=(15, 10))
            # plt.imshow(img_matches)
            # plt.title('Good Matches after Lowe\'s Ratio Test')
            # plt.axis('off')
            # plt.show()

            # compute the transformation matrix
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            mine_start = time.time()

            if src_pts.shape[0] < 4 or dst_pts.shape[0] < 4:
                M = np.eye(3)
            else:
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            mine_end = time.time()
            homography_array.append((mine_end - mine_start) * 1000)

            if M is None:
                if r is None:
                    return None
                if r < 300:
                    return homography(src, dst, center_src, center_dst, r + 50)
                else:
                    return homography(src, dst, center_src, center_dst, None)

    return M

def homography_fast(src, dst, center_src, center_dst, r):
    height, width = src.shape[:2]

    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    radius_array.append(r)

    # find the keypoints and descriptors with SIFT
    if r is None:
        mine_start = time.time()
        fast = cv2.FastFeatureDetector_create()
        kp1 = fast.detect(src, None)
        mine_end = time.time()
        kp1_time_array.append((mine_end - mine_start) * 1000)
        mine_start = time.time()
        fast = cv2.FastFeatureDetector_create()
        kp2 = fast.detect(dst, None)
        mine_end = time.time()
        kp2_time_array.append((mine_end - mine_start) * 1000)
    else:
        # compute the mask
        (x1, y1) = center_src
        M1 = np.zeros((height, width))
        cv2.circle(M1, (int(x1), int(y1)), r, (255, 255, 255), -1)
        M1 = np.uint8(M1)

        (x2, y2) = center_dst
        M2 = np.zeros((height, width))
        cv2.circle(M2, (int(x2), int(y2)), r, (255, 255, 255), -1)
        M2 = np.uint8(M2)

        mine_start = time.time()
        fast = cv2.FastFeatureDetector_create()
        kp1 = fast.detect(src, mask=M1)
        mine_end = time.time()
        kp1_time_array.append((mine_end - mine_start) * 1000)
        mine_start = time.time()
        fast = cv2.FastFeatureDetector_create()
        kp2 = fast.detect(dst, mask=M2)
        mine_end = time.time()
        kp2_time_array.append((mine_end - mine_start) * 1000)

    kp1_extracted_array.append(len(kp1))
    kp2_extracted_array.append(len(kp2))

    if len(kp1) < 100 or len(kp2) < 100:
        if r is None:
            return None
        if r < 300:
            return homography(src, dst, center_src, center_dst, r + 50)
        else:
            return homography(src, dst, center_src, center_dst, None)
    else:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        if des1 is None or des2 is None:
            if r is None:
                return None
            if r < 300:
                return homography(src, dst, center_src, center_dst, r + 50)
            else:
                return homography(src, dst, center_src, center_dst, None)
        else:
            mine_start = time.time()
            matches = flann.knnMatch(des1, des2, k=2)
            # store all the good matches as per Lowe's ratio test.
            good = []
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good.append(m)
            mine_end = time.time()
            matcher_array.append((mine_end - mine_start) * 1000)

            # compute the transformation matrix
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            mine_start = time.time()
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            mine_end = time.time()
            homography_array.append((mine_end - mine_start) * 1000)

            if M is None:
                if r is None:
                    return None
                if r < 300:
                    return homography(src, dst, center_src, center_dst, r + 50)
                else:
                    return homography(src, dst, center_src, center_dst, None)

    return M

###############################################################################################################################################
# Function: compute_norm_fixation
# Description: Compute normalized fixation coordinates after applying a homography transformation.
# Input:
#   - x: X-coordinate of the original fixation point.
#   - y: Y-coordinate of the original fixation point.
#   - M: Homography transformation matrix (3x3).
# Process:
#   - Create a homogeneous coordinate vector [x, y, 1].
#   - Apply the homography transformation to obtain a new homogeneous coordinate vector.
#   - Normalize the coordinates by dividing by the third element of the new homogeneous coordinate vector.
# Output:
#   - Tuple containing the normalized Cartesian coordinates (x, y) of the fixation point.
###############################################################################################################################################
def project_fixation(x, y, M):
    # Create a homogeneous coordinate vector [x, y, 1]
    homog_coor = np.asarray([x, y, 1])

    # Apply the homography transformation
    new_homoh_coor = M.dot(homog_coor)

    # Normalize the coordinates
    cart = [new_homoh_coor.item(0) / new_homoh_coor.item(2), new_homoh_coor.item(1) / new_homoh_coor.item(2)]

    # Return the normalized Cartesian coordinates
    return (cart[0], cart[1])