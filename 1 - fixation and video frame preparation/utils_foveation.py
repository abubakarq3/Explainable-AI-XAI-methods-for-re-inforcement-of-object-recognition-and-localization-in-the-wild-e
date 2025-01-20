"""
Utility functions for computing focal areas and foveation
"""

import os
import numpy as np
import math
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

def compute_sigma_mm(z, alpha):
    """
    Computes the standard deviation (sigma) in millimeters based on the depth (z) and angle (alpha).

    Parameters:
    - z (float): Depth or distance.
    - alpha (float): Angle in degrees.

    Returns:
    - float: Standard deviation in millimeters.
    """
    alpha = np.deg2rad(alpha)

    tan_alpha = np.tan(alpha)
    print(tan_alpha)
    return z * np.tan(alpha)


def compute_fov_mm(z, cam_fov_x, cam_fov_y):
    """
    Computes the field of view in millimeters based on depth (z) and camera's horizontal and vertical field of view.

    Parameters:
    - z (float): Depth or distance.
    - cam_fov_x (float): Camera's horizontal field of view in degrees.
    - cam_fov_y (float): Camera's vertical field of view in degrees.

    Returns:
    - tuple: Field of view in millimeters (fov_x_mm, fov_y_mm).
    """
    fov_x = np.deg2rad(cam_fov_x)
    fov_y = np.deg2rad(cam_fov_y)

    fov_x_mm = 2 * z * np.tan(fov_x / 2)
    fov_y_mm = 2 * z * np.tan(fov_y / 2)

    return fov_x_mm, fov_y_mm


def compute_sigma_pixels(z, alpha, width, height, cam_fov_x, cam_fov_y):
    """
    Computes the standard deviation in pixels based on depth (z), angle (alpha), image dimensions, and camera's field of view.

    Parameters:
    - z (float): Depth or distance.
    - alpha (float): Angle in degrees.
    - width (int): Image width in pixels.
    - height (int): Image height in pixels.
    - cam_fov_x (float): Camera's horizontal field of view in degrees.
    - cam_fov_y (float): Camera's vertical field of view in degrees.

    Returns:
    - tuple: Standard deviation in pixels (sigma_x, sigma_y).
    """
    fov_x_mm, fov_y_mm = compute_fov_mm(z, cam_fov_x, cam_fov_y)

    # print(f'fov_x_mm: {fov_x_mm}, fov_y_mm: {fov_y_mm}')

    sigma_x_mm = compute_sigma_mm(z, alpha)
    sigma_y_mm = compute_sigma_mm(z, alpha)

    pix_x = width / fov_x_mm
    pix_y = height / fov_y_mm

    # print(f'pix_x: {pix_x}, pix_y: {pix_y}')

    sigma_x = sigma_x_mm * pix_x
    sigma_y = sigma_y_mm * pix_y

    return sigma_x, sigma_y


def compute_saliency_map(image, fixation_point, sigma_x, sigma_y, A=1):
    """
    Computes the saliency map based on the input image, fixation point, and standard deviations in x and y.

    Parameters:
    - image (numpy.ndarray): Input image.
    - fixation_point (tuple): Coordinates of the fixation point (x, y).
    - sigma_x (float): Standard deviation in the x-direction.
    - sigma_y (float): Standard deviation in the y-direction.
    - A (float): Amplitude parameter (default=1).

    Returns:
    - numpy.ndarray: Saliency map.
    """
    h, w, _ = image.shape
    
    print(f"the size of the image{h,w}")
    x_coords, y_coords = np.mgrid[0:w, 0:h]
    coords = np.vstack([x_coords.ravel(), y_coords.ravel()])

    saliency_map = A * np.exp(-((coords[0] - fixation_point[0])**2 / (2 * sigma_x**2)) - ((coords[1] - fixation_point[1])**2 / (2 * sigma_y**2)))
    saliency_map = saliency_map.reshape(w, h)

    return saliency_map

def gfdm_to_pil(gfdm):
    assert np.min(gfdm) >= 0.0 and np.max(gfdm) <= 1.0
    img = (gfdm * 255).astype(np.uint8)
    img = Image.fromarray(img)
    return img

def represent_heatmap(saliency, cmap):
    heatmap_colored = np.asarray(cmap(saliency))[:, :, :3]
    heatmap_img = Image.fromarray((heatmap_colored * 255).astype(np.uint8))
    return heatmap_img

def represent_overlaid_heatmap(saliency, image, cmap):
    heat = represent_heatmap(saliency, cmap)
    return Image.blend(
            image, 
            heat.convert(image.mode), 
            0.6
        )
 

def main(pathfordata):

    path_for_image = os.path.join(pathfordata, "Frames")
    path_for_txt= os.path.join(pathfordata, "annotation_with_z.txt")
    
    savesaliencypath = os.path.join(pathfordata, "saliencyframes")
    # Check if the directory exists
    if not os.path.exists(savesaliencypath):
        # Create the directory and any necessary parent directories
        os.makedirs(savesaliencypath)
        print(f"Directory created: {savesaliencypath}")
    else:
        print(f"Directory already exists: {savesaliencypath}")

    Frames = os.listdir(path_for_image)

    with open(path_for_txt, 'r') as file:
        lines = file.readlines()

    # Process and print each line
    for line in lines:
        input_string = line.strip()
        filename = input_string.split()[-5] + ".jpg"
        fx = int(input_string.split()[-3])
        fy = int(input_string.split()[-2])
        z = int(input_string.split()[-1])
        
        imgpath = os.path.join(path_for_image,filename)

        print(f"The fixation point is ({fx}, {fy}) and the z-value is {z}")

        # Compute sigma values
        sigma_x, sigma_y = compute_sigma_pixels(z, 2.0, 1920, 1080, 82, 52)

        # Load and convert the image to RGB
        image = cv2.imread(imgpath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        
        image_pil = Image.open(imgpath)
        image_array = np.array(image_pil)
        
        # Compute the saliency map
        saliency = compute_saliency_map(image, [fx, fy], sigma_x, sigma_y, A=1)

        gfdm_pil = gfdm_to_pil(saliency)
        gfdm_pil_swapped = gfdm_pil.transpose(Image.TRANSPOSE)      

        cmapped = represent_overlaid_heatmap(gfdm_pil_swapped, image_pil , cmap=plt.get_cmap("jet"))
        cmapped.save(os.path.join(savesaliencypath,filename) +".png", "PNG")
        
"""

if __name__ == '__main__':
    path_for_txt= "/net/travail/rramesh/TRDP/Dataset/Bowl/BowlPlace1Subject1/annotation_with_z.txt"

    with open(path_for_txt, 'r') as file:
        lines = file.readlines()

    z_new= []
    sigmax = []
    sigmay = []
    # Process and print each line
    for line in lines:
        input_string = line.strip()
        filename = input_string.split()[-5] + ".jpg"
        fx = int(input_string.split()[-3])
        fy = int(input_string.split()[-2])
        z = int(input_string.split()[-1]) 

        z_new.append(z)
        sigma_x, sigma_y = compute_sigma_pixels(z, 2.0, 1920, 1080, 82, 52)

        sigmax.append(sigma_x)
        sigmay.append(sigma_y)



    print(sigmax)
    print(sigmay)    

    # print(z_new) 

    time = range(len(z_new))

    

    # # Create the plot
    # plt.figure(figsize=(12, 6))
    # plt.plot(time, z_new, marker='o')
    # plt.title('Plotting the Z Value for bowlplace1subject1')
    # plt.xlabel("Frame Number")
    # plt.ylabel('Depth or distance (mm)')
    # plt.grid()
    # plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(time, sigmax, marker='o',color = 'blue')
    #plt.plot(time, sigmay, marker='o',color = 'orange')
    plt.title('Plotting the sigma Value for bowlplace1subject1')
    plt.xlabel("Frame Number")
    plt.ylabel('sigma x  (mm)')
    plt.legend()
    plt.grid()
    plt.show()
     
""" 

"""
    # Split the input string and extract fixation points and z-value
    fx = int(input_string.split()[-3])
    fy = int(input_string.split()[-2])
    z = int(input_string.split()[-1])
"""
    


"""
    print(f"The fixation point is ({fx}, {fy}) and the z-value is {z}")

    # Compute sigma values
    sigma_x, sigma_y = compute_sigma_pixels(z, 2.0, 1920, 1080, 82, 52)

    # Load and convert the image to RGB
    image = cv2.imread(path_for_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    image_pil = Image.open(path_for_image)
    image_array = np.array(image_pil)
    
    # Compute the saliency map
    saliency = compute_saliency_map(image, [fx, fy], sigma_x, sigma_y, A=1)

    gfdm_pil = gfdm_to_pil(saliency)
    gfdm_pil.save("testsal.png", "PNG")
 
    gfdm_pil_swapped = gfdm_pil.transpose(Image.TRANSPOSE)      

    cmapped = represent_overlaid_heatmap(gfdm_pil_swapped, image_pil , cmap=plt.get_cmap("jet"))
    cmapped.save("ipose.png", "PNG")
    

    # Plot the original image and saliency map
    plt.figure(figsize=(12, 6))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis("off")

    # Saliency Map
    plt.subplot(1, 2, 2)
    plt.title("Saliency Map")
    plt.imshow(saliency, cmap='hot')  # Using 'hot' colormap for heatmap-like visualization
    plt.axis("off")

    # Display the images
    plt.tight_layout()
    plt.show() """