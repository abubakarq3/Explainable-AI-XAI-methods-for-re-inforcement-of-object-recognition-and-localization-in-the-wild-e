import os
import json
import cv2
import numpy as np
from scipy import interpolate
import skimage.io
import skimage.transform
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pdb
import skvideo.io

def generate_annotation_txt(datasetDir, scale_reduction=1.0):
    # Checking Directories
    annotationFile = os.path.join(datasetDir, 'annotation.txt')
    if os.path.exists(annotationFile):
        print("Annotation file already exists")

    video_name = datasetDir.split("/")[-1]
    videoFile = os.path.join(datasetDir, video_name + '.mp4')
    if os.path.exists(videoFile):
        print(f'Processing {datasetDir.split("/")[-1]} video')

    # Checking and creating directory for the frames
    framesOutDir = os.path.join(datasetDir, 'Frames')
    if not os.path.exists(framesOutDir):
        os.makedirs(framesOutDir)

    # Extracting frames from the video file
    cap = cv2.VideoCapture(videoFile)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    nFrames = len(frames)
    height = frames[0].shape[0]
    width = frames[0].shape[1]
    print(f'Number of frames: {nFrames}')
    print(f'Frame resolution: {height}x{width}')

    # Optional scale reduction
    height /= scale_reduction
    width /= scale_reduction

    # Saving frames as JPEG files
    for fr in range(nFrames):
        frt = fr * 40
        skimage.io.imsave("%s/Frame_%d.jpg" % (framesOutDir, frt), frames[fr])

    # Loading the fixation file
    jsonFile = os.path.join(datasetDir, video_name + '.json')

    if not os.path.exists(jsonFile):
        print(f"Error: No file {jsonFile}")

    contFix = 0
    gazePoints = 0

    # Count the gaze points
    with open(jsonFile) as f:
        # First count the number of gaze points
        for line in f:
            data = json.loads(line)
            if 'gp' in data and data['s'] == 0:
                gazePoints += 1

    # Read the gaze points
    start = 0
    fixdata = np.zeros((gazePoints, 3))
    with open(jsonFile) as f:
        for line in f:
            data = json.loads(line)
            if 'gp' in data and data['s'] == 0:
                fixdata[contFix, 0] = data['ts']
                fixdata[contFix, 1] = data['gp'][0]
                fixdata[contFix, 2] = data['gp'][1]
                contFix += 1

            if 'vts' in data and data['s'] == 0:
                if data['vts'] == 0:
                    start = data['ts']

    if start == 0:
        print("Error, no vts=0 found")

    # Changing fixation frame rate from 50Hz to match video frame rate (25Hz)
    # Interpolating the fixation values.
    fr_instants = np.arange(start, start + nFrames * 40000, 40000)

    # We need to interpolate the fixations using splines interpolation
    tck_x = interpolate.splrep(fixdata[..., 0], fixdata[..., 1], s=0.00)
    tck_y = interpolate.splrep(fixdata[..., 0], fixdata[..., 2], s=0.00)
    xfr = interpolate.splev(fr_instants, tck_x, der=0)
    yfr = interpolate.splev(fr_instants, tck_y, der=0)

    # Normalizing the fixations to the frame resolution
    fixations = np.zeros((nFrames, 2), dtype=np.int64)
    for fr in range(nFrames):
        frt = fr * 40
        fixations[fr, 0] = int(width * xfr[fr])
        fixations[fr, 1] = int(height * yfr[fr])

    print(fixations[0])

    # Generate the final annotation
    # BoundingBox File
    bbFile = os.path.join(datasetDir, 'bounding_box.txt')
    if not os.path.exists(bbFile):
        print(f"Error: No file {bbFile}")

    bbf = open(bbFile, "r")

    # Get the segment
    initf = 0
    endf = 0
    fr = 0
    for line in bbf:
        data = line.split()
        bb = np.array(data[1:], dtype='|S4')
        bb = bb.astype(np.float64)
        # If the bounding box is 0 0 0 0, then consider it as negative
        if np.sum(bb) > 0:
            if initf == 0:
                initf = fr
            else:
                endf = fr
        fr += 1
    bbf.close()

    # Open the AnnotationFile for writing (fixating + grasping the object)
    af = open(annotationFile, "w")
    bbf = open(bbFile, "r")
    # print(f'BBF Length: {len(bbf)}')
    print(f'fixations shape: {fixations.shape}')
    fr = 0
    for line in bbf:
        data = line.split()
        # print(f'Data length: {len(data)}')
        bb = np.array(data[1:], dtype='|S4')
        bb = bb.astype(np.float64)
        # If frame is within the determined segment, label as 1, else 0
        if initf <= fr <= endf:
            label = 1
        else:
            label = 0
        
        outs = f"{data[0][0:-4]} {label} {fixations[fr, 0]} {fixations[fr, 1]}\n"
        af.write(outs)
        fr += 1

    bbf.close()
    af.close()
    print("Annotation file generated")

    # To Plot the fixations on image 110
    # plt.figure()
    # plt.imshow(frames[110])
    # plt.plot(fixations[110, 0], fixations[110, 1], 'r.')
    # plt.savefig('output.png')  # replace with your desired output file
    # plt.show()


def generate_annotation_txt_with_z(datasetDir, scale_reduction=1.0):
    # Checking Directories
    annotationFile = os.path.join(datasetDir, 'annotation_with_z.txt')
    if os.path.exists(annotationFile):
        print("Annotation file already exists")

    video_name = datasetDir.split("/")[-1]
    videoFile = os.path.join(datasetDir, video_name + '.mp4')
    if os.path.exists(videoFile):
        print(f'Processing {datasetDir.split("/")[-1]} video')

    # Checking and creating directory for the frames
    framesOutDir = os.path.join(datasetDir, 'Frames')
    if not os.path.exists(framesOutDir):
        os.makedirs(framesOutDir)

    # Extracting frames from the video file
    cap = cv2.VideoCapture(videoFile)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    nFrames = len(frames)
    height = frames[0].shape[0]
    width = frames[0].shape[1]
    print(f'Number of frames: {nFrames}')
    print(f'Frame resolution: {height}x{width}')

    # Optional scale reduction
    height /= scale_reduction
    width /= scale_reduction

    # Saving frames as JPEG files
    for fr in range(nFrames):
        frt = fr * 40
        skimage.io.imsave("%s/Frame_%d.jpg" % (framesOutDir, frt), frames[fr])

    # Loading the fixation file
    jsonFile = os.path.join(datasetDir, video_name + '.json')

    if not os.path.exists(jsonFile):
        print(f"Error: No file {jsonFile}")

    contFix = 0
    gazePoints = 0

    # Count the gaze points
    with open(jsonFile) as f:
        # First count the number of gaze points
        for line in f:
            data = json.loads(line)
            if 'gp' in data and data['s'] == 0:
                gazePoints += 1

    # Read the gaze points
    start = 0
    fixdata = np.zeros((gazePoints, 4))
    with open(jsonFile) as f:
        for line in f:
            data = json.loads(line)
            if 'gp' in data and data['s'] == 0:
                fixdata[contFix, 0] = data['ts']
                fixdata[contFix, 1] = data['gp'][0]
                fixdata[contFix, 2] = data['gp'][1]
                

            if 'gp3' in data and data['s'] == 0:
                # print(data['gp3'])
                fixdata[contFix, 3] = data['gp3'][2]
                contFix += 1


            if 'vts' in data and data['s'] == 0:
                if data['vts'] == 0:
                    start = data['ts']

    if start == 0:
        print("Error, no vts=0 found")

    # Changing fixation frame rate from 50Hz to match video frame rate (25Hz)
    # Interpolating the fixation values.
    fr_instants = np.arange(start, start + nFrames * 40000, 40000)

    # We need to interpolate the fixations using splines interpolation
    tck_x = interpolate.splrep(fixdata[..., 0], fixdata[..., 1], s=0.00)
    tck_y = interpolate.splrep(fixdata[..., 0], fixdata[..., 2], s=0.00)
    tck_z = interpolate.splrep(fixdata[..., 0], fixdata[..., 3], s=0.00)
    xfr = interpolate.splev(fr_instants, tck_x, der=0)
    yfr = interpolate.splev(fr_instants, tck_y, der=0)
    zfr = interpolate.splev(fr_instants, tck_z, der=0)

    # Normalizing the fixations to the frame resolution
    fixations = np.zeros((nFrames, 3), dtype=np.int64)
    for fr in range(nFrames):
        frt = fr * 40
        fixations[fr, 0] = int(width * xfr[fr])
        fixations[fr, 1] = int(height * yfr[fr])
        if zfr[fr] < 1:
            fixations[fr, 2] = 1
        else:
            fixations[fr, 2] = int(zfr[fr])


    print(fixations[0])

    # Generate the final annotation
    # BoundingBox File
    bbFile = os.path.join(datasetDir, 'bounding_box.txt')
    if not os.path.exists(bbFile):
        print(f"Error: No file {bbFile}")

    bbf = open(bbFile, "r")

    # Get the segment
    initf = 0
    endf = 0
    fr = 0
    for line in bbf:
        data = line.split()
        bb = np.array(data[1:], dtype='|S4')
        bb = bb.astype(np.float64)
        # If the bounding box is 0 0 0 0, then consider it as negative
        if np.sum(bb) > 0:
            if initf == 0:
                initf = fr
            else:
                endf = fr
        fr += 1
    bbf.close()

    # Open the AnnotationFile for writing (fixating + grasping the object)
    af = open(annotationFile, "w")
    bbf = open(bbFile, "r")
    fr = 0
    print(f'fixations shape: {fixations.shape}')
    for line in bbf:
        data = line.split()
        bb = np.array(data[1:], dtype='|S4')
        bb = bb.astype(np.float64)
        # If frame is within the determined segment, label as 1, else 0
        if initf <= fr <= endf:
            label = 1
        else:
            label = 0
        outs = f"{data[0][0:-4]} {label} {fixations[fr, 0]} {fixations[fr, 1]} {fixations[fr, 2]}\n"
        af.write(outs)
        fr += 1

    bbf.close()
    af.close()
    print("Annotation file generated")
