import numpy as np
import cv2
import get_fiducial_points

def optical_flow(input_rendered, pose_image):
    # params for corner detection
    feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                           10, 0.03))

    # Take greyscaled rendered average mesh and find corners in it
    pose_grey = cv2.cvtColor(pose_image,
                        cv2.COLOR_BGR2GRAY)

    '''
    p0 = cv2.goodFeaturesToTrack(pose_grey, mask=None,
                             **feature_params
    '''
    p0 = get_fiducial_points.getFiducialPoints(input_rendered, 'Data/shape_predictor_68_face_landmarks.dat')

    # Take greyscale of input frame
    input_grey = cv2.cvtColor(input_rendered,
                              cv2.COLOR_BGR2GRAY)


    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(pose_grey,
                input_grey, p0, None,**lk_params)
    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    return p1



