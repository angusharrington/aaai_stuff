import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def image_derivatives(img_path):
    img = cv.imread(img)
    ufilter = np.array([-1, 8, 0, -8, 1])/12
    vfilter = np.array([[-1], [8], [0], [-8], [1]])/12

    u_derivative = cv.filter2D(img,-1,ufilter)
    v_derivative = cv.filter2D(img,-1,vfilter)

    return u_derivative, v_derivative











