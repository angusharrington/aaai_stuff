import numpy as np
import cv2 as cv
import image_lanmarks
import pyrender
import matplotlib.pyplot as plt
import trimesh
import optcal_flow


def alg_1(img, avg_face_mesh):

    light_est, shape_est = InitialLightingAndShapeEstimation(img)
    size = img.shape()

    # 2D image points
    image_points = optcal_flow.optical_flow(img, img_renderer(img, avg_face_mesh, light_est))

    # 3D object points (model point)
    '''templateCoordinates = templateFiducialPoints1

    ptCloud = open3d.io.read_point_cloud(templateModel)

    templatePoints = UpsamplePtCloud(ptCloud)

    templatePoints, templateCoordinates = process_template(templatePoints, templateCoordinates)

    model_points = templateCoordinates'''
    model_points = np.array([[35.53, -428.7, 1152],
                                        [39.07, -456.8, 1148],
                                        [42.61, -495, 1148],
                                        [39.77, -528, 1154],
                                        [85.22, -456.7, 1138],
                                        [85.77, -475.1, 1122],
                                        [88.47, -490.1, 1136],
                                        [107.7, -448.1, 1147],
                                        [107.8, -499.6, 1144]])

    # Camera
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)

    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype = "double")

    dist_coeffs = np.zeros((4,1))

    # Solve PnP Ransac on 9 fiducials
    (rotation_vector1, translation_vector1, inlier_indices) = cv2.solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs[, rvec[, tvec[, useExtrinsicGuess[,
                       iterationsCount[, reprojectionError[, minInliersCount[, inliers[, flags]]]]]]]])

    # Solve Pnp on all inliers
     (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.CV_ITERATIVE)








