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
    templateCoordinates = templateFiducialPoints1

    ptCloud = open3d.io.read_point_cloud(templateModel)

    templatePoints = UpsamplePtCloud(ptCloud)

    templatePoints, templateCoordinates = process_template(templatePoints, templateCoordinates)

    model_points = templateCoordinates

    # Camera
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)

    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype = "double")

    dist_coeffs = np.zeros((4,1)) 

    # Solve PnP Ransac on 3 fiducials
    (rotation_vector1, translation_vector1, inlier_indices) = cv2.solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs[, rvec[, tvec[, useExtrinsicGuess[,
                       iterationsCount[, reprojectionError[, minInliersCount[, inliers[, flags]]]]]]]])

    # Solve Pnp on all inliers
     (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.CV_ITERATIVE)















