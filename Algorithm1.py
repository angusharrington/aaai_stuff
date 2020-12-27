#%%

import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import get_fiducial_points
import open3d as o3d
import optcal_flow
import image_renderer
import UpsamplePtCloud


def alg_1(img):

    image = cv2.imread(img)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert image to grayscale

    size = image.shape
    #templateModel = o3d.io.read_point_cloud("data/template.ply")
    pcd = o3d.io.read_point_cloud("Data/face_mesh_000306.ply")

    # estimate radius for rolling ball
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist

    average_face_mesh = o3d.io.read_triangle_mesh('Data/face_mesh_000306.ply')
    # 2D image points
    image_points = optcal_flow.optical_flow(img, image_renderer.img_renderer(average_face_mesh))

    # 3D object points (model point)

    def process_template(templatePoints, templateCoordinates):
        """This function recentres the point cloud about the min value in each axis.
        Also applies the same transformations to the point cloud fiducial points.
        :param templatePoints: object that stores information about the point cloud
        :type templatePoints: open3d Point Cloud object
        :param templateCoordinates:
        :type templateCoordinates:
        """

        # get the min value in each axis
        p1 = min(templatePoints[:, 0])
        p2 = min(templatePoints[:, 1])
        p3 = min(templatePoints[:, 2])

        # subtract the mean
        templatePoints[:, 0] -= p1
        templatePoints[:, 1] -= p2
        templatePoints[:, 2] -= p3

        templateCoordinates[:, 0] -= p1
        templateCoordinates[:, 1] -= p2
        templateCoordinates[:, 2] -= p3

        # divide by two. Why? -> maybe to reduce the range of values??
        templatePoints[:, 0] /= 2
        templatePoints[:, 1] /= 2
        templatePoints[:, 2] /= 2

        templateCoordinates[:, 0] /= 2
        templateCoordinates[:, 1] /= 2
        templateCoordinates[:, 2] /= 2

        return templatePoints, templateCoordinates

    templateCoordinates = np.array([[34.9, 2.458, 1230],  # right eye right
                                        [37, -31.89, 1236],  # right eye left
                                        [45.4, -66.51, 1248],  # left eye right
                                        [41.24, -94.92, 1267],  # left eye left
                                        [82.49, -29.01, 1227],  # nose right
                                        [80.5, -53.6, 1212],  # nose tip
                                        [79.03, -71.2, 1240],  # nose left
                                        [105.5, -23.84, 1228],  # mouth right
                                        [111.5, -72.14, 1251]  # mouth left
                                        ])

    #templateModel = o3d.io.read_point_cloud("data/template.ply")
    ptCloud = o3d.io.read_point_cloud("/Users/angusharrington/Documents/AEK_total_moving_faces/3D/perfect_dubbing_v1/3D_flow/data/template.ply")

    templatePoints = UpsamplePtCloud(ptCloud)

    templatePoints, templateCoordinates = process_template(templatePoints, templateCoordinates)

    model_points = o3d.find3dpoints(templateCoordinates)

    # Camera
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)

    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype = "double")

    dist_coeffs = np.zeros((4,1))
    iterations_count = 50
    reprojection_error = 0.1
    min_inliers_count = 400


    '''Python: cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs[, rvec[, tvec[, useExtrinsicGuess[,
                         flags]]]]) → retval, rvec, tvec'''
    # Solve PnP Ransac(but there are only nine points so is this necessary?)
    (rotation_vector1, translation_vector1, inlier_indices) = cv2.solvePnPRansac(model_points, image_points, camera_matrix, dist_coeffs,
                       iterations_count, reprojection_error, min_inliers_count, flags=cv2.CV_ITERATIVE)
    inliers_object = [model_points[i] for i in inlier_indices]
    inliers_image = [image_points[i] for i in inlier_indices]
    # Solve Pnp on all inliers
    (success, rotation_vector, translation_vector) = cv2.solvePnP(inliers_object, inliers_image, camera_matrix, dist_coeffs, flags=cv2.CV_ITERATIVE)
    return print(success, rotation_vector, translation_vector)
    
#%%
alg_1('Data/g_bush_01.jpg')







# %%
