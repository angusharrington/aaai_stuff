import numpy as np
import cv2 as cv2
import image_lanmarks
import pyrender
import matplotlib.pyplot as plt
import trimesh
import optcal_flow
import open3d as o3d


def alg_1(img, avg_face_mesh):

    light_est, shape_est = InitialLightingAndShapeEstimation(img)
    size = img.shape()
    templateModel = o3d.io.read_point_cloud("Data/face_mesh_000306.ply")
    pcd = o3d.io.read_point_cloud(templateModel)
    radii =
    avg_face_mesh = o3d.create_from_point_cloud_ball_pivoting(pcd, radii)

    # 2D image points
    image_points = optcal_flow.optical_flow(img, img_renderer(img, avg_face_mesh, light_est))

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

    templateCoordinates = np.array([[35.53, -428.7, 1152],
                                        [39.07, -456.8, 1148],
                                        [42.61, -495, 1148],
                                        [39.77, -528, 1154],
                                        [85.22, -456.7, 1138],
                                        [85.77, -475.1, 1122],
                                        [88.47, -490.1, 1136],
                                        [107.7, -448.1, 1147],
                                        [107.8, -499.6, 1144]])

    templateModel = o3d.io.read_point_cloud("Data/face_mesh_000306.ply")
    ptCloud = o3d.io.read_point_cloud(templateModel)

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
    inliers_object =
    inliers_image =
    # Solve Pnp on all inliers
     (success, rotation_vector, translation_vector) = cv2.solvePnP(inliers_object, inliers_image, camera_matrix, dist_coeffs, flags=cv2.CV_ITERATIVE)








