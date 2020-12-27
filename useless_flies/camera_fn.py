import numpy as np
import cv2 as cv
import image_lanmarks
import pyrender
import matplotlib.pyplot as plt
import trimesh

def alg_1(img, avg_face_mesh):

    light_est, shape_est = InitialLightingAndShapeEstimation(img)
    size = img.shape()

    def img_renderer(img, mesh, light_est):


        # compose scene
        scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[0, 0, 0])
        camera = pyrender.PerspectiveCamera( yfov=np.pi / 3.0)
        light = pyrender.DirectionalLight(color=[1,1,1], intensity=2e3)

        scene.add(mesh, pose=  np.eye(4))
        scene.add(light_est, pose=  np.eye(4))

        # c = 2**-0.5
        # scene.add(camera, pose=[[ 1,  0,  0,  0],
        #                [ 0,  c, -c, -2],
        #                [ 0,  c,  c,  2],
        #                [ 0,  0,  0,  1]])

        # render scene
        r = pyrender.OffscreenRenderer(512, 512)
        color, _ = r.render(scene)

        plt.figure(figsize=(8,8)), plt.imshow(color);
        
    # 2D image points
    nose = image_lanmarks.nose_point

    # 3D image points (model point)

    # Camera
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)

    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype = "double")

    dist_coeffs = np.zeros((4,1)) 
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.CV_ITERATIVE)
    
















