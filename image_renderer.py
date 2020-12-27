#%%
import pyrender
import numpy as np
import open3d as o3d
import trimesh
import pyglet
import matplotlib.pyplot as plt

def img_renderer(pointcloud):

    pcd = o3d.io.read_point_cloud(pointcloud)
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    pcd.estimate_normals()

    # estimate radius for rolling ball
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 5 * avg_dist   

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
           pcd,
           o3d.utility.DoubleVector([radius, radius * 2]))

    # create the triangular mesh with the vertices and faces from open3d
    tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                          vertex_normals=np.asarray(mesh.vertex_normals))

    #tri_mesh = trimesh.convex.is_convex(tri_mesh)

    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    s = np.sqrt(2)/2
    camera_pose = np.array([
    [0.0, -s,   s,   0.3],
    [1.0,  0.0, 0.0, 0.0],
    [0.0,  s,   s,   0.35],
    [0.0,  0.0, 0.0, 1.0], ])
    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                            innerConeAngle=np.pi/16.0,
                            outerConeAngle=np.pi/6.0)
    scene.add(light, pose=camera_pose)
    pyrender.Viewer(scene)

    r = pyrender.OffscreenRenderer(512, 512)
    color, _ = r.render(scene)

    plt.figure(figsize=(8,8))
    plt.imshow(color)
#%%
img_renderer(r'C:\Users\angus\OneDrive\Documents\AAAi\perfect_dubbing_v1\perfect_dubbing_v1\3D_flow\data\face_mesh_000306.ply')


# %%
