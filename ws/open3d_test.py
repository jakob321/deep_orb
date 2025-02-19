import numpy as np
import open3d as o3d

# Create a sample point array (N x 3)
# Replace this with your actual point data.
num_points = 1000
points = np.random.rand(num_points, 3)

# Create a matching RGB array (N x 3) in the range [0, 1]
# Replace this with your actual RGB data.
colors = np.random.rand(num_points, 3)

# Create an Open3D PointCloud object
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
point_cloud.colors = o3d.utility.Vector3dVector(colors)

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])
