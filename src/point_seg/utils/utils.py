## The code below is "ported" from 
# https://github.com/ros/common_msgs/tree/noetic-devel/sensor_msgs/src/sensor_msgs
# I'll make an official port and PR to this repo later: 
# https://github.com/ros2/common_interfaces
import sys
from collections import namedtuple
import ctypes
import math
import struct
from sensor_msgs.msg import PointCloud2, PointField
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt

def scanclustering(pcd):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Info) as cm:labels = np.array(
        pcd.cluster_dbscan(eps=0.1, min_points=5, print_progress=False))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    # o3d.visualization.draw_geometries([pcd],
    #                                 zoom=0.455,
    #                                 front=[-0.4999, -0.1659, -0.8499],
    #                                 lookat=[2.1813, 2.0619, 2.0999],
    #                                 up=[0.1204, -0.9852, 0.1215])
    return pcd