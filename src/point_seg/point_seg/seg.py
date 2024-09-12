import rclpy
from rclpy.node import Node
import numpy as np
import open3d as o3d
import random
import ros2_numpy
import threading
import time
from shape_msgs.msg import Mesh, MeshTriangle
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2


class PointCloudSubscriber(Node):
    def __init__(self):
        super().__init__('pointcloud_subscriber')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/lidar_points',  # Replace with your point cloud topic
            self.listener_callback,
            10
        )
        self.subscription
        self.point_cloud_data = None

    def listener_callback(self, msg):
        # Read the point cloud data
        # self.point_cloud_data = pc2.read_points(msg, field_names=['x', 'y', 'z', "intensity"], skip_nans=True)
        pc = ros2_numpy.numpify(msg)
        print(pc['xyz'].shape)
        # self.points=np.zeros((pc.shape[0],3))
        # self.points[:,0]=pc['x']
        # self.points[:,1]=pc['y']
        # self.points[:,2]=pc['z']
        self.points=pc['xyz']
        self.process_point_cloud()

    def process_point_cloud(self):
        if self.points is None:
            return

        # Convert to numpy array and ensure dtype is float64
        # print(self.points.shape)
        point_cloud_np = np.array(self.points, dtype=np.float64)
        if point_cloud_np.shape[0] == 0:
            return

        # Create an Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_np)
        downpcd = pcd.voxel_down_sample(voxel_size=0.05)
        print(downpcd)
        downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # using all defaults
        oboxes = downpcd.detect_planar_patches(
            normal_variance_threshold_deg=60,
            coplanarity_deg=75,
            outlier_ratio=0.25,
            min_plane_edge_length=0,
            min_num_points=0,
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

        print("Detected {} patches".format(len(oboxes)))

        geometries = []
        plane_normals = []
        combined_points = []
        for obox in oboxes:
            mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox, scale=[1, 1, 0.0000001])
            mesh.paint_uniform_color(obox.color)
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            # vertices = np.asarray(mesh.vertices)
            combined_points.extend(np.asarray(mesh.sample_points_uniformly(number_of_points=5000).points))
            geometries.append(mesh)
            plane_normals.append(np.asarray(mesh.triangle_normals)[2]) # the 0th item or the 2th item indicate the normal for the pseudo-plane
            # geometries.append(obox)
            geometries.append(downpcd)

        # o3d.visualization.draw_geometries(geometries,
        #                                 zoom=0.62,
        #                                 front=[0.4361, -0.2632, -0.8605],
        #                                 lookat=[2.4947, 1.7728, 1.5541],
        #                                 up=[-0.1726, -0.9630, 0.2071])
        meshpoint_publisher.pointcloud_publish(combined_points)


class MeshToPointCloudPublisher(Node):
    def __init__(self):
        super().__init__('mesh_to_pointcloud_publisher')
        self.publisher = self.create_publisher(PointCloud2, '/plane_pointcloud', 1)
        self.timer = self.create_timer(0.1, self.pointcloud_publish)  # Publish every 0.1 second


    def pointcloud_publish(self, points):

        
        # Convert combined points to PointCloud2 message
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'hesai_at128'  # Update to the appropriate frame_id

        pc_data = pc2.create_cloud_xyz32(header, points)
        
        # Publish the PointCloud2 message
        self.publisher.publish(pc_data)
        self.get_logger().info('Published Combined PointCloud2 Message')



def main(args=None):
    rclpy.init(args=args)
    pointcloud_subscriber = PointCloudSubscriber()
    global meshpoint_publisher
    meshpoint_publisher = MeshToPointCloudPublisher()
    try:
        rclpy.spin(pointcloud_subscriber)
        rclpy.spin(meshpoint_publisher)
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
    finally:
        pointcloud_subscriber.destroy_node()
        meshpoint_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
