import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import cv2
from cv_bridge import CvBridge
from threading import Thread, Lock

class MultiTopicPublisher(Node):
    def __init__(self):
        super().__init__('multi_topic_publisher')

        # Create publishers for multiple topics
        self.image_publisher = self.create_publisher(Image, 'camera/image', 10)
        self.camera_info_publisher = self.create_publisher(CameraInfo, 'camera/camera_info', 10)

        # Initialize OpenCV VideoCapture for the USB camera
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2 )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 10)
        # self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

    
        self.lock = Lock()

        # Create a separate thread for capturing frames
        self.frame = None
        self.capture_thread = Thread(target=self.capture_frames)
        self.capture_thread.start()

        # Create timers for publishing messages at different frequencies
        self.image_timer = self.create_timer(0.05, self.publish_image)        # Publish images at 10 Hz
        self.camera_info_timer = self.create_timer(1.0, self.publish_camera_info)  # Publish camera info at 1 Hz

        # Create a CvBridge to convert between OpenCV images and ROS Image messages
        self.br = CvBridge()

    def capture_frames(self):
        while rclpy.ok():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def publish_image(self):
        with self.lock:
            if self.frame is not None:
                # Convert the OpenCV image (BGR) to a ROS Image message
                image_message = self.br.cv2_to_imgmsg(self.frame, encoding="bgr8")
                image_message.header.stamp = self.get_clock().now().to_msg()
                image_message.header.frame_id = "camera_frame"

                # Publish the image
                self.image_publisher.publish(image_message)

    def publish_camera_info(self):
        # Create a fake CameraInfo message
        camera_info_msg = CameraInfo()
        camera_info_msg.header.stamp = self.get_clock().now().to_msg()
        camera_info_msg.header.frame_id = "camera_frame"
        camera_info_msg.width = 1920
        camera_info_msg.height = 1080
        camera_info_msg.k = [995.678, 0.0, 973.222, 0.0, 997.51, 520.94, 0.0, 0.0, 1.0]
        camera_info_msg.d = [0.139704, -0.109251, -0.001611, 0.001925, 0.0]
        camera_info_msg.distortion_model = "plumb_bob"
        camera_info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        camera_info_msg.p = [1033.7482, 0.0, 970.2617, 0.0, 0.0, 1042.58, 518.215, 0.0, 0.0, 0.0, 1.0, 0.0]

        # Publish the CameraInfo message
        self.camera_info_publisher.publish(camera_info_msg)

def main(args=None):
    rclpy.init(args=args)

    # Create the multi-topic publisher node
    multi_topic_publisher = MultiTopicPublisher()

    # Spin the node so it keeps running
    rclpy.spin(multi_topic_publisher)

    # Clean up on exit
    multi_topic_publisher.cap.release()
    multi_topic_publisher.capture_thread.join()
    multi_topic_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
