import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import cv2
from cv_bridge import CvBridge
from threading import Thread, Lock
import zmq
import base64
import numpy as np

class MultiTopicPublisher(Node):
    def __init__(self):
        super().__init__('multi_topic_publisher')


        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.PUB)
        self.socket.bind("tcp://*:5555")  # Binding to port 5555 for sending images

        self.receive_context = zmq.Context()
        self.receive_socket = self.receive_context.socket(zmq.SUB)
        self.receive_socket.connect("tcp://localhost:5556")  # Binding to port 5555 for sending images
        self.receive_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages\

        # Create publishers for multiple topics
        self.image_publisher = self.create_publisher(Image, 'camera/image', 10)
        self.semantic_publisher = self.create_publisher(Image, 'camera/semantic_image', 10)
        self.camera_info_publisher = self.create_publisher(CameraInfo, 'camera/camera_info', 10)

        # Initialize OpenCV VideoCapture for the USB camera
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2 )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 10)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 1)

    
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
            # equilize the frame, it is tooo bright
            if ret:
                with self.lock:
                    self.frame = frame

    def publish_image(self):
        with self.lock:
            if self.frame is not None:
                

                '''send the image through zmq to other scritps that does not support ROS2'''
                _, buffer = cv2.imencode('.jpg', self.frame)
                img_as_text = base64.b64encode(buffer).decode('utf-8')
                self.socket.send_string(f"{img_as_text}")

                
                '''recieve segmentated images'''
                img_as_text_receive = self.receive_socket.recv_string()
                # Decode the base64 string back to binary
                img_data_receive = base64.b64decode(img_as_text_receive)
                # Convert the binary data to a NumPy array and decode the JPEG
                np_img_receive = np.frombuffer(img_data_receive, dtype=np.uint8)
                img_segment = cv2.imdecode(np_img_receive, cv2.IMREAD_COLOR)

                '''send original image'''
                # Convert the OpenCV image (BGR) to a ROS Image message
                image_message = self.br.cv2_to_imgmsg(self.frame, encoding="bgr8")
                image_message.header.stamp = self.get_clock().now().to_msg()
                image_message.header.frame_id = "camera_frame"
                # Publish the image
                self.image_publisher.publish(image_message)

                '''send segmented image'''
                image_message_segment = self.br.cv2_to_imgmsg(img_segment, encoding="bgr8")
                image_message_segment.header.stamp = self.get_clock().now().to_msg()
                image_message_segment.header.frame_id = "camera_frame"
                # Publish the image
                self.semantic_publisher.publish(image_message_segment)


    def publish_camera_info(self):
        # Create a fake CameraInfo message
        camera_info_msg = CameraInfo()
        camera_info_msg.header.stamp = self.get_clock().now().to_msg()
        camera_info_msg.header.frame_id = "camera_frame"
        camera_info_msg.width = 1920
        camera_info_msg.height = 1080
        camera_info_msg.k = [1043.02215,    0.     ,  963.4692 ,
                             0.     , 1043.30157,  528.77189,
                             0.     ,    0.     ,    1.     ]
        camera_info_msg.d = [0.153638, -0.143077, 0.003250, -0.001801, 0.000000]
        camera_info_msg.distortion_model = "plumb_bob"
        camera_info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        camera_info_msg.p = [1075.50488,    0.     ,  958.35386,    0.  ,
                             0.     , 1085.85059,  531.53889,    0.     ,
                             0.     ,    0.     ,    1.     ,    0.     ]

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
