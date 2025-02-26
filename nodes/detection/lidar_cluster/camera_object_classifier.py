#!/usr/bin/env python3

import rospy
import tf2_ros
import onnxruntime
import message_filters
from image_geometry import PinholeCameraModel

from sensor_msgs.msg import CameraInfo, Image
from autoware_mini.msg import DetectedObjectArray

from cv_bridge import CvBridge

class CameraObjectClassifier:
    def __init__(self):
        onnx_path = ""
        self.iou_threshold = 0.5
        self.transform_timeout = 0.06

        self.camera_model = None
        
        self.bridge = CvBridge()
        self.model = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Publishers
        self.detected_objects_classified_sub = rospy.Publisher('detected_objects_classified', DetectedObjectArray, queue_size=1, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('camera_info', CameraInfo, self.camera_info_callback, queue_size=1, tcp_nodelay=True)
        image_raw_sub = message_filters.Subscriber('image_raw', Image, queue_size=1, buff_size=2**26, tcp_nodelay=True)
        detected_objects_sub = message_filters.Subscriber('detected_objects', DetectedObjectArray, queue_size=1, buff_size=2**20, tcp_nodelay=True)

        ts = message_filters.ApproximateTimeSynchronizer([image_raw_sub, detected_objects_sub], queue_size=15, slop=0.15)
        ts.registerCallback(self.detected_objects_callback)

    def camera_info_callback(self, camera_info_msg):
        if self.camera_model is None:
            self.camera_model = PinholeCameraModel()
        self.camera_model.fromCameraInfo(camera_info_msg)

    def detected_objects_callback(self, image_msg, det_objects_msg):
        if self.camera_model is None:
            rospy.logwarn_throttle(10, "%s - No camera model received, skipping image", rospy.get_name())
            return
        
        detected_objects = det_objects_msg.objects

        # extract image
        image = self.bridge.imgmsg_to_cv2(image_msg,  desired_encoding='rgb8')
        
        print(image.shape)

        # extract transform
        try:
            transform = self.tf_buffer.lookup_transform(image_msg.header.frame_id, det_objects_msg.header.frame_id, image_msg.header.stamp, rospy.Duration(self.transform_timeout))
        except (tf2_ros.TransformException, rospy.ROSTimeMovedBackwardsException) as e:
            rospy.logwarn("%s - %s", rospy.get_name(), e)
            return
        
        for detected_object in detected_objects:
            pass
            #print(detected_object)
            """
            for x, y, z in traffic_lights.values():
                point_map = Point(float(x), float(y), float(z))

                # transform point to camera frame and then to image frame
                point_camera = transform_point(point_map, transform)
                u, v = self.camera_model.project3dToPixel((point_camera.x, point_camera.y, point_camera.z))
            """

        self.detected_objects_classified_sub.publish(det_objects_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('camera_object_classifier', log_level=rospy.INFO)
    node = CameraObjectClassifier()
    node.run()