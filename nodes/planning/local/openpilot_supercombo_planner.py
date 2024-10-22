#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import message_filters
import tf2_ros
from image_geometry import PinholeCameraModel

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from visualization_msgs.msg import MarkerArray, Marker

from cv_bridge import CvBridge

from helpers.supercombo import SupercomboModel, SupercomboConstants
from helpers.transform import transform_point


class OpenpilotSupercomboPlanner:

    def __init__(self):

        # Parameters
        supercombo_path = rospy.get_param("~supercombo_path")
        supercombo_metadata_path = rospy.get_param("~supercombo_metadata_path")
        #self.transform_timeout = rospy.get_param("~transform_timeout")
        self.transform_timeout = 0.06

        # Variables
        self.prev_fov60_image = None
        self.prev_fov120_image = None
        self.prev_output = None

        self.bridge = CvBridge()
        self.supercombo_model = SupercomboModel(supercombo_path, supercombo_metadata_path)

        # Publishers
        self.plan_pub = rospy.Publisher('supercombo_plan', MarkerArray, queue_size=10, tcp_nodelay=True)
        self.supercombo_img_pub = rospy.Publisher('supercombo_image', Image, queue_size=1, tcp_nodelay=True)

        # Camera models
        self.camera_model_fov60 = None
        self.camera_model_fov120 = None
        rospy.Subscriber('camera_info_fov60', CameraInfo, self.camera_info_fov60_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('camera_info_fov120', CameraInfo, self.camera_info_fov120_callback, queue_size=1, tcp_nodelay=True)

        self.tf_buffer = tf2_ros.Buffer()

        # Subscribers
        image_fov60_sub = message_filters.Subscriber('image_raw_fov60', Image, queue_size=1, buff_size=2**26, tcp_nodelay=True)
        image_fov120_sub = message_filters.Subscriber('image_raw_fov120', Image, queue_size=1, buff_size=2**26, tcp_nodelay=True)

        ts = message_filters.ApproximateTimeSynchronizer([image_fov60_sub, image_fov120_sub], queue_size=2, slop=0.1)
        ts.registerCallback(self.synchronized_camera_image_callback)

    def camera_info_fov60_callback(self, camera_info_msg):
        if self.camera_model_fov60 is None:
            self.camera_model_fov60 = PinholeCameraModel()
        self.camera_model_fov60.fromCameraInfo(camera_info_msg)

    def camera_info_fov120_callback(self, camera_info_msg):
        if self.camera_model_fov120 is None:
            self.camera_model_fov120 = PinholeCameraModel()
        self.camera_model_fov120.fromCameraInfo(camera_info_msg)

    def synchronized_camera_image_callback(self, fov60_image_msg, fov120_image_msg):
        if self.camera_model_fov60 is None:
            rospy.logwarn_throttle(10, "%s - No FOV 60 camera model received, skipping image", rospy.get_name())
            return
        
        if self.camera_model_fov120 is None:
            rospy.logwarn_throttle(10, "%s - No FOV 120 camera model received, skipping image", rospy.get_name())
            return

        # extract images
        fov60_image = self.bridge.imgmsg_to_cv2(fov60_image_msg,  desired_encoding='rgb8')
        fov120_image = self.bridge.imgmsg_to_cv2(fov120_image_msg,  desired_encoding='rgb8')

        if self.prev_fov60_image is None or self.prev_fov120_image is None:
            self.prev_fov60_image = fov60_image
            self.prev_fov120_image = fov120_image
            return
        
        image_time_stamp = fov60_image_msg.header.stamp
        transform_from_frame = fov60_image_msg.header.frame_id
        
        try:
            transform = self.tf_buffer.lookup_transform("map", transform_from_frame, image_time_stamp, rospy.Duration(self.transform_timeout))
        except (tf2_ros.TransformException, rospy.ROSTimeMovedBackwardsException) as e:
            rospy.logwarn("%s - %s", rospy.get_name(), e)
            return

        model_output = self.supercombo_model.predict((self.prev_fov60_image, fov60_image), (self.prev_fov120_image, fov120_image), self.prev_output)
        image_pub = self.project_supercombo_output_to_image(fov60_image, model_output)

        img_msg = self.bridge.cv2_to_imgmsg(image_pub, encoding='rgb8')
        img_msg.header.stamp = fov60_image_msg.header.stamp
        self.supercombo_img_pub.publish(img_msg)

        marker_array = self.publish_plan_markers(fov60_image_msg.header, model_output, transform)
        self.plan_pub.publish(marker_array)

    def publish_plan_markers(self, header, model_output, transform):
        plan_points = []
        for x, y, z in model_output["plan"][0, :, :3]:
            point_camera = Point(x=x,y=y,z=z)
            point_map = transform_point(point_camera, transform)
            plan_points.append(point_map)

        marker_array = MarkerArray()

        marker = Marker(header=header)
        marker.ns = "Supercombo plan"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)
        marker.points = ()
        marker_array.markers.append(marker)

        return marker_array

    def project_supercombo_output_to_image(self, image, model_output):
        # visualize lane boundaries from model
        x_coords = np.array(SupercomboConstants.X_IDXS)[:, np.newaxis]
        for i in range(4):
            prev_p = None
            points = np.hstack((x_coords, model_output["lane_lines"][0, i, :, :]))
            for x, y, z in points:
                if x != 0:
                    u, v = self.camera_model_fov60.project3dToPixel((y, z, x))
                    current_p = (int(u), int(v))
                    cv2.circle(image, current_p, 5, (0, 255, 0), -1)
                    if prev_p is not None:
                        cv2.line(image, prev_p, current_p, (0, 255, 0), 3)
                    prev_p = current_p

        # visualize plan from model
        prev_p = None
        for x, y, z in model_output["plan"][0, :, :3]:
            if x > 0:
                u, v = self.camera_model_fov60.project3dToPixel((y, z, x))
                current_p = (int(u), int(v))
                cv2.circle(image, current_p, 5, (0, 0, 255), -1)
                if prev_p is not None:
                    cv2.line(image, prev_p, current_p, (0, 0, 255), 3)
                prev_p = current_p
        
        return image


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('camera_traffic_light_detector', log_level=rospy.INFO)
    node = OpenpilotSupercomboPlanner()
    node.run()