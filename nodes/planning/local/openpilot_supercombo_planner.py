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
from geometry_msgs.msg import Point, TwistStamped, TransformStamped, Quaternion
from geometry_msgs.msg import Point, TwistStamped, TransformStamped, Quaternion
from visualization_msgs.msg import MarkerArray, Marker

from cv_bridge import CvBridge

from helpers.transform import transform_point
from helpers.supercombo import SupercomboModel, SupercomboConstants
from tf.transformations import quaternion_from_euler
from tf.transformations import quaternion_from_euler


class OpenpilotSupercomboPlanner:

    def __init__(self):

        # Parameters
        supercombo_path = rospy.get_param("~supercombo_path")
        supercombo_metadata_path = rospy.get_param("~supercombo_metadata_path")
        #self.transform_timeout = rospy.get_param("~transform_timeout")
        self.transform_timeout = 0.06
        self.use_wide_camera = rospy.get_param("~use_wide_camera")
        self.narrow_image_crop_factor = rospy.get_param("~narrow_image_crop_factor")
        self.wide_image_crop_factor = rospy.get_param("~wide_image_crop_factor")
        self.use_wide_camera = rospy.get_param("~use_wide_camera")
        self.narrow_image_crop_factor = rospy.get_param("~narrow_image_crop_factor")
        self.wide_image_crop_factor = rospy.get_param("~wide_image_crop_factor")

        # Variables
        self.prev_fov60_image = None
        self.prev_fov120_image = None
        self.prev_output = None
        self.current_speed = None

        self.bridge = CvBridge()
        self.supercombo_model = SupercomboModel(supercombo_path, supercombo_metadata_path, self.use_wide_camera, 
                                                self.narrow_image_crop_factor, self.wide_image_crop_factor)
        self.supercombo_model = SupercomboModel(supercombo_path, supercombo_metadata_path, self.use_wide_camera, 
                                                self.narrow_image_crop_factor, self.wide_image_crop_factor)

        # Publishers
        self.supercombo_plan_pub = rospy.Publisher('supercombo_plan', MarkerArray, queue_size=10, tcp_nodelay=True)
        self.supercombo_lanes_pub = rospy.Publisher('supercombo_lanes', MarkerArray, queue_size=10, tcp_nodelay=True)
        self.supercombo_plan_pub = rospy.Publisher('supercombo_plan', MarkerArray, queue_size=10, tcp_nodelay=True)
        self.supercombo_lanes_pub = rospy.Publisher('supercombo_lanes', MarkerArray, queue_size=10, tcp_nodelay=True)
        self.supercombo_img_pub = rospy.Publisher('supercombo_image', Image, queue_size=1, tcp_nodelay=True)

        # Camera models
        self.camera_model_fov60 = None
        self.camera_model_fov120 = None
        rospy.Subscriber('camera_info_fov60', CameraInfo, self.camera_info_fov60_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('camera_info_fov120', CameraInfo, self.camera_info_fov120_callback, queue_size=1, tcp_nodelay=True)

        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer()
        self.publish_camera_to_supercombo_tf()

        # Subscribers
        image_fov60_sub = message_filters.Subscriber('image_raw_fov60', Image, queue_size=1, buff_size=2**26, tcp_nodelay=True)
        image_fov120_sub = message_filters.Subscriber('image_raw_fov120', Image, queue_size=1, buff_size=2**26, tcp_nodelay=True)
        rospy.Subscriber('/localization/current_velocity', TwistStamped, self.current_velocity_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/localization/current_velocity', TwistStamped, self.current_velocity_callback, queue_size=1, tcp_nodelay=True)

        ts = message_filters.ApproximateTimeSynchronizer([image_fov60_sub, image_fov120_sub], queue_size=2, slop=0.051)
        ts.registerCallback(self.synchronized_camera_image_callback)

    def current_velocity_callback(self, msg):
        self.current_speed = msg.twist.linear.x

    def current_velocity_callback(self, msg):
        self.current_speed = msg.twist.linear.x

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
        
        current_speed = self.current_speed
        if current_speed is None:
            rospy.logwarn_throttle(3, "%s - current speed not received!", rospy.get_name())
            return

        current_speed = max(0, current_speed)
        current_speed = self.current_speed
        if current_speed is None:
            rospy.logwarn_throttle(3, "%s - current speed not received!", rospy.get_name())
            return

        current_speed = max(0, current_speed)
        image_time_stamp = fov60_image_msg.header.stamp
        transform_from_frame = fov60_image_msg.header.frame_id
        transform = None

        """
        try:
            transform = self.tf_buffer.lookup_transform("map", "supercombo", image_time_stamp, rospy.Duration(self.transform_timeout))
        except (tf2_ros.TransformException, rospy.ROSTimeMovedBackwardsException) as e:
            rospy.logwarn("%s - %s", rospy.get_name(), e)
            return
        """
        model_output = self.supercombo_model.predict((self.prev_fov60_image, fov60_image), (self.prev_fov120_image, fov120_image), self.prev_output, current_speed)
        image_pub = self.project_supercombo_output_to_image(fov60_image, model_output)

        img_msg = self.bridge.cv2_to_imgmsg(image_pub, encoding='rgb8')
        img_msg.header.stamp = fov60_image_msg.header.stamp
        self.supercombo_img_pub.publish(img_msg)

        self.publish_plan_and_lane_markers(fov60_image_msg.header, model_output, transform)
        self.publish_plan_and_lane_markers(fov60_image_msg.header, model_output, transform)

    def publish_plan_and_lane_markers(self, header, model_output, transform):
        #print(header)

        plan_points = []
        for x, y, z in model_output["plan"][0, :, :3]:
            point_camera = Point(x=y,y=z,z=x)
            #point_map = transform_point(point_camera, transform)
            plan_points.append(point_camera)

        plan_marker_array = MarkerArray()
        #print(plan_points)

        marker = Marker(header=header)
        marker.ns = "Supercombo plan"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)
        marker.points = plan_points
        plan_marker_array.markers.append(marker)

        self.supercombo_plan_pub.publish(plan_marker_array)

        lanes_marker_array = MarkerArray()
        x_coords = np.array(SupercomboConstants.X_IDXS)[:, np.newaxis]

        for i in range(4):
            lane_points = np.hstack((x_coords, model_output["lane_lines"][0, i, :, :]))

            marker = Marker(header=header)
            marker.ns = "Supercombo lane"
            marker.id = i+1
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            marker.color = ColorRGBA(0.0, 1.0, 0.7, 1.0)
            marker.points = [Point(x=y,y=z,z=x) for x, y, z in lane_points]
            lanes_marker_array.markers.append(marker)
        
        self.supercombo_lanes_pub.publish(lanes_marker_array)


    def project_supercombo_output_to_image(self, image, model_output, image_time_stamp):
        try:
            transform = self.tf_buffer.lookup_transform("interfaceb_link0", "supercombo", image_time_stamp, rospy.Duration(self.transform_timeout))
        except (tf2_ros.TransformException, rospy.ROSTimeMovedBackwardsException) as e:
            rospy.logwarn("%s - %s", rospy.get_name(), e)
            return

        # visualize lane boundaries from model
        x_coords = np.array(SupercomboConstants.X_IDXS)[:, np.newaxis]
        for i in range(4):
            prev_p = None
            points = np.hstack((x_coords, model_output["lane_lines"][0, i, :, :]))
            for x, y, z in points:
                if x != 0:
                    point_supercombo = Point(x=x,y=y,z=z)
                    point_camera = transform_point(point_supercombo, transform)
                    u, v = self.camera_model_fov60.project3dToPixel((point_camera.x, point_camera.y, point_camera.z))
                    current_p = (int(u), int(v))
                    cv2.circle(image, current_p, 5, (0, 255, 0), -1)
                    if prev_p is not None:
                        cv2.line(image, prev_p, current_p, (0, 255, 0), 3)
                    prev_p = current_p

        # visualize plan from model
        prev_p = None
        for x, y, z in model_output["plan"][0, :, :3]:
            if x > 0:
                point_supercombo = Point(x=x,y=y,z=z)
                point_camera = transform_point(point_supercombo, transform)
                u, v = self.camera_model_fov60.project3dToPixel((point_camera.x, point_camera.y, point_camera.z))
                current_p = (int(u), int(v))
                cv2.circle(image, current_p, 5, (255, 0, 0), -1)
                if prev_p is not None:
                    cv2.line(image, prev_p, current_p, (255, 0, 0), 3)
                prev_p = current_p
        
        return image
    
    def publish_camera_to_supercombo_tf(self):
        br = tf2_ros.TransformBroadcaster()
        t = TransformStamped()

        x, y, z, w = quaternion_from_euler(0, -np.pi/2, -np.pi)
        orientation = Quaternion(x, y, z, w)

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "interfaceb_link0"
        t.child_frame_id = "supercombo"

        t.transform.translation.x = 0
        t.transform.translation.y = 0
        t.transform.translation.z = 0
        t.transform.rotation = orientation

        br.sendTransform(t)


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('camera_traffic_light_detector', log_level=rospy.INFO)
    node = OpenpilotSupercomboPlanner()
    node.run()