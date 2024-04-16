#!/usr/bin/env python3

import math
import rospy
import message_filters
from tf.transformations import quaternion_from_euler
from tf2_ros import TransformBroadcaster

from novatel_oem7_msgs.msg import INSPVA, BESTPOS
from geometry_msgs.msg import PoseStamped, TwistStamped, Quaternion, TransformStamped, PoseWithCovarianceStamped, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from ros_numpy import numpify, msgify
import numpy as np

from localization.WGS84ToUTMTransformer import WGS84ToUTMTransformer
from localization.WGS84ToLest97Transformer import WGS84ToLest97Transformer


class NovatelOem7Localizer:
    def __init__(self):

        # Parameters
        self.coordinate_transformer = rospy.get_param("coordinate_transformer")
        self.use_custom_origin = rospy.get_param("use_custom_origin")
        self.utm_origin_lat = rospy.get_param("utm_origin_lat")
        self.utm_origin_lon = rospy.get_param("utm_origin_lon")
        self.lest97_origin_northing = rospy.get_param("lest97_origin_northing")
        self.lest97_origin_easting = rospy.get_param("lest97_origin_easting")
        self.use_msl_height = rospy.get_param("~use_msl_height")
        self.child_frame = rospy.get_param("~child_frame")
        self.enable_setting_initialpose = rospy.get_param("~enable_setting_initialpose")

        # variable to store undulation value from bestpos message
        self.undulation = 0.0
        self.current_pose_msg = None
        self.relative_pose_matrix = None

        # initialize coordinate_transformer
        if self.coordinate_transformer == "utm":
            self.transformer = WGS84ToUTMTransformer(self.use_custom_origin, self.utm_origin_lat, self.utm_origin_lon)
        elif self.coordinate_transformer == "lest97":
            self.transformer = WGS84ToLest97Transformer(self.use_custom_origin, self.lest97_origin_northing, self.lest97_origin_easting)
        else:
            rospy.logfatal("%s - coordinate_transformer not supported: %s ", rospy.get_name(), str(self.coordinate_transformer))
            exit(1)

        # Publishers
        self.current_pose_pub = rospy.Publisher('current_pose', PoseStamped, queue_size=1, tcp_nodelay=True)
        self.current_velocity_pub = rospy.Publisher('current_velocity', TwistStamped, queue_size=1, tcp_nodelay=True)
        self.odometry_pub = rospy.Publisher('odometry', Odometry, queue_size=1, tcp_nodelay=True)
        
        # Subscribers
        rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.initialpose_callback, queue_size=1, tcp_nodelay=True)
        if self.use_msl_height:
            self.bestpos_sub = rospy.Subscriber('/novatel/oem7/bestpos', BESTPOS, self.bestpos_callback, queue_size=1, tcp_nodelay=True)

        inspva_sub = message_filters.Subscriber('/novatel/oem7/inspva', INSPVA, queue_size=1, tcp_nodelay=True)
        imu_sub = message_filters.Subscriber('/gps/imu', Imu, queue_size=1, tcp_nodelay=True)

        ts = message_filters.ApproximateTimeSynchronizer([inspva_sub, imu_sub], queue_size=5, slop=0.03)
        ts.registerCallback(self.synchronized_callback)
        # output information to console
        rospy.loginfo("%s - localizer initialized using %s coordinates", rospy.get_name(), str(self.coordinate_transformer))


    def initialpose_callback(self, pose_msg):
        if self.current_pose_msg is None:
            return
        # take the z value for initialpose from current_pose
        pose_msg.pose.pose.position.z = self.current_pose_msg.pose.position.z
        initialpose_matrix = numpify(pose_msg.pose.pose)
        current_pose_matrix = numpify(self.current_pose_msg.pose)
        # get the difference between initialpose and current_pose
        self.relative_pose_matrix = initialpose_matrix.dot(np.linalg.inv(current_pose_matrix))

    def synchronized_callback(self, inspva_msg, imu_msg):

        stamp = inspva_msg.header.stamp

        # transform GNSS coordinates and correct azimuth
        x, y = self.transformer.transform_lat_lon(inspva_msg.latitude, inspva_msg.longitude, inspva_msg.height)
        azimuth = self.transformer.correct_azimuth(inspva_msg.latitude, inspva_msg.longitude, inspva_msg.azimuth)

        linear_velocity = math.sqrt(inspva_msg.east_velocity**2 + inspva_msg.north_velocity**2)
        angular_velocity = imu_msg.angular_velocity

        # angles from GNSS (degrees) need to be converted to orientation (quaternion) in map frame
        orientation = convert_angles_to_orientation(inspva_msg.roll, inspva_msg.pitch, azimuth)

        # inspva_msg contains ellipsoid height if msl (mean sea level) height is wanted then undulation is subtracted
        height = inspva_msg.height
        if self.use_msl_height:
            height -= self.undulation

        current_pose_msg = PoseStamped()
        current_pose_msg.header.stamp = stamp
        current_pose_msg.header.frame_id = "map"

        current_pose_msg.pose.position.x = x
        current_pose_msg.pose.position.y = y
        current_pose_msg.pose.position.z = height
        current_pose_msg.pose.orientation = orientation

        self.current_pose_msg = current_pose_msg

        if self.enable_setting_initialpose and self.relative_pose_matrix is not None:
            current_pose_matrix = numpify(current_pose_msg.pose)
            new_current_pose_matrix = self.relative_pose_matrix.dot(current_pose_matrix)
            current_pose_msg.pose = msgify(Pose, new_current_pose_matrix)

        # Publish 
        self.current_pose_pub.publish(current_pose_msg)
        self.publish_current_velocity(stamp, linear_velocity, angular_velocity)
        self.publish_map_to_baselink_tf(stamp, x, y, height, orientation)
        self.publish_odometry(stamp, linear_velocity, x, y, height, orientation, angular_velocity)

    def bestpos_callback(self, bestpos_msg):
        self.undulation = bestpos_msg.undulation

    def publish_current_velocity(self, stamp, linear_velocity, angular_velocity):
        
        vel_msg = TwistStamped()

        vel_msg.header.stamp = stamp
        vel_msg.header.frame_id = self.child_frame
        vel_msg.twist.linear.x = linear_velocity
        vel_msg.twist.angular = angular_velocity

        self.current_velocity_pub.publish(vel_msg)

    def publish_odometry(self, stamp, velocity, x, y, height, orientation, angular_velocity):

        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = 'map'
        odom_msg.child_frame_id = 'base_link'

        odom_msg.pose.pose.position.x = x
        odom_msg.pose.pose.position.y = y
        odom_msg.pose.pose.position.z = height

        odom_msg.pose.pose.orientation = orientation
        odom_msg.twist.twist.linear.x = velocity
        odom_msg.twist.twist.angular = angular_velocity

        self.odometry_pub.publish(odom_msg)


    def publish_map_to_baselink_tf(self, stamp, x, y, height, orientation):
            
        br = TransformBroadcaster()
        t = TransformStamped()

        t.header.stamp = stamp
        t.header.frame_id = "map"
        t.child_frame_id = self.child_frame

        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = height
        t.transform.rotation = orientation

        br.sendTransform(t)


    def run(self):
        rospy.spin()


# Helper functions

def convert_angles_to_orientation(roll, pitch, yaw):
    
    # convert angles to radians
    roll = math.radians(roll)
    pitch = math.radians(pitch)
    yaw = math.radians(yaw)

    roll, pitch, yaw = convertAzimuthToENU(roll, pitch, yaw)
    x, y, z, w = quaternion_from_euler(roll, pitch, yaw)
    return Quaternion(x, y, z, w)


def convertAzimuthToENU(roll, pitch, yaw):

    # These transforms are taken from gpsins_localizer_nodelet.cpp
    # Convert from Azimuth (CW from North) to ENU (CCW from East)
    yaw = -yaw + math.pi/2

    # Clamp within 0 to 2 pi
    if yaw > 2 * math.pi:
        yaw = yaw - 2 * math.pi
    elif yaw < 0:
        yaw += 2 * math.pi
    
    # Novatel GPS uses different vehicle body frame (y forward, x right, z up)
    pitch = -pitch

    return roll, pitch, yaw


if __name__ == '__main__':
    rospy.init_node('novatel_oem7_localizer', log_level=rospy.INFO)
    node = NovatelOem7Localizer()
    node.run()
