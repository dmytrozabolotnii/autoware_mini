#!/usr/bin/env python3

import rospy
import numpy as np
import tf2_ros
from std_msgs.msg import ColorRGBA, Float32MultiArray
from geometry_msgs.msg import Point, TransformStamped, Quaternion
from visualization_msgs.msg import MarkerArray, Marker

from helpers.transform import transform_point
from tf.transformations import quaternion_from_euler

NO_TRAVERSAL_LIMIT = 2**64-1

class SupercomboPredictionVisualizer:

    def __init__(self):

        # Parameters
        self.transform_timeout = rospy.get_param("~transform_timeout")

        # Publishers
        self.supercombo_plan_pub = rospy.Publisher('supercombo_plan', MarkerArray, queue_size=10, tcp_nodelay=True)
        self.supercombo_lanes_pub = rospy.Publisher('supercombo_lanes', MarkerArray, queue_size=10, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('/openpilot/position', Float32MultiArray, self.position_callback, queue_size=None, tcp_nodelay=True)
        rospy.Subscriber('/openpilot/lane_lines', Float32MultiArray, self.lane_lines_callback, queue_size=None, tcp_nodelay=True)

        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.publish_camera_to_supercombo_tf()


    def position_callback(self, msg):
        try:
            transform = self.tf_buffer.lookup_transform("base_link", "openpilot", rospy.Time.now(), rospy.Duration(self.transform_timeout))
        except (tf2_ros.TransformException, rospy.ROSTimeMovedBackwardsException) as e:
            rospy.logwarn("%s - %s", rospy.get_name(), e)
            return


        position = np.array(msg.data).reshape(msg.layout.dim[0].size, msg.layout.dim[1].size)

        plan_points = []
        for x, y, z, t in position.T:
            point_supercombo = Point(x=x,y=y,z=z)
            point_base_link = transform_point(point_supercombo, transform)
            plan_points.append(point_base_link)

        plan_marker_array = MarkerArray()

        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
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


    def lane_lines_callback(self, msg):
        try:
            transform = self.tf_buffer.lookup_transform("base_link", "openpilot", rospy.Time.now(), rospy.Duration(self.transform_timeout))
        except (tf2_ros.TransformException, rospy.ROSTimeMovedBackwardsException) as e:
            rospy.logwarn("%s - %s", rospy.get_name(), e)
            return

        lane_lines = np.array(msg.data).reshape(msg.layout.dim[0].size, msg.layout.dim[1].size, msg.layout.dim[2].size)

        lanes_marker_array = MarkerArray()

        for i in range(4):
            lane_points = []
            for x, y, z, t in lane_lines[i].T:
                point_supercombo = Point(x=x,y=y,z=z)
                point_base_link = transform_point(point_supercombo, transform)
                lane_points.append(point_base_link)

            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "Supercombo lane"
            marker.id = i+1
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            marker.color = ColorRGBA(0.0, 1.0, 0.7, 1.0)
            marker.points = lane_points
            lanes_marker_array.markers.append(marker)
        
        self.supercombo_lanes_pub.publish(lanes_marker_array)
        return


    def publish_camera_to_supercombo_tf(self):
        
        t = TransformStamped()

        x, y, z, w = quaternion_from_euler(np.pi, 0.0, 0.0, axes='rxyz')
        orientation = Quaternion(x, y, z, w)

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "base_link"
        t.child_frame_id = "openpilot"

        t.transform.translation.x = 2.41 #nvidia cam +0.5
        t.transform.translation.y = 0.09
        t.transform.translation.z = 0.87 #nvidia cam -0.3
        t.transform.rotation = orientation

        self.tf_broadcaster.sendTransform(t)


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('supercombo_prediction_visualizer')
    node = SupercomboPredictionVisualizer()
    node.run()