#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import ColorRGBA, Float32MultiArray
from geometry_msgs.msg import Point
from visualization_msgs.msg import MarkerArray, Marker

from rospy.numpy_msg import numpy_msg

NO_TRAVERSAL_LIMIT = 2**64-1

class OpenPilotPredictionVisualizer:

    def __init__(self):

        # Publishers
        self.openpilot_plan_pub = rospy.Publisher('openpilot_plan_markers', MarkerArray, queue_size=1, tcp_nodelay=True)
        self.openpilot_lanes_pub = rospy.Publisher('openpilot_lanes_markers', MarkerArray, queue_size=1, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('/openpilot/position', numpy_msg(Float32MultiArray), self.position_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/openpilot/lane_lines', numpy_msg(Float32MultiArray), self.lane_lines_callback, queue_size=1, tcp_nodelay=True)

    def float32_multiarray_to_numpy(self, multiarray):
        dims = tuple(map(lambda x: x.size, multiarray.layout.dim))
        return np.array(multiarray.data, dtype=float).reshape(dims).astype(np.float32)

    def position_callback(self, msg):
        position = self.float32_multiarray_to_numpy(msg)

        plan_points = []
        for x, y, z, t in position.T:
            point_openpilot= Point(x=x,y=y,z=z)
            plan_points.append(point_openpilot)

        plan_marker_array = MarkerArray()

        marker = Marker()
        marker.header.frame_id = "openpilot"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "Openpilot plan"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)
        marker.points = plan_points
        plan_marker_array.markers.append(marker)

        self.openpilot_plan_pub.publish(plan_marker_array)


    def lane_lines_callback(self, msg):
        lane_lines = self.float32_multiarray_to_numpy(msg)

        lanes_marker_array = MarkerArray()

        for i in range(4):
            lane_points = []
            for x, y, z, t in lane_lines[i].T:
                point_openpilot = Point(x=x,y=y,z=z)
                lane_points.append(point_openpilot)

            marker = Marker()
            marker.header.frame_id = "openpilot"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "Openpilot lane"
            marker.id = i+1
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            marker.color = ColorRGBA(0.0, 1.0, 0.7, 1.0)
            marker.points = lane_points
            lanes_marker_array.markers.append(marker)
        
        self.openpilot_lanes_pub.publish(lanes_marker_array)
        return

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('openpilot_prediction_visualizer')
    node = OpenPilotPredictionVisualizer()
    node.run()