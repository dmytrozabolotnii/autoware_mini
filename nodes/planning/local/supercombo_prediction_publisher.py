#!/usr/bin/env python3

import rospy
import zmq
import capnp
import threading
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from visualization_msgs.msg import MarkerArray, Marker

from cereal import log

NO_TRAVERSAL_LIMIT = 2**64-1

class SupercomboPredictionPublisher:

    def __init__(self):

        # parameters

        # variables

        # Establish connction with Comma
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.socket.connect('tcp://127.0.0.1:27098')
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        # publishers
        self.supercombo_plan_pub = rospy.Publisher('supercombo_plan', MarkerArray, queue_size=10, tcp_nodelay=True)
        self.supercombo_lanes_pub = rospy.Publisher('supercombo_lanes', MarkerArray, queue_size=10, tcp_nodelay=True)

    def log_from_bytes(self, dat: bytes, struct: capnp.lib.capnp._StructModule = log.Event) -> capnp.lib.capnp._DynamicStructReader:
        with struct.from_bytes(dat, traversal_limit_in_words=NO_TRAVERSAL_LIMIT) as msg:
            return msg


    def publish_supercombo_predictions(self):
        while True:
            byte_message = self.socket.recv()
            if byte_message is not None:
                message = self.log_from_bytes(byte_message)
                self.publish_plan_and_lane_markers(message.modelV2)


    def publish_plan_and_lane_markers(self, model_output):
        plan_points = []
        plan = model_output.position
        for x, y, z in zip(plan.x, plan.y, plan.z):
            point_supercombo = Point(x=x,y=y,z=z)
            #point_map = transform_point(point_supercombo, transform)
            plan_points.append(point_supercombo)

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
        
        lanes_marker_array = MarkerArray()

        for i in range(4):
            lane_line = model_output.laneLines[i]

            lane_points = []
            for x, y, z in zip(lane_line.x, lane_line.y, lane_line.z):
                point_supercombo = Point(x=x,y=y,z=z)
                #point_map = transform_point(point_supercombo, transform)
                lane_points.append(point_supercombo)

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


    def run(self):
        # Create a thread for receiving supercombo predictions
        t = threading.Thread(target = self.publish_supercombo_predictions)
        t.daemon = True
        t.start()

        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('supercombo_prediction_publisher')
    node = SupercomboPredictionPublisher()
    node.run()