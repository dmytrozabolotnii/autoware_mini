#!/usr/bin/env python3

import rospy
import numpy as np
import zmq
import capnp
import threading
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from cereal import log

NO_TRAVERSAL_LIMIT = 2**64-1

class SupercomboPredictionPublisher:

    def __init__(self):

        # Parameters
        openpilot_ip = rospy.get_param("~openpilot_ip")
        openpilot_port = rospy.get_param("~openpilot_port")

        # Variables
        self.output_size = 33

        # Establish connction with Comma
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.socket.connect(f'tcp://{openpilot_ip}:{openpilot_port}')
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        # Publishers
        self.position_pub = rospy.Publisher('position', Float32MultiArray, queue_size=10, tcp_nodelay=True)
        self.orientation_pub = rospy.Publisher('orientation', Float32MultiArray, queue_size=10, tcp_nodelay=True)
        self.velocity_pub = rospy.Publisher('velocity', Float32MultiArray, queue_size=10, tcp_nodelay=True)
        self.orientation_rate_pub = rospy.Publisher('orientation_rate', Float32MultiArray, queue_size=10, tcp_nodelay=True)
        self.acceleration_pub = rospy.Publisher('acceleration', Float32MultiArray, queue_size=10, tcp_nodelay=True)
        self.lane_lines_pub = rospy.Publisher('lane_lines', Float32MultiArray, queue_size=10, tcp_nodelay=True)
        self.road_edges_pub = rospy.Publisher('road_edges', Float32MultiArray, queue_size=10, tcp_nodelay=True)

    def log_from_bytes(self, dat: bytes, struct: capnp.lib.capnp._StructModule = log.Event) -> capnp.lib.capnp._DynamicStructReader:
        # Decodes the ZMQ message 
        with struct.from_bytes(dat, traversal_limit_in_words=NO_TRAVERSAL_LIMIT) as msg:
            return msg
        
    def publish_2d_multiarray(self, data, publisher):
        data_array = [data.x, data.y, data.z, data.t]
        msg = Float32MultiArray()

        msg.layout.data_offset = 0 
        msg.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]

        msg.layout.dim[0].label = "axis"
        msg.layout.dim[0].size = 4
        msg.layout.dim[0].stride = 4*self.output_size

        msg.layout.dim[1].label = "values"
        msg.layout.dim[1].size = self.output_size
        msg.layout.dim[1].stride = self.output_size

        msg.data = np.array(data_array).reshape((4*self.output_size,))

        publisher.publish(msg)

    def publish_supercombo_predictions(self):
        while True:
            byte_message = self.socket.recv()
            if byte_message is None:
                continue
            
            message = self.log_from_bytes(byte_message)
            modelV2 = message.modelV2

            # position
            self.publish_2d_multiarray(modelV2.position, self.position_pub)

            # orientation
            self.publish_2d_multiarray(modelV2.orientation, self.orientation_pub)

            # velocity
            self.publish_2d_multiarray(modelV2.velocity, self.velocity_pub)

            # orientation rate
            self.publish_2d_multiarray(modelV2.orientationRate, self.orientation_rate_pub)

            # acceleration
            self.publish_2d_multiarray(modelV2.acceleration, self.acceleration_pub)
        
            # lane lines
            lane_lines_array = []
            for i in range(4):
                lane_lines_array.append([modelV2.laneLines[i].x, modelV2.laneLines[i].y, modelV2.laneLines[i].z, modelV2.laneLines[i].t])

            lane_lines_msg = Float32MultiArray()
            lane_lines_msg.layout.data_offset = 0 

            lane_lines_msg.layout.dim = [MultiArrayDimension(), MultiArrayDimension(), MultiArrayDimension()]

            lane_lines_msg.layout.dim[0].label = "lane_line"
            lane_lines_msg.layout.dim[0].size = 4
            lane_lines_msg.layout.dim[0].stride = 16*self.output_size

            lane_lines_msg.layout.dim[1].label = "axis"
            lane_lines_msg.layout.dim[1].size = 4
            lane_lines_msg.layout.dim[1].stride = 4*self.output_size

            lane_lines_msg.layout.dim[2].label = "values"
            lane_lines_msg.layout.dim[2].size = self.output_size
            lane_lines_msg.layout.dim[2].stride = self.output_size

            lane_lines_msg.data = np.array(lane_lines_array).reshape((16*self.output_size,))

            self.lane_lines_pub.publish(lane_lines_msg)

            # road edges
            road_edges_array = []
            for i in range(2):
                road_edges_array.append([modelV2.roadEdges[i].x, modelV2.roadEdges[i].y, modelV2.roadEdges[i].z, modelV2.roadEdges[i].t])

            road_edges_msg = Float32MultiArray()
            road_edges_msg.layout.data_offset = 0 

            road_edges_msg.layout.dim = [MultiArrayDimension(), MultiArrayDimension(), MultiArrayDimension()]

            road_edges_msg.layout.dim[0].label = "road_edge"
            road_edges_msg.layout.dim[0].size = 2
            road_edges_msg.layout.dim[0].stride = 8*self.output_size

            road_edges_msg.layout.dim[1].label = "axis"
            road_edges_msg.layout.dim[1].size = 4
            road_edges_msg.layout.dim[1].stride = 4*self.output_size

            road_edges_msg.layout.dim[2].label = "values"
            road_edges_msg.layout.dim[2].size = self.output_size
            road_edges_msg.layout.dim[2].stride = self.output_size

            road_edges_msg.data = np.array(road_edges_array).reshape((8*self.output_size,))

            self.road_edges_pub.publish(road_edges_msg)

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