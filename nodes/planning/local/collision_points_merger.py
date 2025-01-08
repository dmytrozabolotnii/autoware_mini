#!/usr/bin/env python3

import rospy
import numpy as np
from ros_numpy import msgify, numpify
import message_filters
import traceback
from sensor_msgs.msg import PointCloud2

class CollisionPointsMerger:

    def __init__(self):

        # parameters
        synchronization_method = rospy.get_param("~synchronization_method")
        synchronization_queue_size = rospy.get_param("~synchronization_queue_size")
        synchronization_slop = rospy.get_param("~synchronization_slop")

        topics = [
            rospy.get_param("~goal_checker_topic", None),
            rospy.get_param("~object_checker_topic", None),
            rospy.get_param("~traffic_light_checker_topic", None),
            rospy.get_param("~crosswalk_checker_topic", None),
            rospy.get_param("~auto_stop_checker_topic", None),
            rospy.get_param("~trajectory_checker_topic", None)
        ]

        # publishers
        self.collision_points_pub = rospy.Publisher('collision_points', PointCloud2, queue_size=1, tcp_nodelay=True)

        # subscribers
        subscribers = []
        for topic in topics:
            if topic is not None:
                subscribers.append(message_filters.Subscriber(topic, PointCloud2, tcp_nodelay=True))

        if not subscribers:
            raise ValueError("No topics to subscribe to.")

        # Synchronize messages
        if synchronization_method == "approximate":
            ts = message_filters.ApproximateTimeSynchronizer(subscribers, queue_size=synchronization_queue_size, slop=synchronization_slop)
        elif synchronization_method == "exact":
            ts = message_filters.TimeSynchronizer(subscribers, queue_size=2)
        else:
            raise ValueError(f"'{synchronization_method}' is not a known synchronization method")

        ts.registerCallback(self.collision_points_callback)

    def collision_points_callback(self, *msgs):
        try:
            # Convert all incoming messages to numpy arrays
            collision_points_np_list = [numpify(msg) for msg in msgs]
            collision_points_np = np.concatenate(collision_points_np_list)

            # Create a new PointCloud2 message
            collision_points_msg = msgify(PointCloud2, collision_points_np)
            collision_points_msg.header = msgs[0].header  # Use the header of the first message

            # Publish the merged collision points
            self.collision_points_pub.publish(collision_points_msg)

        except Exception as e:
            rospy.logerr_throttle(10, "%s - Exception in callback: %s", rospy.get_name(), traceback.format_exc())

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('collision_points_merger')
    node = CollisionPointsMerger()
    node.run()