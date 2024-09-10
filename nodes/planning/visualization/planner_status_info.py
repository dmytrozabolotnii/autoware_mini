#!/usr/bin/env python3

import rospy
from autoware_msgs.msg import Lane
from jsk_rviz_plugins.msg import OverlayText
from helpers.collision import COLLISION_POINT_CATEGORY_TO_LOCAL_PLANNER_STATUS

class PlannerStatusInfo:
    def __init__(self):

        # Publishers
        self.planner_status_pub = rospy.Publisher('planner_status', OverlayText, queue_size=1, latch=True, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('/planning/local_path', Lane, self.local_path_callback, queue_size=1, tcp_nodelay=True)


    def local_path_callback(self, msg):

        planner_status = OverlayText()
        text = "Planner: "

        if len(msg.waypoints) == 0:
            text += "<span style='color: white;'>No path</span>\n"
        else:
            text += "<span style='color: white;'>{}</span>\n".format(COLLISION_POINT_CATEGORY_TO_LOCAL_PLANNER_STATUS[msg.increment])

        planner_status.text = text
        self.planner_status_pub.publish(planner_status)


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('planner_status_info', log_level=rospy.INFO)
    node = PlannerStatusInfo()
    node.run()