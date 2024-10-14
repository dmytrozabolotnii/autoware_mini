#!/usr/bin/env python3

import rospy

from std_srvs.srv import SetBool
from carla_msgs.msg import CarlaControl, CarlaStatus


class PauseBag:

    def __init__(self):

        # Publishers 
        self.carla_status_pub = rospy.Publisher('/carla/status', CarlaStatus, queue_size=10, latch=True)

        # Subscribers
        rospy.Subscriber('/carla/control', CarlaControl, self.carla_control_callback, queue_size=None)

        # Services
        rospy.wait_for_service('/player/pause_playback')
        self.pause_playback = rospy.ServiceProxy('/player/pause_playback', SetBool)

    def carla_control_callback(self, msg):
        if msg.command == CarlaControl.PLAY:
            self.publish_carla_status(True)
            response = self.pause_playback(False)
            rospy.loginfo(response.message)

        elif msg.command == CarlaControl.PAUSE:
            self.publish_carla_status(False)
            response = self.pause_playback(True)
            rospy.loginfo(response.message)
    
    def publish_carla_status(self, running):
        carla_status = CarlaStatus()
        carla_status.frame = 0
        carla_status.fixed_delta_seconds = 0.0
        carla_status.synchronous_mode = True
        carla_status.synchronous_mode_running = running

        self.carla_status_pub.publish(carla_status)

    def run(self):
        self.publish_carla_status(True)
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('pause_bag')
    node = PauseBag()
    node.run()