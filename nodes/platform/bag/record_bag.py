#!/usr/bin/env python3

import os
import rospy
import subprocess
from datetime import datetime

from jsk_rviz_plugins.msg import RecordCommand

class RecordBag:
    def __init__(self):

        # Parameters
        self.blacklist_file = rospy.get_param("~blacklist_file")
        self.recorded_bags_dir = rospy.get_param("~recorded_bags_dir")

        os.makedirs(self.recorded_bags_dir, exist_ok=True)
        self.recording_process = None

        # Subsrcibers
        rospy.Subscriber('/record_command', RecordCommand, self.record_bag_callback, queue_size=1, tcp_nodelay=True)

    
    def record_bag_callback(self, msg):
        if msg.command == 0 and self.recording_process is None:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            output_file = f"{timestamp}_{msg.target}"

            # Blacklist file command
            blacklist_cmd = f"grep -v -P '^#(.*)' {self.blacklist_file} | xargs | sed -e 's/ /|/g'"

            # Full command
            cmd = f"rosbag record -a -O {output_file} -x \"$({blacklist_cmd})\""

            self.recording_process = subprocess.Popen(cmd, shell=True, executable="/bin/bash", cwd=self.recorded_bags_dir)

        elif msg.command == 1 and self.recording_process is not None:
            self.recording_process.terminate()
            self.recording_process.wait()
            self.recording_process = None

            rospy.loginfo("Stop recording bag")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('record_bag')
    node = RecordBag()
    node.run()