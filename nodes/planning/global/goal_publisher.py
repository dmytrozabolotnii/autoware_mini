#!/usr/bin/env python3

import os
import glob
import yaml

import rospy
from geometry_msgs.msg import PoseStamped

from carla_ros_scenario_runner_types.msg import CarlaScenarioList, CarlaScenario
from carla_ros_scenario_runner_types.srv import ExecuteScenario

class GoalPublisher:

    def __init__(self):

        # Node parameters
        self.map_name = rospy.get_param("~map_name")

        # Publishers
        self.available_scenarios_pub = rospy.Publisher('/carla/available_scenarios', CarlaScenarioList, queue_size=10, latch=True)
        self.goal_publisher = rospy.Publisher(
            '/move_base_simple/goal', PoseStamped, queue_size=0, tcp_nodelay=True, latch=True)

        # Services
        rospy.Service('/scenario_runner/execute_scenario', ExecuteScenario, self.publish_goal_callback)

    def publish_goal_callback(self, msg):
        with open(msg.scenario.scenario_file, 'r') as file:
            goals = yaml.safe_load_all(file)

            # Publish all goals in the yaml file
            for goal in goals:
                goal_pose = self.goal_from_yaml(goal["pose"])
                self.goal_publisher.publish(goal_pose)

        return True
    
    def goal_from_yaml(self, yaml_data):
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"

        goal.pose.position.x = yaml_data["position"]["x"]
        goal.pose.position.y = yaml_data["position"]["y"]
        goal.pose.position.z = yaml_data["position"]["z"]

        goal.pose.orientation.x = yaml_data["orientation"]["x"]
        goal.pose.orientation.y = yaml_data["orientation"]["y"]
        goal.pose.orientation.z = yaml_data["orientation"]["z"]
        goal.pose.orientation.w = yaml_data["orientation"]["w"]
        
        return goal

    def run(self):
        # Read all goals files from the given path
        goal_files = glob.glob(os.path.join(self.map_name, '*.yaml'))

        # Goal all file names
        goals_list = CarlaScenarioList()
        for goal_file in goal_files:
            scenario = CarlaScenario(
                name=os.path.basename(goal_file).split('.')[0],
                scenario_file=goal_file
            )
            goals_list.scenarios.append(scenario)

        self.available_scenarios_pub.publish(goals_list)

        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('goal_planner')
    node = GoalPublisher()
    node.run()