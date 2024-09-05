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

            for goal in goals:
                if goal["name"] == msg.scenario.name:
                    for position in goal["position"]:
                        goal_pose = PoseStamped()
                        goal_pose.header.stamp = rospy.Time.now()
                        goal_pose.header.frame_id = "map"

                        goal_pose.pose.position.x = position["x"]
                        goal_pose.pose.position.y = position["y"]
                        goal_pose.pose.position.z = position["z"]

                        goal_pose.pose.orientation.x = 0
                        goal_pose.pose.orientation.y = 0
                        goal_pose.pose.orientation.z = 0
                        goal_pose.pose.orientation.w = 0

                        self.goal_publisher.publish(goal_pose)
                        
                        return True

        return False
    
    def goal_from_yaml(self, yaml_data):
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"

        goal.pose.position.x = yaml_data["position"]["x"]
        goal.pose.position.y = yaml_data["position"]["y"]
        goal.pose.position.z = yaml_data["position"]["z"]

        goal.pose.orientation.x = 0
        goal.pose.orientation.y = 0
        goal.pose.orientation.z = 0
        goal.pose.orientation.w = 0
        
        return goal

    def run(self):
        goals_list = CarlaScenarioList()

        with open(self.map_name, 'r') as file:
            goals = yaml.safe_load_all(file)
        
            for goal in goals:
                scenario = CarlaScenario(
                    name=goal["name"],
                    scenario_file=self.map_name
                )
                goals_list.scenarios.append(scenario)

        self.available_scenarios_pub.publish(goals_list)

        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('goal_planner')
    node = GoalPublisher()
    node.run()