#!/usr/bin/env python3
import yaml

import rospy
from geometry_msgs.msg import PoseStamped
from autoware_msgs.msg import Lane

from carla_ros_scenario_runner_types.msg import CarlaScenarioList, CarlaScenario, CarlaScenarioRunnerStatus
from carla_ros_scenario_runner_types.srv import ExecuteScenario

from helpers.geometry import get_distance_between_two_points_2d

class GoalPublisher:

    def __init__(self):

        # Node parameters
        self.map_name = rospy.get_param("~map_name")
        self.distance_to_centerline_limit = rospy.get_param("distance_to_centerline_limit")

        # Internal variables
        self.global_path_last_waypoint = None

        # Publishers
        self.available_scenarios_pub = rospy.Publisher('/carla/available_scenarios', CarlaScenarioList, queue_size=10, latch=True)
        self.goal_publisher = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10, tcp_nodelay=True, latch=True)
        self.scenario_status_publisher = rospy.Publisher('/scenario_runner/status', CarlaScenarioRunnerStatus, queue_size=10, tcp_nodelay=True, latch=True)
        
        # Subscribers
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback, queue_size=None, tcp_nodelay=True)
        rospy.Subscriber('global_path', Lane, self.global_path_callback, queue_size=None, tcp_nodelay=True)

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
    
    def goal_callback(self, msg): 
        if self.global_path_last_waypoint is None:
            return
        
        distance = get_distance_between_two_points_2d(msg.pose.position, self.global_path_last_waypoint)

        if distance > self.distance_to_centerline_limit:
            # If the last point of the global path is too far from the goal, then set scenario runner status to SHUTTINGDOWN (yellow)
            self.scenario_status_publisher.publish(CarlaScenarioRunnerStatus(3))
    
    def global_path_callback(self, msg):
        if len(msg.waypoints) > 0:
            self.global_path_last_waypoint = msg.waypoints[-1].pose.pose.position
            # Set scenario runner status to RUNNING (green)
            self.scenario_status_publisher.publish(CarlaScenarioRunnerStatus(2))
        else:
            # If the global path vanishes, then set scenario runner status to STOPPED (red)
            self.scenario_status_publisher.publish(CarlaScenarioRunnerStatus(0))
            self.global_path_last_waypoint = None

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