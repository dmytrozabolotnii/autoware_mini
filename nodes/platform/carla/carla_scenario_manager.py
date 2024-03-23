#!/usr/bin/env python3
#
# Copyright (c) 2023 Autonomous Driving Lab (ADL), University of Tartu.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import os
import glob
import pathlib
from datetime import datetime
import rospy

from std_msgs.msg import Empty
from carla_ros_scenario_runner_types.msg import CarlaScenarioRunnerStatus, CarlaScenario
from carla_ros_scenario_runner_types.srv import ExecuteScenario


class CarlaScenarioManager:

    def __init__(self):

        # Node parameters
        self.scenario_runner_root = rospy.get_param("~scenario_runner_root")
        scenario_name = rospy.get_param("~scenario_name", default='')
        scenario_path = rospy.get_param("~scenario_path")
        self.map_name = rospy.get_param("~map_name")
        self.save_results = rospy.get_param("~save_results", default=True)

        if scenario_name == '':
            rospy.loginfo("%s Scenario name is not given. Running all scenarios of %s map.",
                          rospy.get_name(), self.map_name)

            # Read all scenario files from the given path
            self.scenario_files = glob.glob(os.path.join(scenario_path, '*.xosc'))
        else:
            rospy.loginfo("%s Scenario name is given. Running only %s scenario of %s map.",
                            rospy.get_name(), scenario_name, self.map_name)

            # Read the given scenario file
            self.scenario_files = [os.path.join(scenario_path, scenario_name + '.xosc')]
        
        self.scenario_counter = 0

        # Publishers
        self.cancel_route = rospy.Publisher('/planning/cancel_route', Empty, queue_size=1, tcp_nodelay=True)

        rospy.wait_for_service('/scenario_runner/execute_scenario')
        self.execute_scenario = rospy.ServiceProxy('/scenario_runner/execute_scenario', ExecuteScenario)

        # Subscribers
        rospy.Subscriber("/scenario_runner/status", CarlaScenarioRunnerStatus, self.current_status, queue_size=1, tcp_nodelay=True)

    def current_status(self, msg):
        """
        callback current scenario status
        """

        if msg.status == CarlaScenarioRunnerStatus.STOPPED:

            if self.scenario_counter < len(self.scenario_files):
                rospy.loginfo("%s Starting scenario: %s", rospy.get_name(), self.scenario_files[self.scenario_counter])

                try:
                    # Cancel current route to make sure the car is not moving
                    self.cancel_route.publish(Empty())
                    rospy.sleep(1.0)

                    # Call scenario runner service
                    scenario_name = pathlib.Path(self.scenario_files[self.scenario_counter]).stem
                    carla_scenario = CarlaScenario(scenario_name, self.scenario_files[self.scenario_counter])
                    resp = self.execute_scenario(carla_scenario)

                    if resp:
                        rospy.loginfo("%s - Scenario started successfully!", rospy.get_name())
                    else:
                        rospy.logerr("%s - Scenario responded as failed!", rospy.get_name())

                except rospy.ServiceException as e:
                    rospy.logerr("%s Service call failed: %s", rospy.get_name(), str(e))

                self.scenario_counter += 1

            else:
                # Compile all results
                if self.save_results:
                    self.compile_and_merge()

                rospy.loginfo("%s - All scenarios are executed. Shutting down scenario manager.", rospy.get_name())
                rospy.signal_shutdown("All scenarios are executed.")

    def compile_and_merge(self):
        """
        Save all scenario results to a single file
        """
        # Get all result files
        result_files = glob.glob(self.scenario_runner_root + '/results/*.txt')

        # Path of today's benchmarking results
        out_path = os.path.join(self.scenario_runner_root, 'results/', datetime.now().strftime('%Y-%m-%d'))

        # Create the directory if it does not exist
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        output_file = os.path.join(out_path, self.map_name + '_results.txt')

        # Open the output file in write mode
        with open(output_file, "w") as combined_file:
            for file_name in result_files:
                try:
                    # Open each input file in read mode
                    with open(file_name, "r") as input_file:
                        # Read the contents of the input file
                        file_contents = input_file.read()
                        # Write the contents to the output file
                        combined_file.write(file_contents)
                        # Add a separator
                        combined_file.write("\n\n")
                    # Remove the original file
                    os.remove(file_name)
                    rospy.loginfo("%s - File removed: %s", rospy.get_name(), file_name)
                except FileNotFoundError:
                    rospy.logerr("%s - File not found: %s", rospy.get_name(), file_name)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('carla_scenario_manager', log_level=rospy.INFO)
    node = CarlaScenarioManager()
    node.run()
