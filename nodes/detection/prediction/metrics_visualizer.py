#!/usr/bin/env python3

import numpy as np
import rospy
import threading
import os, sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

from autoware_msgs.msg import Lane, DetectedObjectArray, Waypoint
from jsk_rviz_plugins.msg import OverlayText
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float32

from helpers.message_cache import MessageCache


def calculate_ade(x, y):
    return np.mean(((x[:, 0] - y[:, 0]) ** 2 + (x[:, 1] - y[:, 1]) ** 2) ** 0.5)


def calculate_fde(x, y):
    return ((x[-1, 0] - y[-1, 0]) ** 2 + (x[-1, 1] - y[-1, 1]) ** 2) ** 0.5


class MetricsVisualizer:
    def __init__(self):
        self.ade_history = {}
        self.fde_history = {}
        self.cache = {}

        self.metrics_timer_duration = rospy.get_param('inference_timer')
        self.skip_points = int(rospy.get_param('step_length') / rospy.get_param('inference_timer')) - 1
        self.pad_future = int(rospy.get_param('prediction_horizon'))

        self.ade = rospy.Publisher('/dashboard/ade', Float32, queue_size=1)
        self.fde = rospy.Publisher('/dashboard/fde', Float32, queue_size=1)
        self.aware_ade = rospy.Publisher('/dashboard/ade', Float32, queue_size=1)
        self.aware_fde = rospy.Publisher('/dashboard/fde', Float32, queue_size=1)


        self.sub = rospy.Subscriber('predicted_objects', DetectedObjectArray, self.objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)
        self.self_sub = rospy.Subscriber('/planning/local_path', Lane, self.local_path_callback, queue_size=1)
        rospy.loginfo("%s - initialized", rospy.get_name())

    def objects_callback(self, detectedobjectsarray):
        for detectedobject in detectedobjectsarray.objects:
            if detectedobject.label == 'pedestrian' and len(detectedobject.candidate_trajectories.lanes) > 0:
                position = np.array([detectedobject.candidate_trajectories.lanes[0].waypoints[0].pose.pose.position.x,
                                     detectedobject.candidate_trajectories.lanes[0].waypoints[0].pose.pose.position.y])
                velocity = np.array([0, 0])
                acceleration = np.array([0, 0])
                candidate_traj_header = detectedobject.candidate_trajectories.lanes[0].header
                _id = detectedobject.id
                # Create new cached messages
                if _id not in self.cache:
                    self.cache[_id] = MessageCache(_id, position, velocity, acceleration, candidate_traj_header,
                                                   delta_t=self.metrics_timer_duration)
                    predictions = []
                    for lane in detectedobject.candidate_trajectories.lanes:
                        predictions.append([
                            (wp.pose.pose.position.x, wp.pose.pose.position.y) for wp in lane.waypoints[1:]])
                    self.cache[_id].extend_prediction_history(predictions)
                    self.ade_history[_id] = []
                    self.fde_history[_id] = []
                # Extend and update only if header stamp time attached to candidate trajectories is different
                # (means inference happened)
                else:
                    if (candidate_traj_header.stamp - self.cache[_id].return_last_header().stamp >=
                            rospy.Duration(self.metrics_timer_duration)):
                        self.cache[_id].move_endpoints()
                        self.cache[_id].update_last_trajectory(position, velocity, acceleration, candidate_traj_header)
                        predictions = []
                        for lane in detectedobject.candidate_trajectories.lanes:
                            predictions.append(
                                [(wp.pose.pose.position.x, wp.pose.pose.position.y) for wp in lane.waypoints[1:]])
                        self.cache[_id].extend_prediction_history(predictions)
                        # Check what prediction we can check for metrics
                        prediction_we_can_check = (self.cache[_id].endpoints_count
                                                   - self.pad_future * (self.skip_points + 1) + 1)
                        if prediction_we_can_check > 0:
                            # Obtain ground-truth trajectory
                            gt_trajectory = np.array(self.cache[_id].raw_trajectories[-1::-1 * (self.skip_points
                                                                                    + 1)][:self.pad_future][::-1])
                            num_of_predictions = len(self.cache[_id].prediction_history[prediction_we_can_check])
                            # print('Ground truth traj:')
                            # print(gt_trajectory)
                            temp_ade = np.zeros(num_of_predictions)
                            temp_fde = np.zeros(num_of_predictions)
                            for i in range(num_of_predictions):
                                pred_trajectory = np.array(self.cache[_id].prediction_history[prediction_we_can_check][i])
                                # print('Pred traj:')
                                # print(pred_trajectory)
                                temp_ade[i] = calculate_ade(gt_trajectory, pred_trajectory)
                                temp_fde[i] = calculate_fde(gt_trajectory, pred_trajectory)
                            # Add dynamic minADE/FDE from multiple predictions to history
                            self.ade_history[_id].append(np.min(temp_ade))
                            self.fde_history[_id].append(np.min(temp_fde))
        non_empty_ade = [self.ade_history[agent_id] for agent_id in self.ade_history
                         if len(self.ade_history[agent_id]) > 0]
        non_empty_fde = [self.fde_history[agent_id] for agent_id in self.fde_history
                         if len(self.fde_history[agent_id]) > 0]
        # Calculate global dynamic ADE/FDE by first averaging over all dynamic metrics of every agent,
        # then averaging the resulted averages
        if len(non_empty_ade) > 0 and len(non_empty_fde) > 0:
            global_ade = np.mean([np.mean(dyn_ade) for dyn_ade in non_empty_ade])
            global_fde = np.mean([np.mean(dyn_fde) for dyn_fde in non_empty_fde])
        else:
            global_ade = 0
            global_fde = 0

        self.ade.publish(Float32(global_ade))
        self.fde.publish(Float32(global_fde))

    def local_path_callback(self, lane):
        points = [waypoint.pose.pose.position for waypoint in lane.waypoints]

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('prediction_metrics_visualizer', log_level=rospy.INFO)
    node = MetricsVisualizer()
    node.run()
