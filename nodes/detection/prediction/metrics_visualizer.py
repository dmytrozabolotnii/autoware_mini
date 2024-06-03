#!/usr/bin/env python3

import numpy as np
import rospy
import threading
import os, sys, time
import os.path as osp
import bisect
from scipy.special import softmax

from autoware_msgs.msg import Lane, DetectedObjectArray, Waypoint
from jsk_rviz_plugins.msg import OverlayText
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float32
from shapely import LineString, prepare

from helpers.message_cache import MessageCache
from helpers.geometry import get_distance_between_two_points_2d


def calculate_ade(x, y):
    return np.mean(((x[:, 0] - y[:, 0]) ** 2 + (x[:, 1] - y[:, 1]) ** 2) ** 0.5)


def calculate_fde(x, y):
    return ((x[-1, 0] - y[-1, 0]) ** 2 + (x[-1, 1] - y[-1, 1]) ** 2) ** 0.5


def calculate_ade_grad(x, y, danger_zone=10.0):
    distance_between_points = (np.linalg.norm(x - y))

    return np.sum(np.interp(distance_between_points, [danger_zone, danger_zone * 2], [1, 0]))


class MetricsVisualizer:
    def __init__(self):
        self.lock = threading.Lock()
        self.ade_history = {}
        self.ade_grad_history = {}
        self.aware_ade_history = {}
        self.fde_history = {}
        self.cache = {}
        self.planned_local_path_cache = {}

        self.metrics_timer_duration = rospy.get_param('inference_timer')
        self.skip_points = int(rospy.get_param('step_length') / rospy.get_param('inference_timer')) - 1
        self.pad_future = int(rospy.get_param('prediction_horizon'))
        self.bagscenarioname = rospy.get_param('~bag_file')[:-4]
        self.category_name = '_new_fixed_tracker'
        self.dir_name = self.bagscenarioname + self.category_name
        self.predictorname = rospy.get_param('~predictor')
        self.csvfilename = osp.join(rospy.get_param('~csv_file_result'), self.category_name, self.dir_name, self.dir_name + '_' + self.predictorname + '_' + str(time.time()) + '.csv')
        if not osp.exists(osp.join(rospy.get_param('~csv_file_result'), self.dir_name)):
            os.makedirs(osp.join(rospy.get_param('~csv_file_result'), self.dir_name))

        self.result_log = []
        self.ade = rospy.Publisher('/dashboard/ade', Float32, queue_size=1)
        self.fde = rospy.Publisher('/dashboard/fde', Float32, queue_size=1)
        self.aware_ade = rospy.Publisher('/dashboard/aware_ade', Float32, queue_size=1)
        self.aware_fde = rospy.Publisher('/dashboard/aware_fde', Float32, queue_size=1)

        self.sub = rospy.Subscriber('predicted_objects', DetectedObjectArray, self.objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)
        self.local_path_sub = rospy.Subscriber('/planning/local_path', Lane, self.local_path_callback, queue_size=1)
        rospy.on_shutdown(self.shutdown)
        with open(self.csvfilename, 'w') as file:
            file.write('stamp,ade,fde,aware_ade,n_ped')
            file.write('\n')

        rospy.loginfo("%s - initialized", rospy.get_name())

    def objects_callback(self, detectedobjectsarray):
        with self.lock:
            local_planned_local_path_cache = self.planned_local_path_cache

        ade_grad_dict = {}
        header_stamp = detectedobjectsarray.header.stamp
        n_ped = 0
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
                    self.cache[_id].extend_prediction_header_history(candidate_traj_header)
                    self.ade_history[_id] = []
                    self.fde_history[_id] = []
                    self.ade_grad_history[_id] = []
                    self.aware_ade_history[_id] = []

                # Extend and update only if header stamp time attached to candidate trajectories is different
                # (means inference happened)
                else:
                    if (candidate_traj_header.stamp - self.cache[_id].return_last_header().stamp >=
                            rospy.Duration(self.metrics_timer_duration)):
                        if (candidate_traj_header.stamp - self.cache[_id].return_last_header().stamp >=
                              4 * rospy.Duration(self.metrics_timer_duration)):
                            rospy.logwarn_throttle(3, "%s - Predictions lagging behind messages",
                                                   rospy.get_name())
                        self.cache[_id].move_endpoints()
                        self.cache[_id].update_last_trajectory(position, velocity, acceleration, candidate_traj_header)
                        predictions = []
                        for lane in detectedobject.candidate_trajectories.lanes:
                            predictions.append(
                                [(wp.pose.pose.position.x, wp.pose.pose.position.y) for wp in lane.waypoints[1:]])
                        self.cache[_id].extend_prediction_history(predictions)
                        self.cache[_id].extend_prediction_header_history(candidate_traj_header)
                        # Check what prediction we can check for metrics
                        prediction_we_can_check = (self.cache[_id].endpoints_count
                                                   - self.pad_future * (self.skip_points + 1) + 1)
                        if prediction_we_can_check > 0:
                            # Obtain ground-truth trajectory
                            gt_trajectory = np.array(self.cache[_id].raw_trajectories[-1::-1 * (self.skip_points
                                                                                    + 1)][:self.pad_future][::-1])
                            num_of_predictions = len(self.cache[_id].prediction_history[prediction_we_can_check])
                            # gt_trajectory_linestring = LineString(gt_trajectory)
                            # minx, miny, maxx, maxy = gt_trajectory_linestring.bounds
                            # if (maxx - minx < 2.0) and (maxy - miny < 2.0):
                            #     print('Bounding box of ground truth movement of pedestrian', _id, 'too small, skipping')
                            #     continue
                            n_ped += 1
                            # print('Ground truth traj:')
                            # print(gt_trajectory)
                            # Obtain closest planned local path according to stamp
                            if len(list(local_planned_local_path_cache.keys())) > 0:
                                header_pred_trajectory = self.cache[_id].predictions_history_headers[
                                    prediction_we_can_check]
                                index = bisect.bisect_left(list(local_planned_local_path_cache.keys()), header_pred_trajectory.stamp) - 1
                                stamp_of_planned_local_path_at_stamp = list(local_planned_local_path_cache.keys())[index]
                                if (stamp_of_planned_local_path_at_stamp - header_pred_trajectory.stamp
                                        <= rospy.Duration(self.metrics_timer_duration) * 2):
                                    planned_local_path_at_stamp = local_planned_local_path_cache[stamp_of_planned_local_path_at_stamp]
                                    planned_local_path_at_stamp = np.asarray(planned_local_path_at_stamp.coords)
                                else:
                                    planned_local_path_at_stamp = None
                            else:
                                planned_local_path_at_stamp = None
                            temp_ade = np.zeros(num_of_predictions)
                            temp_fde = np.zeros(num_of_predictions)
                            temp_ade_grad = np.zeros(num_of_predictions)
                            for i in range(num_of_predictions):
                                pred_trajectory = np.array(self.cache[_id].prediction_history[prediction_we_can_check][i])
                                # print('Pred traj:')
                                # print(pred_trajectory)
                                # Calculate grad for every planned trajectory separately
                                if planned_local_path_at_stamp is not None:
                                    temp_ade_grad[i] = calculate_ade_grad(pred_trajectory, planned_local_path_at_stamp)
                                temp_ade[i] = calculate_ade(gt_trajectory, pred_trajectory)
                                temp_fde[i] = calculate_fde(gt_trajectory, pred_trajectory)
                            # Add dynamic minADE/FDE from multiple predictions to history
                            self.ade_history[_id].append(np.min(temp_ade))
                            self.fde_history[_id].append(np.min(temp_fde))
                            # Store sum of gradients of this one agent
                            if planned_local_path_at_stamp is not None:
                                self.ade_grad_history[_id].append(np.sum(temp_ade_grad))
                            else:
                                self.ade_grad_history[_id].append(0.0)

                            ade_grad_dict[_id] = (self.ade_grad_history[_id][-1])
        if len(ade_grad_dict) > 0:
            ade_grad_normalized_values = np.array(list(ade_grad_dict.values()))
            ade_grad_normalized_values = ade_grad_normalized_values / np.sum(ade_grad_normalized_values) \
                if np.sum(ade_grad_normalized_values) > 0 else np.zeros_like(ade_grad_normalized_values)
            ade_grad_list_softmax_dict = {_id: ade_grad_normalized_values[i] for i, _id in enumerate(ade_grad_dict)}
            # Find task-aware ade after we calculated grad softmax for every agent's predictions
            for _id, softmax_value in ade_grad_list_softmax_dict.items():
                self.aware_ade_history[_id].append(self.ade_history[_id][-1] *
                                                   (1 + ade_grad_list_softmax_dict[_id]))

        non_empty_ade = [self.ade_history[agent_id] for agent_id in self.ade_history
                         if len(self.ade_history[agent_id]) > 0]
        non_empty_fde = [self.fde_history[agent_id] for agent_id in self.fde_history
                         if len(self.fde_history[agent_id]) > 0]
        non_empty_aware_ade = [self.aware_ade_history[agent_id] for agent_id in self.aware_ade_history
                               if len(self.aware_ade_history[agent_id]) > 0]
        # Calculate global dynamic ADE/FDE by first averaging over all dynamic metrics of every agent,
        # then averaging the resulted averages
        if len(non_empty_ade) > 0 and len(non_empty_fde) > 0:
            global_ade = np.mean([np.mean(dyn_ade) for dyn_ade in non_empty_ade])
            global_fde = np.mean([np.mean(dyn_fde) for dyn_fde in non_empty_fde])
            global_aware_ade = np.mean([np.mean(dyn_aware_ade) for dyn_aware_ade in non_empty_aware_ade])
        else:
            global_ade = 0
            global_fde = 0
            global_aware_ade = 0

        self.ade.publish(Float32(global_ade))
        self.fde.publish(Float32(global_fde))
        self.aware_ade.publish(Float32(global_aware_ade))
        self.result_log.append(','.join([str(header_stamp),
                                         str(global_ade), str(global_fde), str(global_aware_ade), str(n_ped)]))

    def local_path_callback(self, lane):
        # Calculate planned local path from the autoware message
        # and save it to cache if its newer than previous message for duration

        points = [waypoint.pose.pose.position for waypoint in lane.waypoints]
        if (len(points) > 1) and ((not any(self.planned_local_path_cache)) or (lane.header.stamp - list(self.planned_local_path_cache.keys())[-1] >=
                            rospy.Duration(self.metrics_timer_duration))):
            linepoints = LineString([(point.x, point.y) for point in points])
            prepare(linepoints)
            speeds = [waypoint.twist.twist.linear.x for waypoint in lane.waypoints]

            dist_between_points = np.array([get_distance_between_two_points_2d(points[i], points[i + 1]) for i in range(len(points) - 1) if range(len(points) > 1)])
            avg_speed_between_points = np.array([(speeds[i] + speeds[i + 1]) / 2 for i in range(len(speeds) - 1) if range(len(speeds) > 1)])
            times_between_points = dist_between_points / avg_speed_between_points
            cum_times_between_points = np.cumsum(times_between_points)
            future_metric_time = np.arange(0, self.metrics_timer_duration * self.pad_future, self.metrics_timer_duration)
            velocity_interpolated_values = np.interp(future_metric_time, cum_times_between_points, avg_speed_between_points)
            future_distances = np.array([np.trapz(velocity_interpolated_values[:i + 1], future_metric_time[:i + 1]) for i in range(self.pad_future)])
            expected_trajectory = LineString([linepoints.interpolate(future_distances[i]) for i in range(self.pad_future)])

            with self.lock:
                self.planned_local_path_cache[lane.header.stamp] = expected_trajectory

        # print(self.planned_local_path_cache)

    def shutdown(self):
        with open(self.csvfilename, 'a') as file:
            for line in self.result_log:
                file.write(line)
                file.write('\n')

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('prediction_metrics_visualizer', log_level=rospy.INFO)
    node = MetricsVisualizer()
    node.run()
