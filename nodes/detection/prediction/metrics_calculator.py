import bisect
import os
import threading
import time
from os import path as osp

import rospy
import numpy as np
from shapely import Polygon, LineString, prepare
from autoware_mini.geometry import get_distance_between_two_points_2d
from std_msgs.msg import Float32
from autoware_mini.msg import Path

import lanelet2
from lanelet2.core import BasicPoint2d
from lanelet2.geometry import findWithin2d
from autoware_mini.lanelet2 import load_lanelet2_map

def calculate_ade(x, y):
    return np.mean(((x[:, 0] - y[:, 0]) ** 2 + (x[:, 1] - y[:, 1]) ** 2) ** 0.5)


def calculate_fde(x, y):
    return ((x[-1, 0] - y[-1, 0]) ** 2 + (x[-1, 1] - y[-1, 1]) ** 2) ** 0.5


def calculate_ade_grad(x, y, danger_zone=10.0):
    distance_between_points = (np.linalg.norm(x - y))

    return np.sum(np.interp(distance_between_points, [danger_zone, danger_zone * 2], [1, 0]))

def calculate_dac(lanelet2_map, predictions, danger_zone=10.0):
    # search matching lanelets to a initial point of all predictions
    x, y = float(predictions[0][0, 0]), float(predictions[0][0, 1])
    object_location = BasicPoint2d(x, y)
    # find lanelets within distance to object_location - distance measured from lanelet borders
    lanelets_within_distance = findWithin2d(lanelet2_map.laneletLayer, object_location, danger_zone)
    crosswalks = []
    lanelets = []

    for d, lanelet in lanelets_within_distance:
        if lanelet.attributes and lanelet.attributes["subtype"] == 'crosswalk':
            crosswalk = Polygon([(p.x, p.y) for p in lanelet.polygon2d()])
            prepare(crosswalk)
            crosswalks.append(crosswalk)
        else:
            lanelet = Polygon([(p.x, p.y) for p in lanelet.polygon2d()])
            prepare(lanelet)
            lanelets.append(lanelet)
    m = 0.0
    n = len(predictions)

    for prediction in predictions:
        prediction_linestring  = LineString(np.array(prediction)[1:])
        on_crosswalk = False
        for crosswalk in crosswalks:
            if crosswalk.intersects(prediction_linestring):
                on_crosswalk = True
                break
        if not on_crosswalk:
            for lanelet in lanelets:
                if lanelet.intersects(prediction_linestring):
                    m += 1
                    break

    return (n - m) / n

MR_LIMIT = 2.0

class MetricsCalculator:
    def __init__(self):
        self.lock = threading.Lock()

        self.ade_history = {}
        self.ade_grad_history = {}
        self.aware_ade_history = {}
        self.fde_history = {}
        self.mr_history = {}
        self.dac_history = {}
        self.presence_time_cache = {}
        self.pedestrian_with_head_pose_count = 0
        # self.cache = cache
        lanelet2_map_name = rospy.get_param("/planning/lanelet2_global_planner/lanelet2_map_path")
        self.lanelet2_map = load_lanelet2_map(lanelet2_map_name)
        self.planned_local_path_cache = {}

        self.metrics_timer_duration = rospy.get_param('inference_timer')
        self.skip_points = int(rospy.get_param('step_length') / rospy.get_param('inference_timer')) - 1
        self.pad_future = int(rospy.get_param('prediction_horizon'))
        self.bagscenarioname = rospy.get_param('bag_file')[:-4]
        self.category_name = rospy.get_param('results_category_folder')
        self.dir_name = self.bagscenarioname + self.category_name
        self.predictorname = rospy.get_param('predictor')
        self.csvfilename = osp.join(rospy.get_param('csv_file_result'), self.category_name, self.dir_name, self.dir_name + '_' + self.predictorname + '_' + str(time.time()) + '.csv')
        if not osp.exists(osp.join(rospy.get_param('csv_file_result'), self.category_name, self.dir_name)):
            os.makedirs(osp.join(rospy.get_param('csv_file_result'), self.category_name, self.dir_name))

        self.result_log = []
        self.ade = rospy.Publisher('/dashboard/ade', Float32, queue_size=1)
        self.fde = rospy.Publisher('/dashboard/fde', Float32, queue_size=1)
        self.aware_ade = rospy.Publisher('/dashboard/aware_ade', Float32, queue_size=1)
        self.mr = rospy.Publisher('/dashboard/mr', Float32, queue_size=1)
        self.dac = rospy.Publisher('/dashboard/dac', Float32, queue_size=1)

        # self.sub = rospy.Subscriber('predicted_objects', DetectedObjectArray, self.objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)
        self.local_path_sub = rospy.Subscriber('/planning/local_path', Path, self.local_path_callback, queue_size=1)
        rospy.on_shutdown(self.shutdown)
        with open(self.csvfilename, 'w') as file:
            file.write('stamp,ade,fde,aware_ade,mr,dac,n_ped_max,n_ped_total,ped_avg_time')
            file.write('\n')

        rospy.loginfo("%s - initialized", rospy.get_name())

    def calculate_metrics(self, cache):
        with self.lock:
            local_planned_local_path_cache = self.planned_local_path_cache

        ade_grad_dict = {}
        header_stamp = next(iter(cache.values())).return_last_header().stamp
        n_ped = 0
        for _id, message in cache.items():
            if _id not in self.ade_history:
                self.ade_history[_id] = []
                self.fde_history[_id] = []
                self.ade_grad_history[_id] = []
                self.aware_ade_history[_id] = []
                self.mr_history[_id] = []
                self.dac_history[_id] = []
                self.presence_time_cache[_id] = 0
                if message.label == 'pedestrian_with_head_pose':
                    self.pedestrian_with_head_pose_count += 1

            header = message.return_last_header()
            # Update presence time of pedestrian
            if message.label == 'pedestrian_with_head_pose':
                self.presence_time_cache[_id] = (header.stamp - message.headers[0].stamp).to_sec() + 0.05
            # Check what prediction we can check for metrics
            prediction_we_can_check = 0
            for i, prediction_header in enumerate(message.predictions_history_headers[1:]):
                if abs((header.stamp - prediction_header.stamp).to_sec() - self.pad_future * self.metrics_timer_duration) < self.metrics_timer_duration:
                    prediction_we_can_check = i + 1
                    break
            if prediction_we_can_check > 0:
                # Obtain ground-truth trajectory
                gt_trajectory = message.return_last_interpolated_trajectory(self.pad_future, self.metrics_timer_duration)
                num_of_predictions = len(message.prediction_history[prediction_we_can_check])
                if message.label == 'pedestrian_with_head_pose':
                    n_ped += 1
                # Obtain closest planned local path according to stamp
                if len(list(local_planned_local_path_cache.keys())) > 0:
                    header_pred_trajectory = message.predictions_history_headers[
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
                temp_dac = calculate_dac(self.lanelet2_map, message.prediction_history[prediction_we_can_check])
                for i in range(num_of_predictions):
                    pred_trajectory = np.array(message.prediction_history[prediction_we_can_check][i])[1:]
                    # Calculate grad for every planned trajectory separately
                    if planned_local_path_at_stamp is not None:
                        temp_ade_grad[i] = calculate_ade_grad(pred_trajectory, planned_local_path_at_stamp)
                    temp_ade[i] = calculate_ade(gt_trajectory, pred_trajectory)
                    temp_fde[i] = calculate_fde(gt_trajectory, pred_trajectory)
                # Add dynamic minADE/FDE from multiple predictions to history
                self.ade_history[_id].append(np.min(temp_ade))
                self.fde_history[_id].append(np.min(temp_fde))
                self.dac_history[_id].append(temp_dac)
                if np.min(temp_fde) <= MR_LIMIT:
                    self.mr_history[_id].append(0.0)
                else:
                    self.mr_history[_id].append(1.0)
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
        non_empty_mr = [self.mr_history[agent_id] for agent_id in self.mr_history
                         if len(self.mr_history[agent_id]) > 0]
        non_empty_dac = [self.dac_history[agent_id] for agent_id in self.dac_history
                         if len(self.dac_history[agent_id]) > 0]
        # Calculate global dynamic ADE/FDE by first averaging over all dynamic metrics of every agent,
        # then averaging the resulted averages
        if len(non_empty_ade) > 0 and len(non_empty_fde) > 0:
            global_ade = np.mean([np.mean(dyn_ade) for dyn_ade in non_empty_ade])
            global_fde = np.mean([np.mean(dyn_fde) for dyn_fde in non_empty_fde])
            global_aware_ade = np.mean([np.mean(dyn_aware_ade) for dyn_aware_ade in non_empty_aware_ade])
            global_mr = np.mean([np.mean(dyn_mr) for dyn_mr in non_empty_mr])
            global_dac = np.mean([np.mean(dyn_dac) for dyn_dac in non_empty_dac])
        else:
            global_ade = 0
            global_fde = 0
            global_aware_ade = 0
            global_mr = 0
            global_dac = 1

        self.ade.publish(Float32(global_ade))
        self.fde.publish(Float32(global_fde))
        self.aware_ade.publish(Float32(global_aware_ade))
        self.mr.publish(Float32(global_mr))
        self.dac.publish(Float32(global_dac))
        self.result_log.append(','.join([str(header_stamp),
                                         str(global_ade), str(global_fde), str(global_aware_ade), str(global_mr), str(global_dac), str(n_ped), str(self.pedestrian_with_head_pose_count), str(np.mean(list(self.presence_time_cache.values())))]))

    def local_path_callback(self, lane):
        # Calculate planned local path from the autoware message
        # and save it to cache if its newer than previous message for duration

        points = [waypoint.position for waypoint in lane.waypoints]
        if (len(points) > 1) and ((not any(self.planned_local_path_cache)) or (lane.header.stamp - list(self.planned_local_path_cache.keys())[-1] >=
                            rospy.Duration(self.metrics_timer_duration))):
            linepoints = LineString([(point.x, point.y) for point in points])
            prepare(linepoints)
            speeds = [waypoint.speed for waypoint in lane.waypoints]

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
