#!/usr/bin/env python3

import rospy
import math
import numpy as np
import shapely
import lanelet2
from lanelet2.core import BasicPoint2d
from lanelet2.geometry import findWithin2d, length2d

from autoware_mini.msg import DetectedObjectArray, Path, Waypoint

from helpers.path import calculate_cross_track_error
from helpers.geometry import get_vector_norm_3d, get_heading_between_two_points, get_angle_between_two_headings, get_point_using_heading_and_distance
from helpers.lanelet2 import load_lanelet2_map, follow_lanelets

CAR_INDICATOR_VS_TURN_DIRECTION_SCORING = {
    'straight': {'straight': 1, 'left': 0.5, 'right': 0.5},
    'left': {'straight': 0.5, 'left': 1, 'right': -1},
    'right': {'straight': 0.5, 'left': -1, 'right': 1}
}

class MapBasedPredictor:
    def __init__(self):
        # Parameters
        self.prediction_horizon = rospy.get_param('~prediction_horizon')
        self.prediction_interval = rospy.get_param('~prediction_interval')
        self.trajectories_to_predict = rospy.get_param('~trajectories_to_predict')
        self.prediction_min_speed = rospy.get_param('~prediction_min_speed')
        self.distance_from_lanelet = rospy.get_param('~distance_from_lanelet')
        self.heading_difference_threshold = rospy.get_param('~heading_difference_threshold')
        self.use_offset_for_prediction = rospy.get_param('~use_offset_for_prediction')
        lanelet2_map_name = rospy.get_param("~lanelet2_map_name")

        # Variables
        self.lanelet2_map = load_lanelet2_map(lanelet2_map_name)
        traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany, lanelet2.traffic_rules.Participants.Vehicle)
        self.graph = lanelet2.routing.RoutingGraph(self.lanelet2_map, traffic_rules)
        num_timesteps = int(self.prediction_horizon // self.prediction_interval) + 1
        self.timesteps = np.arange(num_timesteps) * self.prediction_interval

        # Publishers
        self.predicted_objects_pub = rospy.Publisher('predicted_objects', DetectedObjectArray, queue_size=1, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('tracked_objects', DetectedObjectArray, self.tracked_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)

    def tracked_objects_callback(self, msg):

        for obj in msg.objects:
            object_speed = get_vector_norm_3d(obj.velocity)
            distance_to_object_front = obj.dimensions.x / 2
            if object_speed < self.prediction_min_speed:
                continue

            # 1. SEARCH BEST MATCHING LANELET FOR AN OBJECT
            object_position = shapely.Point(obj.position.x, obj.position.y)
            # find lanelets within distance to object_location - distance measured from lanelet borders. Inside lanelet area this distance would be 0
            lanelets_within_distance = findWithin2d(self.lanelet2_map.laneletLayer, BasicPoint2d(obj.position.x, obj.position.y), self.distance_from_lanelet)

            selected_lanelets = []
            for _, lanelet in lanelets_within_distance:
                # Skip undesired lanelets
                if lanelet.attributes["subtype"] == "crosswalk" or lanelet.attributes["subtype"] == "bus_lane":
                    continue

                # Calculate heading difference
                linestring = shapely.LineString([(p.x, p.y) for p in lanelet.centerline])
                object_distance_from_start = linestring.project(object_position)
                object_location_on_lanelet = linestring.interpolate(object_distance_from_start)
                forward_point = linestring.interpolate(object_distance_from_start + 0.1)
                lanelet_heading = get_heading_between_two_points(object_location_on_lanelet, forward_point)
                heading_difference_degrees = math.degrees(get_angle_between_two_headings(obj.heading, lanelet_heading))

                # Add lanelet if heading difference is within threshold
                if heading_difference_degrees < self.heading_difference_threshold:
                    selected_lanelets.append((lanelet, object_distance_from_start, heading_difference_degrees))

            # Sort by heading difference and limit selection to match `trajectories_to_predict`
            if len(selected_lanelets) > self.trajectories_to_predict:
                selected_lanelets.sort(key=lambda l: l[2])
                selected_lanelets = selected_lanelets[:self.trajectories_to_predict]

            # 2. CREATE ALL TRAJECTORIES
            all_trajectories = []
            if len(selected_lanelets) > 0:
                object_accel = get_vector_norm_3d(obj.acceleration)
                velocities = object_speed + object_accel * self.timesteps
                distances = (object_accel * self.timesteps**2) / 2 + object_speed * self.timesteps
                all_trajectories = self.create_trajectories(selected_lanelets, distances[-1], distance_to_object_front)

            # 3. SCORING IF NEEDED
            if len(all_trajectories) > self.trajectories_to_predict:
                trajectory_turn_directions = [[lanelet.attributes["turn_direction"] if "turn_direction" in lanelet.attributes else "straight" for lanelet in trajectory] for trajectory in all_trajectories]
                # Score each trajectory 
                # TODO use first lanelet's turn direction as object indicator, in future should be replaced by object's real indicator information
                scores = [self.score_paths(trajectory_turn_directions[i], trajectory_turn_directions[i][0]) for i in range(len(all_trajectories))]

                # Pair trajectories with their scores, sort and limit the number of trajectories to match `trajectories_to_predict`
                scored_trajectories = list(zip(all_trajectories, scores))
                scored_trajectories.sort(key=lambda t: t[1], reverse=True)
                all_trajectories = [trajectory for trajectory, _ in scored_trajectories[:self.trajectories_to_predict]]

            # 4. CREATE PREDICTIONS AND PUBLISH
            # create shapely linestring from lanelet centerlines and then use it to interpolate points in necessary distances
            for trajectory in all_trajectories:
                centerline_linestring = shapely.LineString([(p.x, p.y, p.z) for lanelet in trajectory for p in lanelet.centerline])
                if self.use_offset_for_prediction:
                    cross_track_offset = -calculate_cross_track_error(centerline_linestring, object_position)
                    trajectory_linestring = centerline_linestring.offset_curve(cross_track_offset, join_style="mitre")
                else:
                    trajectory_linestring = centerline_linestring

                # get prediction origin right in front of the object
                object_front = get_point_using_heading_and_distance(obj.position, obj.heading, distance_to_object_front)
                object_front = shapely.Point(object_front.x, object_front.y, object_front.z)
                object_front_distance_from_trajectory_linestring_start = trajectory_linestring.project(object_front)

                interpolate_distances = distances + object_front_distance_from_trajectory_linestring_start
                # interpolate_distances extend further than trajectory_linestring (case of dangling lanelets and sometimes also offset curve might reduce
                # its length), so clip the exessive distances otherwise duplicate points cause problems later with triangulation
                if interpolate_distances[-1] > trajectory_linestring.length:
                    index = np.argmax(interpolate_distances > trajectory_linestring.length)
                    # adding 1 to include first point past the trajectory length - will be interpolated to the very end of it
                    interpolate_distances = interpolate_distances[:index + 1]

                # for offset curve z is not available, therefore taken from the centerline
                points_centerline = centerline_linestring.interpolate(interpolate_distances)
                if self.use_offset_for_prediction:
                    points_offset = trajectory_linestring.interpolate(interpolate_distances)

                path = Path()
                for i, d in enumerate(interpolate_distances):
                    wp = Waypoint()
                    if self.use_offset_for_prediction:
                        wp.position.x = points_offset[i].x
                        wp.position.y = points_offset[i].y
                    else:
                        wp.position.x = points_centerline[i].x
                        wp.position.y = points_centerline[i].y
                    wp.position.z = points_centerline[i].z
                    wp.speed = velocities[i]
                    path.waypoints.append(wp)
                obj.candidate_trajectories.paths.append(path)

        # Publish predicted objects
        self.predicted_objects_pub.publish(msg)

    def create_trajectories(self, start_lanelets, prediction_length, distance_to_object_front):
        all_trajectories = []
        heading_differences = []
        for start_lanelet, object_distance_from_start, heading_difference in start_lanelets:
            prediction_length_from_start_lanelet = prediction_length + object_distance_from_start + distance_to_object_front
            # explore following lanelets recursively
            trajectories = follow_lanelets(self.graph, start_lanelet, prediction_length_from_start_lanelet)
            # append trajectory if start point is not further than returned trajectory length
            for trajectory in trajectories:
                d = 0
                for lanelet in trajectory:
                    d += length2d(lanelet)
                    if d > object_distance_from_start + distance_to_object_front:
                        all_trajectories.append(trajectory)
                        heading_differences.append(heading_difference)
                        break

        # If there are multiple trajectories that end in the same lanelet, keep the one with the smallest heading difference (better match)
        best_trajectories = {}
        for heading_difference, trajectory in zip(heading_differences, all_trajectories):
            end_lanelet = trajectory[-1]  # Get the last lanelet
            # If the end_id is not in the dictionary or the new heading difference is smaller, update the dictionary
            if end_lanelet.id not in best_trajectories or heading_difference < best_trajectories[end_lanelet.id][0]:
                best_trajectories[end_lanelet.id] = (heading_difference, trajectory)
        # Extract the filtered trajectories
        filtered_trajectories = [item[1] for item in best_trajectories.values()]

        return filtered_trajectories

    def score_paths(self, path, object_indicator):
        path_score = 0
        for i, turn in enumerate(path):
            # score the lanelet according to how well it matches the object indicator
            lanelet_score = CAR_INDICATOR_VS_TURN_DIRECTION_SCORING[object_indicator][turn]
            if i > 0:
                # discount farther lanelets
                lanelet_score /= i
            # path score is sum of lanelet scores
            path_score += lanelet_score
        return path_score

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('map_based_predictor', log_level=rospy.INFO)
    node = MapBasedPredictor()
    node.run()