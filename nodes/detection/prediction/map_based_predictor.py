#!/usr/bin/env python3

import rospy
import math
import numpy as np
import shapely
import lanelet2
from lanelet2.core import BasicPoint2d
from lanelet2.geometry import findWithin2d

from autoware_mini.msg import DetectedObjectArray, Path, Waypoint

from helpers.path import calculate_cross_track_error
from helpers.geometry import get_vector_norm_3d, get_heading_between_two_points, get_angle_between_two_headings, get_point_using_heading_and_distance
from helpers.lanelet2 import load_lanelet2_map

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
        self.angle_threshold = rospy.get_param('~angle_threshold')
        self.use_offset_for_prediction = rospy.get_param('~use_offset_for_prediction')

        lanelet2_map_name = rospy.get_param("~lanelet2_map_name")

        self.lanelet2_map = load_lanelet2_map(lanelet2_map_name)
        traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany, lanelet2.traffic_rules.Participants.Vehicle)
        self.graph = lanelet2.routing.RoutingGraph(self.lanelet2_map, traffic_rules)

        # Publishers
        self.predicted_objects_pub = rospy.Publisher('predicted_objects', DetectedObjectArray, queue_size=1, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('tracked_objects', DetectedObjectArray, self.tracked_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)

    def tracked_objects_callback(self, msg):
        num_timesteps = int(self.prediction_horizon // self.prediction_interval) + 1
        timesteps = np.arange(num_timesteps) * self.prediction_interval

        for obj in msg.objects:
            object_speed = get_vector_norm_3d(obj.velocity)
            if object_speed < self.prediction_min_speed:
                continue

            # 1. SEARCH BEST MATCHING LANELET FOR AN OBJECT
            selected_lanelets = []
            object_centroid = shapely.Point(obj.position.x, obj.position.y)
            # find lanelets within distance to object_location - distance measured from lanelet borders. Inside lanelet area this distance would be 0
            lanelets_within_distance = findWithin2d(self.lanelet2_map.laneletLayer, BasicPoint2d(obj.position.x, obj.position.y), self.distance_from_lanelet)

            for _, lanelet in lanelets_within_distance:
                # Skip undesired lanelets
                if lanelet.attributes["subtype"] == "crosswalk" or lanelet.attributes["subtype"] == "bus_lane":
                    continue

                # Calculate angle difference
                linestring = shapely.LineString([(p.x, p.y) for p in lanelet.centerline])
                lanelet_length = linestring.length
                object_distance_from_start = linestring.project(object_centroid)
                object_location_on_lanelet = linestring.interpolate(object_distance_from_start)
                forward_point = linestring.interpolate(object_distance_from_start + 0.1)
                lanelet_heading = get_heading_between_two_points(object_location_on_lanelet, forward_point)
                heading_difference_degrees = math.degrees(get_angle_between_two_headings(obj.heading, lanelet_heading))

                # Add lanelet if angle difference is within threshold
                if heading_difference_degrees < self.angle_threshold:
                    selected_lanelets.append((lanelet, object_distance_from_start, lanelet_length, heading_difference_degrees))

            # Sort by heading angle difference and limit selection to match `trajectories_to_predict`
            if len(selected_lanelets) > self.trajectories_to_predict:
                selected_lanelets.sort(key=lambda l: l[3])
                selected_lanelets = selected_lanelets[:self.trajectories_to_predict]

            # 2. CREATE ALL TRAJECTORIES
            all_trajectories = []
            if len(selected_lanelets) > 0:
                object_accel = get_vector_norm_3d(obj.acceleration)
                velocities = object_speed + object_accel * timesteps
                distances = (object_accel * timesteps**2) / 2 + object_speed * timesteps

                for lanelet, object_distance_from_start, lanelet_length, _ in selected_lanelets:
                    # routingCostLimit, elementLimit, routingCostId, includeLaneChanges, includeShorterPaths
                    params = lanelet2.routing.PossiblePathsParams(lanelet_length + distances[-1], 100, 0, False, True)
                    all_trajectories_from_lanelet = self.graph.possiblePaths(lanelet, params)
                    all_trajectories.extend(all_trajectories_from_lanelet)

            # 3. SCORING IF NEEDED
            if len(all_trajectories) > self.trajectories_to_predict:
                trajectory_turn_directions = [[lanelet.attributes["turn_direction"] if "turn_direction" in lanelet.attributes else "straight" for lanelet in trajectory] for trajectory in all_trajectories]
                # Score each trajectory 
                # TODO use first lanelet's turn direction as object indicator, in future should be replaced by object's real indicator information
                scores = [self.evaluate_paths(trajectory_turn_directions[i], trajectory_turn_directions[i][0]) for i in range(len(all_trajectories))]

                # Pair trajectories with their scores, sort and limit the number of trajectories to match `trajectories_to_predict`
                scored_trajectories = list(zip(all_trajectories, scores))
                scored_trajectories.sort(key=lambda t: t[1], reverse=True)
                all_trajectories = [trajectory for trajectory, _ in scored_trajectories[:self.trajectories_to_predict]]

            # 4. CREATE PREDICTIONS AND PUBLISH
            # create shapely linestring from lanelet centerlines and then use it to interpolate points in necessary distances
            for trajectory in all_trajectories:
                trajectory_linestring = shapely.LineString([(p.x, p.y, p.z) for lanelet in trajectory for p in lanelet.centerline])
                if self.use_offset_for_prediction:
                    cross_track_offset = -calculate_cross_track_error(centerline_linestring, object_centroid)
                    trajectory_linestring = centerline_linestring.offset_curve(cross_track_offset, join_style="mitre")
                else:
                    trajectory_linestring = centerline_linestring

                # get prediction origin right in front of the object
                object_front = get_point_using_heading_and_distance(obj.position, obj.heading, obj.dimensions.x / 2)
                object_front = shapely.Point(object_front.x, object_front.y, object_front.z)
                object_distance_from_trajectory_linestring_start = trajectory_linestring.project(object_front)

                path = Path()
                for distance, velocity in zip(distances, velocities):
                    wp = Waypoint()
                    p = trajectory_linestring.interpolate(object_distance_from_trajectory_linestring_start + distance)
                    wp.position.x = p.x
                    wp.position.y = p.y
                    if self.use_offset_for_prediction:
                       # for offset curve z is not available, therefore taken from the centerline
                       p = centerline_linestring.interpolate(object_distance_from_trajectory_linestring_start + distance)
                    wp.position.z = p.z
                    wp.speed = velocity
                    path.waypoints.append(wp)
                obj.candidate_trajectories.paths.append(path)

        # Publish predicted objects
        self.predicted_objects_pub.publish(msg)

    def evaluate_paths(self, path, object_indicator):
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