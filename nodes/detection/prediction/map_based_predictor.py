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

        for i, obj in enumerate(msg.objects):

            object_speed = get_vector_norm_3d(obj.velocity)
            if object_speed < self.prediction_min_speed:
                continue

            # 1. SEARCH BEST MATCHING LANELET FOR AN OBJECT
            object_location = BasicPoint2d(obj.position.x, obj.position.y)
            # find lanelets within distance to object_location - distance measured from lanelet borders. Inside lanelet area this distance would be 0
            lanelets_within_distance = findWithin2d(self.lanelet2_map.laneletLayer, object_location, self.distance_from_lanelet)

            selected_lanelet = None

            if len(lanelets_within_distance) > 0:
                min_heading_difference = math.inf
                object_centroid = shapely.Point(obj.position.x, obj.position.y)

                for d, lanelet in lanelets_within_distance:

                    # Skip crosswalks - don't want to snap predictions to crosswalks
                    if lanelet.attributes:
                        if lanelet.attributes["subtype"] == "crosswalk" or lanelet.attributes["subtype"] == "bus_lane":
                            continue

                    linestring = shapely.LineString([(p.x, p.y) for p in lanelet.centerline])
                    object_distance_from_lanelet_start = linestring.project(object_centroid)
                    object_location_on_lanelet = linestring.interpolate(object_distance_from_lanelet_start)

                    # Skip lanelet if angle difference between object heading and lanelet heading is over limit
                    
                    forward_point = linestring.interpolate(object_distance_from_lanelet_start + 0.1)
                    lanelet_heading = get_heading_between_two_points(object_location_on_lanelet, forward_point)
                    heading_difference_degrees = math.degrees(get_angle_between_two_headings(obj.heading, lanelet_heading))
                    if heading_difference_degrees < self.angle_threshold and heading_difference_degrees < min_heading_difference:
                        min_heading_difference = heading_difference_degrees
                        selected_lanelet = lanelet

            # 2. CREATE MAP BASED TRAJECTORIES FOR OBJECT
            if selected_lanelet is not None:

                # Predict future positions and velocities
                object_accel = get_vector_norm_3d(obj.acceleration)
                timesteps = np.arange(num_timesteps) * self.prediction_interval
                velocities = object_speed + object_accel * timesteps
                distances = (object_accel * timesteps**2) / 2 + object_speed * timesteps

                # get all possible paths (lanelet branching), from selected lanelet to max distance
                all_trajectories = self.graph.possiblePaths(selected_lanelet, object_distance_from_lanelet_start + distances[-1])

                if len(all_trajectories) == 0:
                    continue
                elif len(all_trajectories) == 1:
                    selected_trajectory = all_trajectories[0]
                elif len(all_trajectories) > 1:
                    # TODO: Correct object indicator should be taken from the object, currently using closest lanelet
                    object_indicator = selected_lanelet.attributes["turn_direction"] if "turn_direction" in selected_lanelet.attributes else "straight"

                    # Evaluate all possible paths (based on car indicator and path turn directions) and select the best one - highest score!
                    all_trajectories_with_turn_directions = []
                    for i, trajectory in enumerate(all_trajectories):
                        all_trajectories_with_turn_directions.append([lanelet.attributes["turn_direction"] if "turn_direction" in lanelet.attributes else "straight" for lanelet in trajectory])
                    all_trajectories_evaluated = self.evaluate_paths(all_trajectories_with_turn_directions, object_indicator)
                    selected_trajectory = all_trajectories[np.argmax(all_trajectories_evaluated)]

                # create shapely linestring from lanelet centerlines and then use it to interpolate points in necessary distances
                trajectory_linestring = shapely.LineString([(p.x, p.y, p.z) for lanelet in selected_trajectory for p in lanelet.centerline])
                if self.use_offset_for_prediction:
                    cross_track_offset = -calculate_cross_track_error(trajectory_linestring, object_centroid)
                    trajectory_linestring = trajectory_linestring.offset_curve(cross_track_offset, join_style="mitre")

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
                    wp.position.z = obj.position.z  # offset removes z coordinate
                    wp.speed = velocity
                    path.waypoints.append(wp)
                obj.candidate_trajectories.paths.append(path)

        # Publish predicted objects
        self.predicted_objects_pub.publish(msg)

    def evaluate_paths(self, paths, object_indicator):
        scores = []
        for path in paths:
            path_score = 0
            for i, turn in enumerate(path):
                # score the lanelet according to how well it matches the object indicator
                lanelet_score = CAR_INDICATOR_VS_TURN_DIRECTION_SCORING[object_indicator][turn]
                if i > 0:
                    # discount farther lanelets
                    lanelet_score /= i
                # path score is sum of lanelet scores
                path_score += lanelet_score
            scores.append(path_score)
        return scores

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('map_based_predictor', log_level=rospy.INFO)
    node = MapBasedPredictor()
    node.run()