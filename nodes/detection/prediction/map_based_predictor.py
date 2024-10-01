#!/usr/bin/env python3

import rospy
import math
import numpy as np
import lanelet2
from lanelet2.core import BasicPoint2d
from lanelet2.geometry import findWithin2d
from shapely.geometry import LineString, Point as ShapelyPoint
from autoware_msgs.msg import DetectedObjectArray, Lane, Waypoint
from helpers.geometry import get_heading_from_vector, get_vector_norm_3d, get_heading_between_two_points, get_distance_between_two_points_2d, create_vector_from_heading_and_scalar, get_angle_between_two_headings
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
        self.distance_from_lanelet = rospy.get_param('~distance_from_lanelet')
        self.distance_from_centerline = rospy.get_param('~distance_from_centerline')
        self.angle_threshold = rospy.get_param('~angle_threshold')

        lanelet2_map_name = rospy.get_param("/planning/lanelet2_global_planner/lanelet2_map_name")
        coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")

        self.lanelet2_map = load_lanelet2_map(lanelet2_map_name, coordinate_transformer, use_custom_origin, utm_origin_lat, utm_origin_lon)
        traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany, lanelet2.traffic_rules.Participants.VehicleTaxi)
        self.graph = lanelet2.routing.RoutingGraph(self.lanelet2_map, traffic_rules)

        # Publishers
        self.predicted_objects_pub = rospy.Publisher('predicted_objects', DetectedObjectArray, queue_size=1, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('tracked_objects', DetectedObjectArray, self.tracked_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)

    def tracked_objects_callback(self, msg):

        num_timesteps = int(self.prediction_horizon // self.prediction_interval)

        for i, obj in enumerate(msg.objects):

            # 1. SEARCH BEST MATCHING LANELET FOR AN OBJECT
            object_location = BasicPoint2d(obj.pose.position.x, obj.pose.position.y)
            # find lanelets within distance to object_location - distance measured from lanelet borders. Inside lanelet area this distance would be 0
            lanelets_witihn_distance = findWithin2d(self.lanelet2_map.laneletLayer, object_location, self.distance_from_lanelet)

            heading_differences = []
            potential_lanelets = []
            for d, lanelet in lanelets_witihn_distance:

                # Skip crosswalks - don't want to snap predictions to crosswalks
                if lanelet.attributes:
                    if lanelet.attributes["subtype"] == "crosswalk":
                        continue

                linestring = LineString([(p.x, p.y) for p in lanelet.centerline])
                object_distance_from_lanelet_start = linestring.project(ShapelyPoint(object_location.x, object_location.y))
                trajectory_start_point = linestring.interpolate(object_distance_from_lanelet_start)

                # Skip lanelet if distance from centerline over limit
                object_distance_from_centerline = get_distance_between_two_points_2d(trajectory_start_point, object_location)
                if object_distance_from_centerline > self.distance_from_centerline:
                    continue

                # Skip lanelet if angle difference between object heading and lanelet heading is over limit
                object_heading = get_heading_from_vector(obj.velocity.linear)
                forward_point = linestring.interpolate(object_distance_from_lanelet_start + 0.1)
                lanelet_heading = get_heading_between_two_points(trajectory_start_point, forward_point)
                heading_difference_degrees = math.degrees(get_angle_between_two_headings(object_heading, lanelet_heading))
                if heading_difference_degrees > self.angle_threshold:
                    continue

                heading_differences.append(heading_difference_degrees)
                potential_lanelets.append(lanelet)

            selected_lanelet = None
            if len(heading_differences) > 0:
                min_index = np.argmin(heading_differences)
                selected_lanelet = potential_lanelets[min_index]
                # TODO: Correct object indicator should be taken from the object, currently using closest lanelet
                object_indicator = selected_lanelet.attributes["turn_direction"]

            # 2. CREATE MAP BASED TRAJECTORIES FOR OBJECT
            if selected_lanelet is not None:

                object_speed = get_vector_norm_3d(obj.velocity.linear)
                object_accel = get_vector_norm_3d(obj.acceleration.linear)

                # Predict future positions and velocities
                timesteps = np.arange(num_timesteps) * self.prediction_interval
                velocities = object_speed + object_accel * timesteps
                distances = np.cumsum(np.insert(velocities[:-1] * self.prediction_interval, 0, 0))

                # get all possible paths (lanelet branching), from selected lanelet to max distance
                all_trajectories = self.graph.possiblePaths(selected_lanelet, distances[-1])

                if len(all_trajectories) > 0:
                    selected_trajectory = all_trajectories[0]
                    if len(all_trajectories) > 1:
                        # Evaluate all possible paths (based on car indicator and path turn directions) and select the best one - highest score!
                        all_trajectories_with_turn_directions = {}
                        for i, trajectory in enumerate(all_trajectories):
                            all_trajectories_with_turn_directions[i] = [lanelet.attributes["turn_direction"] for lanelet in trajectory]
                        all_trajectories_evaluated = self.evaluate_paths(all_trajectories_with_turn_directions, object_indicator)
                        selected_trajectory = all_trajectories[np.argmax(all_trajectories_evaluated)]

                    # create shapely linestring from lanelet centerlines and then use it to interpolate points in necessary distances
                    trajectory_linestring = LineString([(p.x, p.y, p.z) for lanelet in selected_trajectory for p in lanelet.centerline])
                    object_distance_from_trajectory_linestring_start = trajectory_linestring.project(ShapelyPoint(object_location.x, object_location.y))

                    lane = Lane()
                    for i, d in enumerate(distances):
                        wp = Waypoint()
                        p = trajectory_linestring.interpolate(object_distance_from_trajectory_linestring_start + d)
                        wp.pose.pose.position.x = p.x
                        wp.pose.pose.position.y = p.y
                        wp.pose.pose.position.z = p.z
                        # TODO Recalculating velocity vector based on lanelet heading at the object location.
                        # Wrong when lanelet changes direction (turns), but good enough for now?
                        speed_x, speed_y = create_vector_from_heading_and_scalar(lanelet_heading, velocities[i])
                        wp.twist.twist.linear.x = speed_x
                        wp.twist.twist.linear.y = speed_y
                        lane.waypoints.append(wp)
                    obj.candidate_trajectories.lanes.append(lane)

        # Publish predicted objects
        self.predicted_objects_pub.publish(msg)

    def evaluate_paths(self, paths, object_indicator):
        # Evaluate each path using scoring constants with decreasing weight
        # return list of evaluation scores in the order of given paths

        evaluations = [
            sum(CAR_INDICATOR_VS_TURN_DIRECTION_SCORING[object_indicator][turn] * (1 if i == 0 else 1 / i)
                for i, turn in enumerate(path))
            for path in paths.values()
        ]
        return evaluations

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('map_based_predictor', log_level=rospy.INFO)
    node = MapBasedPredictor()
    node.run()