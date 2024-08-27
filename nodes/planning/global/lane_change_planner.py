#!/usr/bin/env python3

import rospy
import numpy as np
import math

from geometry_msgs.msg import Point
from autoware_msgs.msg import Lane, Waypoint

from helpers.geometry import get_heading_between_two_points, get_orientation_from_heading, \
    get_heading_from_orientation, get_point_using_heading_and_distance, \
    get_distance_between_two_points_2d, get_angle_between_three_points, calculate_points_on_bezier_curve


class LaneChangePlanner:

    def __init__(self):

        # Parameters
        self.waypoint_interval = rospy.get_param("waypoint_interval")
        self.lane_change_base_length = rospy.get_param("lane_change_base_length")
        self.lane_change_perlane_length = rospy.get_param("lane_change_perlane_length")

        # Publishers
        self.lane_change_path_pub = rospy.Publisher('lane_change_path', Lane, queue_size=10, latch=True, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('global_path', Lane, self.global_path_callback, queue_size=None, tcp_nodelay=True)


    def global_path_callback(self, msg):
        lane = Lane()
        lane.header = msg.header

        waypoints = self.create_lane_change_paths(msg.waypoints)
        if waypoints is None:
            rospy.logerr("%s - path contained an impossible lane change!", rospy.get_name())
        else:
            lane.waypoints = waypoints
        self.lane_change_path_pub.publish(lane)

    def create_lane_change_paths(self, waypoints):
        idx = 0
        while idx < len(waypoints):
            # Check for lane change
            if waypoints[idx].wpstate.lanechange_state > 0:
                start_idx = idx
                end_idx = None

                # Check that the lane change waypoint is not the last waypoint of the path
                if start_idx + 1 == len(waypoints):
                    return None
                
                start_point = waypoints[start_idx].pose.pose.position
                other_point = waypoints[start_idx + 1].pose.pose.position
                steering_state = waypoints[start_idx].wpstate.steering_state

                # Skip all lane change waypoints
                while idx < len(waypoints) and waypoints[idx].wpstate.lanechange_state > 0:
                    lanechange_state = waypoints[idx].wpstate.lanechange_state
                    idx += 1

                # Skip all non lane change waypoints until enough distance to perform lane change
                while idx < len(waypoints):
                    current_point = waypoints[idx].pose.pose.position
                    # Get the diagonal distance of the lane change
                    d = get_distance_between_two_points_2d(start_point, current_point)

                    # Calculate the lane change angle
                    a = get_angle_between_three_points(other_point, start_point, current_point)

                    # Calculate lane change length
                    given_lanechange_length = self.lane_change_base_length + lanechange_state * self.lane_change_perlane_length

                    # Use the angle to check that the lane change doesn't happen behind us
                    # Multiply the diagonal distance with cos(a) to get the parallel distance of the lane change 
                    if abs(a) < np.pi/2 and d * np.cos(a) >= given_lanechange_length:
                        end_idx = idx
                        break

                    idx += 1

                # End of path before lane change is complete
                if end_idx is None:
                    return None

                # Replace section of waypoints with spline
                spline = self.calculate_lane_change_spline(waypoints[start_idx], waypoints[end_idx], 
                                                           given_lanechange_length, steering_state)

                waypoints = waypoints[:start_idx] + spline + waypoints[end_idx+1:]

                # Advance the index beyond the lane change
                idx = start_idx + len(spline)

            else:
                idx += 1

        return waypoints

    def calculate_lane_change_spline(self, start_waypoint, end_waypoint, lanechange_length, steering_state):

        ##################################################################
        # Calculate Bezier curve control points p0, p1, p2, p3
        ##################################################################

        start_heading = get_heading_from_orientation(start_waypoint.pose.pose.orientation)
        control_point1 = get_point_using_heading_and_distance(start_waypoint.pose.pose.position, start_heading, lanechange_length / 3)

        end_heading = get_heading_from_orientation(end_waypoint.pose.pose.orientation)
        control_point2 = get_point_using_heading_and_distance(end_waypoint.pose.pose.position, end_heading + math.pi, lanechange_length / 3)

        bezier_points = calculate_points_on_bezier_curve(
            start_waypoint.pose.pose.position,
            control_point1, control_point2, 
            end_waypoint.pose.pose.position,
            int(lanechange_length // self.waypoint_interval)
        )

        ##################################################################
        # Create lane change waypoints
        ##################################################################

        # Calculate the distance of each Bezier point from the benning of the spline
        lane_change_wp_distances = np.cumsum(np.sqrt(np.sum(np.diff(bezier_points, axis=0)**2, axis=1)))
        # Add 0 to the beginning of the array
        lane_change_wp_distances = np.insert(lane_change_wp_distances, 0, 0)
        # Use the first and last distance as datapoints
        distance_datapoints  = np.array([0, lane_change_wp_distances[-1]])
        
        # Speed interpolation
        speed_datapoints = np.array([start_waypoint.twist.twist.linear.x, end_waypoint.twist.twist.linear.x])
        speed = np.interp(lane_change_wp_distances, distance_datapoints, speed_datapoints)

        # Left lane width interpolation
        lw_datapoints = np.array([start_waypoint.dtlane.lw, end_waypoint.dtlane.lw])
        lw = np.interp(lane_change_wp_distances, distance_datapoints, lw_datapoints)

        # Right lane width interpolation
        lw_datapoints = np.array([start_waypoint.dtlane.rw, end_waypoint.dtlane.rw])
        rw = np.interp(lane_change_wp_distances, distance_datapoints, lw_datapoints)

        # z-coordinate interpolation
        z_datapoints = np.array([start_waypoint.pose.pose.position.z, end_waypoint.pose.pose.position.z])
        z_coords = np.interp(lane_change_wp_distances, distance_datapoints, z_datapoints)

        waypoints = []
        for i in range(len(bezier_points)):
            if i == len(bezier_points) - 1:
                heading = end_heading
            else:
                point = Point(x=bezier_points[i, 0], y=bezier_points[i, 1])
                next_point = Point(x=bezier_points[i+1, 0], y=bezier_points[i+1, 1])
                heading = get_heading_between_two_points(point, next_point)

            waypoint = Waypoint()
            waypoint.pose.pose.position.x = bezier_points[i, 0]
            waypoint.pose.pose.position.y = bezier_points[i, 1]
            waypoint.pose.pose.position.z = z_coords[i]
            waypoint.pose.pose.orientation = get_orientation_from_heading(heading)
            waypoint.twist.twist.linear.x = speed[i]
            waypoint.wpstate.steering_state = steering_state
            waypoint.wpstate.lanechange_state = 0
            waypoint.dtlane.lw = lw[i]
            waypoint.dtlane.rw = rw[i]
            
            waypoints.append(waypoint)

        return waypoints

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('lane_change_planner')
    node = LaneChangePlanner()
    node.run()