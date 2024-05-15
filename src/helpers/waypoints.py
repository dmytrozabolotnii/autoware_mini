import math
from autoware_msgs.msg import WaypointState
from geometry_msgs.msg import Point
from helpers.geometry import get_distance_between_two_points_2d, get_angle_three_points_2d, get_heading_between_two_points, get_orientation_from_heading, get_closest_point_on_line
from shapely.geometry import LineString

def get_blinker_state(steering_state):
    """
    Get blinker state  from WaypointState/steering_state
    :param steering_state: steering state
    :return: LampCmd (l, r) included in VehicleCmd
    """

    if steering_state == WaypointState.STR_LEFT:
        return 1, 0
    elif steering_state == WaypointState.STR_RIGHT:
        return 0, 1
    elif steering_state == WaypointState.STR_STRAIGHT:
        return 0, 0
    else:
        return 0, 0
    
def get_blinker_state_with_lookahead(waypoints, waypoint_interval, wp_idx, velocity, lookahead_time, lookahead_distance):
    """
    Get blinker state from current idx and look ahead using time.
    Blinker state at current location has priority (turn needs to be properly finished)
    If no blinker in current position then look ahead up to determined time.
    :param waypoints: list of waypoints
    :param waypoint_interval: waypoint interval in meters
    :param wp_idx: current waypoint index
    :param velocity: current velocity (m/s)
    :param lookahead_time: time to look ahead (s)
    :param lookahead_distance: min distance to look ahead (m)
    :return: LampCmd (l, r) included in VehicleCmd
    """

    # check blinker state in current position
    if waypoints[wp_idx].wpstate.steering_state != WaypointState.STR_STRAIGHT:
        return get_blinker_state(waypoints[wp_idx].wpstate.steering_state)
    else:
        # calc how many waypoints to look ahead
        wp_lookahead = int(max(velocity * lookahead_time, lookahead_distance) / waypoint_interval)

        # return first WaypointState that is not straight within the lookahead distance
        for i in range(wp_idx, min(wp_idx + wp_lookahead, len(waypoints))):
            if waypoints[i].wpstate.steering_state != WaypointState.STR_STRAIGHT:
                return get_blinker_state(waypoints[i].wpstate.steering_state)

        # return straight if no turning waypoint found
        return get_blinker_state(WaypointState.STR_STRAIGHT)

def get_two_nearest_waypoint_idx(waypoint_tree, x, y):
    """
    Find 2 cloest waypoint index values from the waypoint_tree
    :param waypoint_tree:
    :param x
    :param y
    """

    idx = waypoint_tree.kneighbors([(x, y)], 2, return_distance=False)

    # sort to get them in ascending order - follow along path
    idx[0].sort()
    return idx[0][0], idx[0][1]

def interpolate_velocity_between_waypoints(point, backward_wp, forward_wp):
    """
    Interpolate velocity between two waypoints.
    :param point: Point - location where the velocity will be interpolated using backward and forward waypoints.
    :param bacward_wp: Waypoint
    :param forward_wp: Waypoint
    :return: velocity
    """

    # distance to backward waypoint
    distance_to_backward_wp = get_distance_between_two_points_2d(point, backward_wp.pose.pose.position)
    if distance_to_backward_wp < 0.01:
        return backward_wp.twist.twist.linear.x

    # distance to forward waypoint
    distance_to_forward_wp = get_distance_between_two_points_2d(point, forward_wp.pose.pose.position)
    if distance_to_forward_wp < 0.01:
        return forward_wp.twist.twist.linear.x

    backward_wp_vel = backward_wp.twist.twist.linear.x * distance_to_forward_wp / (distance_to_backward_wp + distance_to_forward_wp)
    forward_wp_vel = forward_wp.twist.twist.linear.x * distance_to_backward_wp / (distance_to_backward_wp + distance_to_forward_wp)

    return backward_wp_vel + forward_wp_vel

def get_closest_point_on_path(waypoints, closest_idx, origin_point):
    """
    Project origin_point onto the path. First decide weather to use backward or forward waypointand then call
    get_closest_point_on_line function to get the closest point on path.
    :param waypoints: list of waypoints
    :param closest_idx: index of the closest waypoint
    :param origin_point: Point - origin point
    :return: Point - closest point on path
    """

    #initialize angles
    backward_angle = 0
    forward_angle = 0

    if closest_idx > 0:
        backward_angle = abs(get_angle_three_points_2d(waypoints[closest_idx - 1].pose.pose.position, origin_point, waypoints[closest_idx].pose.pose.position))
    if closest_idx < len(waypoints) - 1:
        forward_angle = abs(get_angle_three_points_2d(waypoints[closest_idx].pose.pose.position, origin_point, waypoints[closest_idx + 1].pose.pose.position))

    # if backward angle is bigger then project point to the backward section
    if backward_angle > forward_angle:
        closest_idx -= 1

    origin_position = Point(x = origin_point.x, y = origin_point.y, z = waypoints[closest_idx].pose.pose.position.z)
    point = get_closest_point_on_line(origin_position, waypoints[closest_idx].pose.pose.position, waypoints[closest_idx + 1].pose.pose.position)
    return point

def get_point_and_orientation_on_path_within_distance(waypoints, distance):
    """
    Get point and perpendicular orientation within distance along the path
    :param waypoints: waypoints
    :param distance: distance where to find the point on the path
    :return: Point, Quaternion
    """

    waypoints_xyz = [(waypoint.pose.pose.position.x, waypoint.pose.pose.position.y, waypoint.pose.pose.position.z) for waypoint in waypoints]
    linestring = LineString(waypoints_xyz)

    # Find the point on the path
    point_location = linestring.interpolate(distance)
    point_before = linestring.interpolate(distance - 0.01)

    heading = get_heading_between_two_points(point_before, point_location)
    orientation = get_orientation_from_heading(heading)

    point = Point(x = point_location.x, y = point_location.y, z = point_location.z)

    return point, orientation