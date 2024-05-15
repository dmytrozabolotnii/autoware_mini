import math
from autoware_msgs.msg import WaypointState
from geometry_msgs.msg import Point
from helpers.geometry import get_distance_between_two_points_2d, get_angle_three_points_2d, get_heading_between_two_points, get_orientation_from_heading, get_closest_point_on_line

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
    
def get_blinker_state_with_lookahead(distance_to_blinker_interpolator, ego_distance_from_path_start, blinker_lookahead_distance):
    """
    Get blinker state. 
    :param distance_to_blinker_interpolator: interpolator containing autoware_msg/WaypointState
    :param ego_distance_from_path_start: distance from path start (m)
    :param blinker_lookahead_d: distance to look ahead point for blinkers (m)
    :return: LampCmd (l, r) included in VehicleCmd
    """

    current_pose_blinker_state = int(distance_to_blinker_interpolator(ego_distance_from_path_start))

    if current_pose_blinker_state != WaypointState.STR_STRAIGHT:
        return get_blinker_state(current_pose_blinker_state)
    else:
        lookahead_blinker_state = int(distance_to_blinker_interpolator(blinker_lookahead_distance))
        return get_blinker_state(lookahead_blinker_state)


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

def get_point_and_orientation_on_path_within_distance(waypoints, front_wp_idx, start_point, distance):
    """
    Get point on path within distance from ego pose
    :param waypoints: waypoints
    :param front_wp_idx: wp index from where to start calculate the distance
    :param start_point: starting point for distance calculation
    :param distance: distance where to find the point on the path
    :return: Point, Quaternion
    """

    point = Point()
    last_idx = len(waypoints) - 1

    i = front_wp_idx
    d = get_distance_between_two_points_2d(start_point, waypoints[i].pose.pose.position)
    while d < distance:
        i += 1
        d += get_distance_between_two_points_2d(waypoints[i-1].pose.pose.position, waypoints[i].pose.pose.position)
        if i == last_idx:
            break

    # Find point orientation and distance difference and correct along path backwards
    end_orientation =  get_heading_between_two_points(waypoints[i].pose.pose.position, waypoints[i - 1].pose.pose.position)
    dx = (distance - d) * math.cos(end_orientation)
    dy = (distance - d) * math.sin(end_orientation)
    point.x = waypoints[i].pose.pose.position.x - dx
    point.y = waypoints[i].pose.pose.position.y - dy
    point.z = waypoints[i].pose.pose.position.z

    orientation = get_orientation_from_heading(end_orientation)

    return point, orientation
