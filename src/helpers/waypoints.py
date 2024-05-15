import math
from autoware_msgs.msg import WaypointState
from geometry_msgs.msg import Point
from helpers.geometry import get_distance_between_two_points_2d, get_heading_between_two_points, get_orientation_from_heading

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