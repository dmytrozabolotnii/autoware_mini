import math
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Point, Quaternion


def get_heading_from_orientation(orientation):
    """
    Get heading angle from orientation.
    :param orientation: Quaternion
    :return: heading in radians
    """

    quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
    _, _, heading = euler_from_quaternion(quaternion)

    return heading

def get_orientation_from_heading(heading):
    """
    Get orientation from heading (-pi...pi)
    :param heading: heading in radians
    :return: orientation
    :rtype: Quaternion
    """

    x, y, z, w = quaternion_from_euler(0, 0, heading)
    return Quaternion(x, y, z, w)

def get_heading_between_two_points(back_p, forward_p):
    """
    Get heading between two points
    :param back_p: Point
    :param forward_p: Point
    :return: heading in radians
    """

    return math.atan2(forward_p.y - back_p.y, forward_p.x - back_p.x)

def get_cross_track_error(ego_pos, pos1, pos2):
    """
    Get cross track error from ego pose and two poses
    # calc distance from track
    # https://robotics.stackexchange.com/questions/22989/what-is-wrong-with-my-stanley-controller-for-car-steering-control

    :param ego_pose: Point
    :param pose1: Point
    :param pose2: Point
    :return: cross track error
    """

    numerator = (pos2.x - pos1.x) * (pos1.y - ego_pos.y) - (pos1.x - ego_pos.x) * (pos2.y - pos1.y)
    denominator = math.sqrt((pos2.x - pos1.x) ** 2 + (pos2.y - pos1.y) ** 2)

    return numerator / denominator

def get_point_using_heading_and_distance(start_point, heading, distance):
    """
    Get point from given point and extrapolating it using heading and distance
    :param start_point: Point
    :param heading: heading in radians
    :param distance: distance in meters
    :return: Point
    """

    x = start_point.x + distance * math.cos(heading)
    y = start_point.y + distance * math.sin(heading)

    return Point(x=x, y=y, z=start_point.z)

def normalize_heading_error(err):
    """
    Get heading error relative to path
    Previously subtracted track and current heading need to be normilized, since the original
    heading angles are within range [-pi, pi]
    :param err: heading error
    :return err: steering difference in radians
    """

    if err > math.pi:
        err -= 2 * math.pi
    elif err < -math.pi:
        err += 2 * math.pi

    return err

def get_distance_between_two_points_2d(p1, p2):
    """
    Get distance between two points
    :param point1: Point
    :param point2: Point
    :return: distance
    """

    return math.hypot(p2.x - p1.x, p2.y - p1.y)

def get_vector_norm_3d(vec):
    """
    Get norm of 3d vector
    :param vec: vector
    :return: norm
    """

    return math.sqrt(vec.x ** 2 + vec.y ** 2 + vec.z ** 2)
