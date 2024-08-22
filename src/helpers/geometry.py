import math
import numpy as np
from scipy.interpolate import BPoly
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Point, Quaternion

def get_heading_from_vector(vector):
    """
    Get heading from vector
    :param vector: vector
    :return: heading in radians
    """

    return math.atan2(vector.y, vector.x)

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

def project_vector_to_heading(heading_angle, vector):
    """
    Project vector to heading
    :param heading_angle: heading angle in radians
    :param vector: vector
    :return: projected vector
    """

    return vector.x * math.cos(heading_angle) + vector.y * math.sin(heading_angle)

def create_vector_from_heading_and_scalar(heading, scalar):
    """
    Create vector from heading and scalar
    :param heading: heading in radians
    :param scalar: scalar
    :return: vector
    """

    return (scalar * math.cos(heading), scalar * math.sin(heading))

def get_angle_between_three_points(first_point, middle_point, third_point):
    """
    Calculates angle between three points
    :param first_point: Point
    :param middle_point: Middle point
    :param third_point: Point
    :return: angle in radians
    """

    BA = np.array([first_point.x - middle_point.x, first_point.y - middle_point.y])
    BC = np.array([third_point.x - middle_point.x, third_point.y - middle_point.y])
    
    dot_product = np.dot(BA, BC)
    
    magnitude_ba = np.linalg.norm(BA)
    magnitude_bc = np.linalg.norm(BC)
    
    cos_theta = dot_product / (magnitude_ba * magnitude_bc)
    
    # Clip the cosine value within range [-1, 1] to avoid numerical issues
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    angle = np.arccos(cos_theta)
    
    return angle

def calculate_points_on_bezier_curve(control_points, n):
    """
    Generates n number of equaly spaced points on a Bezier curve
    :param control_points: Bezier curve conrol points as an numpy array
    :param n: number of points to calculate
    :return: points on Bezier curve
    """

    p0, p1, p2, p3 = control_points
    samples = np.linspace(0, 1, n)
    bezier_points = [(1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t) * t**2 * p2 + t**3 * p3 for t in samples]
    return bezier_points
