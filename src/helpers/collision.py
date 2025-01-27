import enum
import numpy as np
from ros_numpy import msgify
from sensor_msgs.msg import PointCloud2


DTYPE = np.dtype([
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('vx', np.float32),
    ('vy', np.float32),
    ('vz', np.float32),
    ('distance_to_stop', np.float32),
    ('category', np.int32)
])

class CollisionPoints:

    NO_OBSTACLES = 0
    GOAL_POINT = 1
    TRAFFIC_LIGHT_STOPLINE = 2
    STOPPED_OBSTACLE_ON_PATH = 3
    MOVING_OBSTACLE_ON_PATH = 4
    COLLIDING_TRAJECTORY = 5
    MERGING_TRAJECTORY = 6
    OBJECT_ON_CROSSWALK = 7
    TRAJECTORY_ON_CROSSWALK = 8
    YIELDING_TRAJECTORY = 9
    STOP_LINE_FORCED_STOP = 10

    def __init__(self):

        self._array = np.array([], dtype=DTYPE)


    def add_point(self, x, y, z, vx, vy, vz, distance_to_stop, category):
        self._array = np.append(self._array, np.array([(x, y, z, vx, vy, vz, distance_to_stop, category)], dtype=DTYPE))

    def add_points(self, points, vx, vy, vz, distance_to_stop, category):
        for point in points:
            self.add_point(point.x, point.y, point.z, vx, vy, vz, distance_to_stop, category)

    def add_intersection_points(self, intersection_points, z, vx, vy, vz, distance_to_stop, category):
        for x, y in intersection_points:
            self.add_point(x, y, z, vx, vy, vz, distance_to_stop, category)

    def create_message(self):
        return msgify(PointCloud2, self._array)


def calculate_time_to_destination(velocity, acceleration, distances):
    """
    Calculation of the time to reach certain distances given a constant velocity and acceleration.
    :param velocity: float, the velocity of the object
    :param acceleration: float, the acceleration of the object
    :param distances: numpy array of floats, the distances to calculate the time to reach
    :return: numpy array of floats, the time it takes to reach the distances
    """
    
    # Handle zero acceleration and zero velocity cases outside of the main calculation
    if velocity == 0 and acceleration == 0:
        # Object is not moving and has no acceleration
        return np.full_like(distances, np.nan, dtype=float)
    elif acceleration == 0:
        # Object is moving with constant velocity
        time_to_destination = np.where(distances >= 0, distances / velocity, np.nan)
        return time_to_destination

    # Vectorized calculation for cases with acceleration
    discriminant = velocity**2 + 2 * acceleration * distances  # Calculate the discriminant
    valid_discriminant = discriminant >= 0  # Mask invalid discriminants (negative values)

    # Initialize result with NaN (invalid cases)
    time_to_destination = np.full_like(distances, np.nan, dtype=float)

    # Apply formula only where discriminant is valid
    time_to_destination[valid_discriminant] = (
        -velocity / acceleration +
        np.sqrt(discriminant[valid_discriminant]) / np.abs(acceleration)
    )

    return time_to_destination