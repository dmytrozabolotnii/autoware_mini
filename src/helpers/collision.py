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
    def __init__(self):

        self._array = np.array([], dtype=DTYPE)


    def add_point(self, x, y, z, vx, vy, vz, distance_to_stop, category):
        self._array = np.append(self._array, np.array([(x, y, z, vx, vy, vz, distance_to_stop, category)], dtype=DTYPE))

    def add_intersection_points(self, intersection_points, z, vx, vy, vz, distance_to_stop, category):
        for point in intersection_points:
            x = point.x
            y = point.y
            self.add_point(x, y, z, vx, vy, vz, distance_to_stop, category)

    def get_message(self):
        return msgify(PointCloud2, self._array)