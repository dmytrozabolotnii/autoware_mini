import os.path

# import rospy
import numpy as np
from shapely import Point, LineString, LinearRing
from shapely.ops import split
from PIL import Image

# drivable_area = np.swapaxes(np.array(Image.open(rospy.get_param('~data_path_prediction') + 'semantic_rasters/drivable_area_mask_updated.png')), 0, 1)


def color_points(points, trajectory, trajectory_length):
    local_car_track = LineString(trajectory)
    cutoff_point = local_car_track.interpolate(trajectory_length)
    local_car_track = split(local_car_track, cutoff_point).geoms[0]
    array = []
    for point in points:
        local_shapely_point = Point(point)
        distance = local_shapely_point.distance(local_car_track)

        def side_decision(point, track):
            left_buffer = track.buffer(distance + 1, single_sided=True)
            if left_buffer.contains(point):
                return '1'
            else:
                right_buffer = track.buffer(-1 * distance - 1, single_sided=True)
                if right_buffer.contains(point):
                    return '2'
                else:
                    return '0'
        def side_decision_simplified(point, track):
            if LinearRing(point.coords[:] + track.coords[:]).is_ccw:
                return '1'
            else:
                return '2'
        if distance < 3:
            array.append('r')
        # elif drivable_area[(int(point[0]) - 8926) * 5, (int(point[1]) - 10289) * -5]:
        #     array.append('y' + side_decision(local_shapely_point, local_car_track))
        else:
            # array.append('g0')
            array.append('g' + side_decision(local_shapely_point, local_car_track))
    return array


def calculate_danger_value(color1, color2):
    dict_local = {
        'g1': 0,
        'g2': 4,
        'y1': 1,
        'y2': 3,
        'r': 2,
        'g0': 5,
        'y0': 5,
    }
    table_local = np.array(
    [
    [0, 0.05, 1, 0.75, 0.6, 0],
    [0.1, 0.15, 1, 0.8, 0.65, 0.15],
    [0.55, 0.75, 1, 0.75, 0.55, 1],
    [0.65, 0.8, 1, 0.15, 0.1, 0.15],
    [0.6, 0.75, 1, 0.05, 0, 0],
    [0, 0.15, 1, 0.15, 0, 0]
    ]
    )

    return table_local[dict_local[color1], dict_local[color2]]
