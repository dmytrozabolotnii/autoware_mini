#!/usr/bin/env python3

# Adapted naive_predictor for pedestrian prediction experiments with RVO constraints

import rospy
import numpy as np
import time

from autoware_mini.msg import DetectedObjectArray, Path, Waypoint

from net_sub import NetSubscriber
from shapely import GeometryCollection, Polygon, LineString, Point
from shapely.affinity import affine_transform, rotate
from shapely.ops import orient, polygonize_full


def minkowski_sum(polygon_a, polygon_b):
    # to numpyland
    pol_a = np.array(polygon_a.exterior.coords[:])
    pol_b = np.array(polygon_b.exterior.coords[:])
    # find lowest y vertice for algorithm and reorder
    min_a, min_b = np.argmin(pol_a[:, 1]), np.argmin(pol_b[:, 1])
    pol_a = np.vstack((pol_a[min_a:-1], pol_a[:min_a]))
    pol_b = np.vstack((pol_b[min_b:-1], pol_b[:min_b]))
    # Minkowski sum
    msum = []
    i, j = 0, 0
    l1, l2 = len(pol_a), len(pol_b)
    # iterate through all the vertices
    while i < l1 or j < l2:
        msum.append(pol_a[i % l1] + pol_b[j % l2])
        cross = np.cross(pol_a[(i + 1) % l1] - pol_a[i % l1], pol_b[(j + 1) % l2] - pol_b[j % l2])
        # using right-hand rule choose the vector with the lower polar angle and iterate this polygon's vertex
        if cross >= 0:
            i += 1
        else:
            j += 1

    return Polygon(msum)

class RVOPredictor(NetSubscriber):
    def __init__(self):
        super().__init__()
        # Parameters
        self.prediction_horizon = rospy.get_param('prediction_horizon')
        self.prediction_interval = rospy.get_param('step_length')
        self.constant_velocity_mode = bool(rospy.get_param('~constant_velocity'))
        self.is_cluster_detector = bool(rospy.get_param('~is_cluster_detector'))
        self.responsibility_factor = 0.5
        self.line_length = 10000
        # self.range_limit = 10 if self.is_cluster_detector else 100
        self.range_limit = 10
        # Publishers


        # Subscribers

    def inference_callback(self, event):
        if len(self.active_keys):
            with self.lock:
                temp_active_keys = set(self.active_keys)
                # Convert tracked objects to numpy array
                tracked_objects_array = np.empty((len(temp_active_keys)), dtype=[
                    ('centroid', np.float32, (2,)),
                    ('velocity', np.float32, (2,)),
                    ('acceleration', np.float32, (2,)),
                ])
                tracked_objects_convex_hull_array = []
                tracked_objects_array_ids = np.zeros((len(temp_active_keys)))
                for i, key in enumerate(temp_active_keys):
                    tracked_objects_array_ids[i] = key
                    tracked_objects_array[i]['centroid'] = (self.cache[key].raw_trajectories[-1][0], self.cache[key].raw_trajectories[-1][1])
                    tracked_objects_array[i]['velocity'] = (self.cache[key].raw_velocities[-1][0], self.cache[key].raw_velocities[-1][1])
                    if self.constant_velocity_mode:
                        tracked_objects_array[i]['acceleration'] = 0
                    else:
                        tracked_objects_array[i]['acceleration'] = (
                        self.cache[key].raw_accelerations[-1][0], self.cache[key].raw_accelerations[-1][1])
                    if self.cache[key].convex_hull is not None:
                        polygon = Polygon([(p.x, p.y) for p in self.cache[key].convex_hull.points])
                        tracked_objects_convex_hull_array.append(polygon)
                tracked_objects_convex_hull_array = GeometryCollection(tracked_objects_convex_hull_array)
                temp_headers = [self.cache[key].return_last_header() for key in temp_active_keys]


            # Predict future positions and velocities
            num_timesteps = self.prediction_horizon + 1
            predicted_objects_array = np.empty((num_timesteps, len(temp_active_keys)), dtype=[
                ('centroid', np.float32, (2,)),
                ('velocity', np.float32, (2,)),
            ])
            predicted_objects_array[0] = tracked_objects_array[['centroid', 'velocity']]
            rvo_objects_array = np.empty((num_timesteps, len(temp_active_keys)), dtype=[
                ('centroid', np.float32, (2,)),
                ('velocity', np.float32, (2,)),
            ])
            rvo_objects_array[0] = tracked_objects_array[['centroid', 'velocity']]
            # RVO
            mink_time = 0
            total_time = time.time()
            for i in range(0, len(tracked_objects_array)):
                # Change to coordinate system from ego object (and reverse ego polygon for minkowski)
                polygon_a = orient(affine_transform(tracked_objects_convex_hull_array.geoms[i], matrix=[
                    -1, 0, 0, -1, tracked_objects_array[i]['centroid'][0], tracked_objects_array[i]['centroid'][1]]),
                                   sign=1)

                zero_speed_flag = False
                deviation_vectors = []
                deviation_vectors_directions = []
                for j in range(0, len(tracked_objects_array)):
                    if i != j and np.linalg.norm(tracked_objects_array[i]['centroid'] - tracked_objects_array[j]['centroid']) < self.range_limit:
                        polygon_b = orient(affine_transform(tracked_objects_convex_hull_array.geoms[j], matrix=[
                            1, 0, 0, 1, -1 * tracked_objects_array[i]['centroid'][0], -1 * tracked_objects_array[i]['centroid'][1]]), sign=1)
                        # Calculate minkowski sum
                        t0 = time.time()
                        polygon_min = minkowski_sum(polygon_a, polygon_b)
                        t1 = time.time()
                        mink_time += t1 - t0
                        # print(i, j, polygon_b)
                        # print('end')
                        # Reduce minkowski sum by time of planning
                        prediction_horizon_time = self.prediction_horizon * self.prediction_interval
                        polygon_min = affine_transform(polygon_min, [1.0 / prediction_horizon_time, 0, 0, 1.0 / prediction_horizon_time, 0, 0])
                        # Find if relative speed vector intersects reduced minkowski sum
                        rel_speed_vector = tracked_objects_array[i]['velocity'] - tracked_objects_array[j]['velocity']
                        rel_speed_vector_line = LineString([(0, 0), rel_speed_vector])
                        if rel_speed_vector_line.intersects(polygon_min):
                            # print('RVO intersection ', tracked_objects_array_ids[i], tracked_objects_array_ids[j])
                            if polygon_min.contains(Point(0, 0)):
                                zero_speed_flag = True
                                # print('RVO intersection contains point of origin', tracked_objects_array[i]['centroid'], tracked_objects_array[j]['centroid'])
                            # Find the min and max signed angle between relative speed vector and obstacle
                            angles = (np.arctan2(np.array(polygon_min.exterior.coords[:])[:, 1], np.array(polygon_min.exterior.coords[:])[:, 0])
                                      - np.arctan2(rel_speed_vector[1], rel_speed_vector[0]))
                            angles[angles > np.pi] = angles[angles > np.pi] - 2 * np.pi
                            angles[angles < -1 * np.pi] = angles[angles < -1 * np.pi] + 2 * np.pi
                            # Calculate deviation vector
                            if angles[np.argmax(angles)] >= abs(angles[np.argmin(angles)]):
                                min_angle_to_deviate = angles[np.argmin(angles)]
                                deviation_vector = affine_transform(rotate(rel_speed_vector_line, min_angle_to_deviate - (np.pi / 2),
                                                          Point((0, 0)), use_radians=True),
                                                                    [self.responsibility_factor * np.tan(min_angle_to_deviate), 0, 0,
                                                                     self.responsibility_factor * np.tan(min_angle_to_deviate), tracked_objects_array[i]['velocity'][0], tracked_objects_array[i]['velocity'][1]])
                                deviation_vectors_direction = -1
                            else:
                                min_angle_to_deviate = angles[np.argmax(angles)]
                                deviation_vector = affine_transform(rotate(rel_speed_vector_line, min_angle_to_deviate + (np.pi / 2),
                                                          Point((0, 0)), use_radians=True),
                                                                    [self.responsibility_factor * np.tan(min_angle_to_deviate), 0, 0,
                                                                     self.responsibility_factor * np.tan(min_angle_to_deviate), tracked_objects_array[i]['velocity'][0], tracked_objects_array[i]['velocity'][1]])
                                deviation_vectors_direction = 1

                            # print(angles)
                            # print(min_angle_to_deviate)
                            # print(deviation_vector)

                            deviation_vectors.append(deviation_vector)
                            deviation_vectors_directions.append(deviation_vectors_direction)
                # Construct free from obstacle zone from deviation vectors
                if zero_speed_flag:
                    print('Collision RVO case')
                    # predicted_objects_array[0, i]['velocity'] = [0, 0]

                elif len(deviation_vectors) > 1:
                    half_planes = []
                    # for deviation_vector in deviation_vectors:
                    #     left = deviation_vector.parallel_offset(self.line_length / 2, 'left')
                    #     right = deviation_vector.parallel_offset(self.line_length / 2, 'right')
                    #     c = left.coords[1]
                    #     d = right.coords[0]  # note the different orientation for right offset
                    #     # print(LineString([c, d]))
                    #     half_planes.append(LineString([c, d]))
                    # result, cuts, dangles, invalids = polygonize_full(half_planes)
                    print('Hard RVO case')
                    # print(result.geoms[:])
                    # print(cuts.geoms[:])
                    # print(dangles.geoms[:])

                elif len(deviation_vectors) == 1:
                    print('Simple RVO case')
                    # predicted_objects_array[0, i]['velocity'] = deviation_vectors[0].coords[1]
                    rvo_objects_array[0, i]['velocity'] = deviation_vectors[0].coords[1]

            # print('Total time', time.time() - total_time, 'Minkowski sum time', mink_time)

            for i in range(1, num_timesteps):
                predicted_objects_array[i]['centroid'] = predicted_objects_array[i - 1]['centroid'] + \
                                                         predicted_objects_array[i - 1][
                                                             'velocity'] * self.prediction_interval
                predicted_objects_array[i]['velocity'] = predicted_objects_array[i - 1]['velocity'] + \
                                                         tracked_objects_array[
                                                             'acceleration'] * self.prediction_interval
            for i in range(1, num_timesteps):
                rvo_objects_array[i]['centroid'] = rvo_objects_array[i - 1]['centroid'] + \
                                                         rvo_objects_array[i - 1][
                                                             'velocity'] * self.prediction_interval
                rvo_objects_array[i]['velocity'] = rvo_objects_array[i - 1]['velocity'] + \
                                                         tracked_objects_array[
                                                             'acceleration'] * self.prediction_interval

            with self.lock:
                # Create candidate trajectories
                for i, _id in enumerate(temp_active_keys):
                    self.cache[_id].extend_prediction_history([predicted_objects_array[:, i]['centroid'], rvo_objects_array[:, i]['centroid']])
                    self.cache[_id].extend_prediction_header_history(temp_headers[i])
            self.move_endpoints()


    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('rvo_predictor', log_level=rospy.INFO)
    node = RVOPredictor()
    node.run()