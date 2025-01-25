#!/usr/bin/env python3
from audioop import cross

# Adapted naive_predictor for pedestrian prediction experiments with RVO constraints

import rospy
import numpy as np
import time

from autoware_mini.msg import DetectedObjectArray, Path, Waypoint
from math import atan2

from shapely.geometry import LinearRing

from net_sub import NetSubscriber
from shapely import GeometryCollection, Polygon, LineString, Point, prepare
from shapely.affinity import affine_transform, rotate
from shapely.geometry.polygon import orient

import lanelet2
from lanelet2.core import BasicPoint2d
from lanelet2.geometry import findWithin2d
from helpers.lanelet2 import load_lanelet2_map

pedestrian_normal_walking_speed = 1.3888888
fov_acceptable_circle_radius = 0.5
MAX_STEPS_LONG_BRUTEFORCE = 10
MAX_STEPS_LAT_BRUTEFORCE = 10


def oriented_angle(a, b):
    dot = a[0] * b[0] + a[1] * b[1]
    det = a[0] * b[1] - a[1] * b[0]

    return atan2(det, dot)

def cross_prod(a, b):
    return a[0] * b[1] - a[1] * b[0]

def distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

def simple_affine_transform(pol, mult, offset):
    pol = np.array(pol.exterior.coords) * mult + offset

    return Polygon(pol)

def simple_rotate(vector, angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array(((c, -s), (s, c)))

    return np.dot(R, vector)

def minkowski_sum(polygon_a, polygon_b, mult = 1.0):
    # to numpyland
    pol_a = np.array(polygon_a.exterior.coords[:])
    pol_b = np.array(polygon_b.exterior.coords[:])
    # find lowest y vertice for algorithm and reorder
    min_a, min_b = np.argmin(pol_a[:, 1]), np.argmin(pol_b[:, 1])
    pol_a = np.vstack((pol_a[min_a:-1], pol_a[:min_a]))
    pol_b = np.vstack((pol_b[min_b:-1], pol_b[:min_b]))
    # Minkowski sum
    i, j = 0, 0
    l1, l2 = len(pol_a), len(pol_b)
    msum = np.zeros((l1 + l2, 2))

    # iterate through all the vertices
    while i < l1 or j < l2:
        msum[i + j] = (pol_a[i % l1] + pol_b[j % l2]) * mult
        cross = cross_prod(pol_a[(i + 1) % l1] - pol_a[i % l1], pol_b[(j + 1) % l2] - pol_b[j % l2])
        # using right-hand rule choose the vector with the lower polar angle and iterate this polygon's vertex
        if cross >= 0:
            i += 1
        else:
            j += 1

    return Polygon(msum)

def single_check_solution(solution, check_id, pure_deviation_vectors, representative_vectors):
    # Switch to coordinate system of other deviation vector
    temp_vector = solution - pure_deviation_vectors[check_id]
    # Find directed angle between representing vector and solution, this angle should be positive
    angle = oriented_angle(representative_vectors[check_id], temp_vector)
    # print(check_id, solution, temp_vector, representative_vectors[check_id], angle)
    return angle >= 0


def check_solution(solution, do_not_check_id, pure_deviation_vectors, representative_vectors,
                   fov_deviation_vectors, fov_representative_vectors, crosswalks, lanelets):
    solution_passes = True
    for i in range(len(pure_deviation_vectors)):
        if i != do_not_check_id:
            solution_passes = single_check_solution(solution, i, pure_deviation_vectors, representative_vectors)
            if not solution_passes:
                return solution_passes

    for fov_deviation_vector, fov_representative_vector in zip(fov_deviation_vectors, fov_representative_vectors):
        center_distance = distance(solution - fov_deviation_vector, [0, 0])
        if center_distance > fov_acceptable_circle_radius:
            solution_passes = single_check_solution(solution, 0,
                                                    [fov_deviation_vector], [fov_representative_vector])
            if not solution_passes:
                return solution_passes

    point_solution = Point(solution[0], solution[1])
    for crosswalk in crosswalks:
        if crosswalk.contains(point_solution):
            return solution_passes

    for lanelet in lanelets:
        if lanelet.contains(point_solution):
            solution_passes = False
            return  solution_passes

    return solution_passes

def resolve_hard_rvo(velocity, deviation_vectors, fov_constraint=False, crosswalks=None, lanelets=None):
    # Move to velocity vector center coordinate system:
    if crosswalks is None:
        crosswalks = []
    else:
        crosswalks = [simple_affine_transform(crosswalk, 1, -1 * velocity) for crosswalk in crosswalks]
    if lanelets is None:
        lanelets = []
    else:
        lanelets = [simple_affine_transform(lanelet, 1, -1 * velocity) for lanelet in lanelets]

    if len(deviation_vectors) > 0:
        no_deviation_vectors = False
        pure_deviation_vectors = np.zeros((len(deviation_vectors), 2))
        representative_vectors = np.zeros_like(pure_deviation_vectors)
        pure_deviation_vectors_size = np.zeros((len(pure_deviation_vectors)))
        for i in range(len(deviation_vectors)):
            coords = np.asarray(deviation_vectors[i].coords)
            pure_deviation_vectors[i] = coords[1] - coords[0]
            representative_vectors[i] = [pure_deviation_vectors[i][1], -1 * pure_deviation_vectors[i][0]]
            pure_deviation_vectors_size[i] = distance(pure_deviation_vectors[i], [0, 0])
    else:
        # Hack to create fake deviation vector that is extension of velocity
        no_deviation_vectors = True
        pure_deviation_vectors = np.array([velocity / 100])
        representative_vectors = np.zeros_like(pure_deviation_vectors)
        pure_deviation_vectors_size = np.array([distance(pure_deviation_vectors[0], [0, 0])])

    # Find the max length deviation vector which is minimal valid solution
    max_deviation_vector_index = np.argmax(pure_deviation_vectors_size)
    max_deviation_vector = pure_deviation_vectors[max_deviation_vector_index]
    max_deviation_vector_length = np.max(pure_deviation_vectors_size)

    # Default scenario for no constraints at all or only fov constraint
    if no_deviation_vectors and len(crosswalks) == 0 and len(lanelets) == 0:
        return velocity

    # Add fov constraints if necessary
    if fov_constraint:
        # pure_deviation_vectors = np.vstack((pure_deviation_vectors, [-1 * velocity]))
        # representative_vectors = np.vstack((representative_vectors, [simple_rotate(velocity, -1 * np.pi / 2)]))

        fov_deviation_vectors = np.array([-1 * velocity] * 2)
        fov_representative_vectors = np.array([-1 * simple_rotate(velocity, np.pi / 3), simple_rotate(velocity, -1 * np.pi / 3)])
        # representative_vectors = np.vstack((representative_vectors, [simple_rotate(velocity, np.pi / 3), -1 * simple_rotate(velocity, -1 * np.pi / 3)]))
    else:
        fov_deviation_vectors = []
        fov_representative_vectors = []

    # Check if it is valid solution for all constraints
    if check_solution(max_deviation_vector, max_deviation_vector_index, pure_deviation_vectors, representative_vectors,
                      fov_deviation_vectors, fov_representative_vectors, crosswalks, lanelets):
        return velocity + max_deviation_vector
    else:
        # Start bruteforce =(
        longitudal_range = max(distance(velocity, [0, 0]) - max_deviation_vector_length,
                               pedestrian_normal_walking_speed - max_deviation_vector_length)
        # Longitudal bruteforce
        for i in range(MAX_STEPS_LONG_BRUTEFORCE):
            # Find new solution vector
            new_solution = max_deviation_vector / max_deviation_vector_length * (max_deviation_vector_length + longitudal_range * (i + 1) / MAX_STEPS_LONG_BRUTEFORCE)
            if check_solution(new_solution, max_deviation_vector_index, pure_deviation_vectors, representative_vectors,
                              fov_deviation_vectors, fov_representative_vectors, crosswalks, lanelets):
                return velocity + new_solution
            # Latitudal bruteforce
            for j in range(MAX_STEPS_LAT_BRUTEFORCE // 2 + no_deviation_vectors * MAX_STEPS_LAT_BRUTEFORCE // 2):
                angle_to_rotate = (np.pi / 2) * (j + 1) / (MAX_STEPS_LAT_BRUTEFORCE // 2)
                # Counter clockwise solution
                new_solution_cc = simple_rotate(new_solution, angle_to_rotate)
                # Single check if we are out of bounds of original half-plane if we are checking half-plane
                if not no_deviation_vectors and not single_check_solution(new_solution_cc, max_deviation_vector_index, pure_deviation_vectors, representative_vectors):
                    break
                if check_solution(new_solution_cc, max_deviation_vector_index, pure_deviation_vectors, representative_vectors,
                                  fov_deviation_vectors, fov_representative_vectors, crosswalks, lanelets):
                    return velocity + new_solution_cc
                # Clockwise solution
                new_solution_c = simple_rotate(new_solution, -1 * angle_to_rotate)
                if check_solution(new_solution_c, max_deviation_vector_index, pure_deviation_vectors, representative_vectors,
                                  fov_deviation_vectors, fov_representative_vectors, crosswalks, lanelets):
                    return velocity + new_solution_c

    return None


class RVOPredictor(NetSubscriber):
    def __init__(self):
        super().__init__()
        # Parameters
        self.prediction_horizon = rospy.get_param('prediction_horizon')
        self.prediction_interval = rospy.get_param('step_length')
        self.fov_constraint = rospy.get_param('~fov_constraint', False)
        self.map_constraints = rospy.get_param('~map_constraints', False)
        self.cars_constraints = rospy.get_param('~cars_constraints', False)
        self.cars_constraints_minimum_speed = pedestrian_normal_walking_speed
        self.rvo_only = rospy.get_param('~rvo_only', False)

        self.prediction_horizon_time = self.prediction_horizon * self.prediction_interval

        self.constant_velocity_mode = bool(rospy.get_param('~constant_velocity'))
        self.is_cluster_detector = bool(rospy.get_param('~is_cluster_detector'))
        self.responsibility_factor = 0.5
        self.range_limit = 10 if self.is_cluster_detector else 100

        lanelet2_map_name = rospy.get_param("/planning/lanelet2_global_planner/lanelet2_map_name")

        self.lanelet2_map = load_lanelet2_map(lanelet2_map_name)

        # Debug
        self.average_time = 0
        self.average_time_counter = 0
        # Publishers


        # Subscribers

    def inference_callback(self, event):
        if len(self.active_keys):
            with self.lock:
                temp_active_keys = set(self.active_keys)
                temp_active_keys_cars = set(self.active_keys_cars)
                # Convert tracked objects to numpy array
                tracked_objects_array = np.empty((len(temp_active_keys) + self.cars_constraints * len(temp_active_keys_cars)), dtype=[
                    ('centroid', np.float32, (2,)),
                    ('velocity', np.float32, (2,)),
                    ('acceleration', np.float32, (2,)),
                ])
                tracked_objects_convex_hull_array = []
                tracked_objects_array_ids = np.zeros((len(tracked_objects_array)))
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
                        tracked_objects_convex_hull_array.append(orient(polygon))
                ped_count = len(temp_active_keys)
                if self.cars_constraints:
                    for i, key in enumerate(temp_active_keys_cars):
                        tracked_objects_array_ids[ped_count + i] = key
                        tracked_objects_array[ped_count + i]['centroid'] = (
                        self.cache_cars[key].raw_trajectories[-1][0], self.cache_cars[key].raw_trajectories[-1][1])
                        tracked_objects_array[ped_count + i]['velocity'] = (
                        self.cache_cars[key].raw_velocities[-1][0], self.cache_cars[key].raw_velocities[-1][1])
                        if self.constant_velocity_mode:
                            tracked_objects_array[ped_count + i]['acceleration'] = 0
                        else:
                            tracked_objects_array[ped_count + i]['acceleration'] = (
                                self.cache_cars[key].raw_accelerations[-1][0], self.cache_cars[key].raw_accelerations[-1][1])
                        if self.cache_cars[key].convex_hull is not None:
                            polygon = Polygon([(p.x, p.y) for p in self.cache_cars[key].convex_hull.points])
                            tracked_objects_convex_hull_array.append(orient(polygon))

                tracked_objects_convex_hull_array = GeometryCollection(tracked_objects_convex_hull_array)
                temp_headers = [self.cache[key].return_last_header() for key in temp_active_keys]

            # Predict future positions and velocities
            num_timesteps = self.prediction_horizon + 1
            predicted_objects_array = np.empty((num_timesteps, ped_count), dtype=[
                ('centroid', np.float32, (2,)),
                ('velocity', np.float32, (2,)),
            ])
            predicted_objects_array[0] = tracked_objects_array[:ped_count][['centroid', 'velocity']]
            rvo_objects_array = np.empty((num_timesteps, ped_count), dtype=[
                ('centroid', np.float32, (2,)),
                ('velocity', np.float32, (2,)),
            ])
            rvo_objects_array[0] = tracked_objects_array[:ped_count][['centroid', 'velocity']]
            # RVO
            mink_time = 0
            mink_count = 0
            total_time = time.time()
            for i in range(0, ped_count):
                # Change to coordinate system from ego object (and reverse ego polygon for minkowski)
                # polygon_a = affine_transform(tracked_objects_convex_hull_array.geoms[i], matrix=[
                #     -1, 0, 0, -1, tracked_objects_array[i]['centroid'][0], tracked_objects_array[i]['centroid'][1]])
                polygon_a = orient(simple_affine_transform(tracked_objects_convex_hull_array.geoms[i], -1, tracked_objects_array[i]['centroid']))

                collision_flag = False
                deviation_vectors = []
                for j in range(0, len(tracked_objects_array)):
                    if i != j and distance(tracked_objects_array[i]['centroid'], tracked_objects_array[j]['centroid']) < self.range_limit \
                        and (j<ped_count or distance(tracked_objects_array[j]['velocity'], [0, 0]) >= self.cars_constraints_minimum_speed):
                        polygon_b = simple_affine_transform(tracked_objects_convex_hull_array.geoms[j], 1,
                                                            -1 * tracked_objects_array[i]['centroid'])
                        # Calculate minkowski sum
                        t0 = time.time()
                        polygon_min = minkowski_sum(polygon_a, polygon_b, 1 / self.prediction_horizon_time)
                        t1 = time.time()
                        mink_time += t1 - t0
                        mink_count += 1
                        # Find if relative speed vector intersects reduced minkowski sum
                        rel_speed_vector = tracked_objects_array[i]['velocity'] - tracked_objects_array[j]['velocity']
                        rel_speed_vector_line = LineString([(0, 0), rel_speed_vector])
                        if rel_speed_vector_line.intersects(polygon_min):
                            # print('RVO intersection ', tracked_objects_array_ids[i], tracked_objects_array_ids[j])
                            if polygon_min.contains(Point(0, 0)):
                                collision_flag = True
                                if polygon_min.contains(Point(rel_speed_vector[0], rel_speed_vector[1])):
                                    polygon_min_ext = LinearRing(polygon_min.exterior.coords)
                                    d = polygon_min_ext.project(Point(rel_speed_vector[0], rel_speed_vector[1]))
                                    closest_point = polygon_min_ext.interpolate(d)
                                    change_vector = tracked_objects_array[i]['velocity'] + (np.array(closest_point.coords) - rel_speed_vector) * self.responsibility_factor
                                    deviation_vector = LineString([(tracked_objects_array[i]['velocity'][0], tracked_objects_array[i]['velocity'][1]),
                                                                   (change_vector[0, 0], change_vector[0, 1])])
                                    deviation_vectors_direction = 0

                                    deviation_vectors.append(deviation_vector)
                            else:
                                # Find the min and max signed angle between relative speed vector and obstacle
                                angles = [oriented_angle(rel_speed_vector, polygon_min.exterior.coords[i]) for i in range(len(polygon_min.exterior.coords))]
                                # Calculate deviation vector
                                if angles[np.argmax(angles)] >= abs(angles[np.argmin(angles)]):
                                    min_angle_to_deviate = angles[np.argmin(angles)]
                                    deviation_vector = affine_transform(rotate(rel_speed_vector_line, min_angle_to_deviate - (np.pi / 2),
                                                              Point((0, 0)), use_radians=True),
                                                                        [self.responsibility_factor * abs(np.sin(min_angle_to_deviate)), 0, 0,
                                                                         self.responsibility_factor * abs(np.sin(min_angle_to_deviate)), tracked_objects_array[i]['velocity'][0], tracked_objects_array[i]['velocity'][1]])
                                    deviation_vectors_direction = -1

                                else:
                                    min_angle_to_deviate = angles[np.argmax(angles)]
                                    deviation_vector = affine_transform(rotate(rel_speed_vector_line, min_angle_to_deviate + (np.pi / 2),
                                                              Point((0, 0)), use_radians=True),
                                                                        [self.responsibility_factor * abs(np.sin(min_angle_to_deviate)), 0, 0,
                                                                         self.responsibility_factor * abs(np.sin(min_angle_to_deviate)), tracked_objects_array[i]['velocity'][0], tracked_objects_array[i]['velocity'][1]])
                                    deviation_vectors_direction = 1
                                deviation_vectors.append(deviation_vector)
                crosswalks = None
                lanelets = None
                if self.map_constraints:
                    crosswalks = []
                    lanelets = []
                    # search matching lanelets to a centroid
                    x, y = float(tracked_objects_array[i]['centroid'][0]), float(tracked_objects_array[i]['centroid'][1])
                    object_location = BasicPoint2d(x, y)
                    # find lanelets within distance to object_location - distance measured from lanelet borders
                    lanelets_within_distance = findWithin2d(self.lanelet2_map.laneletLayer, object_location,
                                                            max(2 * self.prediction_horizon_time * distance(tracked_objects_array[i]['velocity'], [0, 0]), 2 * self.prediction_horizon_time * pedestrian_normal_walking_speed))
                    for d, lanelet in lanelets_within_distance:
                        # print(lanelet.attributes["subtype"])
                        # print(dir(lanelet))
                        # print(lanelet.polygon2d())
                        # print(dir(lanelet.polygon2d()))
                        if lanelet.attributes and lanelet.attributes["subtype"] == 'crosswalk':
                            crosswalks.append(Polygon([((p.x - x) / self.prediction_horizon_time, (p.y - y) / self.prediction_horizon_time) for p in lanelet.polygon2d()]))
                            # print('Crosswalks', crosswalks)
                        else:
                            lanelets.append(Polygon([((p.x - x)  / self.prediction_horizon_time, (p.y - y)  / self.prediction_horizon_time) for p in lanelet.polygon2d()]))
                            # print('Lanelets', lanelets)

                # Construct free from obstacle zone from deviation vectors

                if len(deviation_vectors) >= 0:
                    result = resolve_hard_rvo(rvo_objects_array[0, i]['velocity'], deviation_vectors, fov_constraint=self.fov_constraint, crosswalks=crosswalks, lanelets=lanelets)
                    if result is not None:
                        rvo_objects_array[0, i]['velocity'] = result
                    else:
                        # print('Hard RVO resolver failed, adjusting speed to zero, id:', tracked_objects_array_ids[i])
                        # print(len(deviation_vectors), deviation_vectors)
                        # print('Crosswalks:', crosswalks)
                        # print('Lanelets:', lanelets)
                        # rvo_objects_array[0, i]['velocity'] = 0
                        print('Hard RVO resolver failed, adjusting speed to base velocity, id:', tracked_objects_array_ids[i])

                if collision_flag and len(deviation_vectors) == 0:
                    print('Collision case without change, id:', tracked_objects_array_ids[i])
                elif collision_flag and len(deviation_vectors) > 0:
                    print('Collision flag with change, id:', tracked_objects_array_ids[i])

            callback_time = time.time() - total_time
            self.average_time = (self.average_time * self.average_time_counter + callback_time) / (self.average_time_counter + 1)
            self.average_time_counter += 1
            # print('Total time', callback_time, 'Average time', self.average_time, 'Minkowski sum time', mink_time, 'Minkowski sum counts', mink_count)

            for i in range(1, num_timesteps):
                predicted_objects_array[i]['centroid'] = predicted_objects_array[i - 1]['centroid'] + \
                                                         predicted_objects_array[i - 1][
                                                             'velocity'] * self.prediction_interval
                predicted_objects_array[i]['velocity'] = predicted_objects_array[i - 1]['velocity'] + \
                                                         tracked_objects_array[:ped_count][
                                                             'acceleration'] * self.prediction_interval
            for i in range(1, num_timesteps):
                rvo_objects_array[i]['centroid'] = rvo_objects_array[i - 1]['centroid'] + \
                                                         rvo_objects_array[i - 1][
                                                             'velocity'] * self.prediction_interval
                rvo_objects_array[i]['velocity'] = rvo_objects_array[i - 1]['velocity'] + \
                                                         tracked_objects_array[:ped_count][
                                                             'acceleration'] * self.prediction_interval

            with self.lock:
                # Create candidate trajectories
                for i, _id in enumerate(temp_active_keys):
                    if self.rvo_only:
                        self.cache[_id].extend_prediction_history([rvo_objects_array[:, i]['centroid']])
                    else:
                        self.cache[_id].extend_prediction_history([predicted_objects_array[:, i]['centroid'], rvo_objects_array[:, i]['centroid']])
                    self.cache[_id].extend_prediction_header_history(temp_headers[i])
            self.move_endpoints()


    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('rvo_predictor', log_level=rospy.INFO)
    node = RVOPredictor()
    node.run()