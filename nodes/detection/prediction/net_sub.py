from abc import ABCMeta, abstractmethod

import numpy as np
import rospy
import threading

from autoware_msgs.msg import Lane, DetectedObjectArray, Waypoint
from geometry_msgs.msg import TwistStamped
from shapely import LineString, Point, LinearRing
from shapely.ops import split

from message_cache import MessageCache


class NetSubscriber(metaclass=ABCMeta):
    def __init__(self):
        self.lock = threading.Lock()
        self.self_traj_sub = rospy.Subscriber("/planning/local_path", Lane, self.self_traj_callback)
        self.velocity_sub = rospy.Subscriber("/localization/current_velocity", TwistStamped, self.velocity_callback)
        self.objects_sub = rospy.Subscriber("tracked_objects",
                                            DetectedObjectArray, self.detected_objects_sub_callback)
        self.objects_pub = rospy.Publisher('predicted_objects', DetectedObjectArray, queue_size=1,
                                           tcp_nodelay=True)
        # Caching structure
        self.self_new_traj = []
        self.self_traj_history = []
        # Dict of Message Cache class values
        self.cache = {}
        self.self_traj = []
        self.self_traj_exists = False
        self.velocity = 0.0
        self.active_keys = set()
        # Basic inference values
        self.inference_timer_duration = 0.5
        self.model = None
        self.predictions_amount = 1
        self.inference_timer = rospy.Timer(rospy.Duration(self.inference_timer_duration), self.inference_callback, reset=True)


    def self_traj_callback(self, lane):
        if len(lane.waypoints) > 3:
            self.self_traj_exists = True
        new_traj = []
        for waypoint in lane.waypoints:
            new_traj.append([waypoint.pose.pose.position.x, waypoint.pose.pose.position.y])
        self.self_new_traj = new_traj.copy()

    def velocity_callback(self, twist):
        self.velocity = (twist.twist.linear.x ** 2 + twist.twist.linear.y ** 2 + twist.twist.linear.z ** 2) ** 0.5

    def detected_objects_sub_callback(self, detectedobjectarray):
        # cache objects with filter, so we can refer to them at inference time
        active_keys = set()
        for i, detectedobject in enumerate(detectedobjectarray.objects):
            if detectedobject.label == 'pedestrian' or detectedobject.label == 'unknown':
                position = np.array([detectedobject.pose.position.x, detectedobject.pose.position.y])
                velocity = np.array([detectedobject.velocity.linear.x, detectedobject.velocity.linear.y])
                header = detectedobject.header
                _id = detectedobject.id
                active_keys.add(_id)
                with self.lock:
                    if not _id in self.cache:
                        self.cache[_id] = MessageCache(_id, position, velocity, header)
                        # self.cache[_id].backpropagate_trajectories()
                    else:
                        self.cache[_id].update_last_trajectory_velocity(position, velocity, header)
        with self.lock:
            self.active_keys = self.active_keys.union(active_keys)
        # Publish objects back retrieving candidate trajectories from history of inferences
        self.publish_predicted_objects(detectedobjectarray)

    # TODO: decide on how to integrate this with planning module or move out in differrent node
    def calculate_danger_values(self, inference_dataset, inference_result, temp_active_keys,
                                future_horizon=12, past_horizon=8):
        if self.self_traj_exists:
            trajectory_length = self.velocity * self.inference_timer_duration * future_horizon
            self.self_traj = self.self_new_traj.copy()
            # Calculating the danger value of agent intersecting with ego-vehicle
            inference_colors = (list(map(color_points, inference_result, [self.self_traj] * len(inference_result),
                                         [trajectory_length] * len(inference_result))))
            endpoint_colors = color_points(inference_dataset.traj_flat[:, past_horizon - 1],
                                           self.self_traj, trajectory_length)
            danger_values = np.zeros((len(endpoint_colors), len(inference_colors)))
            for i in range(len(endpoint_colors)):
                for j in range(self.predictions_amount):
                    danger_values[i, j] = calculate_danger_value(endpoint_colors[i], inference_colors[j][i])
            avg_danger_values = np.mean(danger_values, axis=1)

            for j, _id in enumerate(temp_active_keys):
                self.cache[_id].extend_prediction_history_danger_value(avg_danger_values[j])
            self.self_traj_history.append(self.self_traj[0])

    @abstractmethod
    def inference_callback(self, event):
        pass

    def publish_predicted_objects(self, detectedobjectsarray):
        # Construct candidate predictors from saved history of predictions
        output_msg_array = DetectedObjectArray(header=detectedobjectsarray.header)

        for msg in detectedobjectsarray.objects:
            with self.lock:
                generate_candidate_trajectories = msg.id in self.active_keys
            if generate_candidate_trajectories:
                for predictions in self.cache[msg.id].return_last_prediction():
                    lane = Lane(header=self.cache[msg.id].return_last_header())
                    # Start candidate trajectory from ego vehicle
                    wp = Waypoint()
                    wp.pose.pose.position = msg.pose.position
                    lane.waypoints.append(wp)
                    # Add prediction (endpoint only for now)
                    for j in [predictions]:
                        wp = Waypoint()
                        wp.pose.pose.position.x, wp.pose.pose.position.y = j
                        wp.pose.pose.position.z = msg.pose.position.z
                        lane.waypoints.append(wp)
                    msg.candidate_trajectories.lanes.append(lane)

            output_msg_array.objects.append(msg)

        # Publish objects with predicted candidate trajectories
        self.objects_pub.publish(output_msg_array)

    def move_endpoints(self):
        # Moves end-point of cached trajectory every inference
        with self.lock:
            for _id in self.active_keys:
                self.cache[_id].move_endpoints()
            self.active_keys = set()

    def run(self):
        rospy.spin()


def color_points(points, trajectory, trajectory_length):
    # drivable_area = np.swapaxes(np.array(Image.open(rospy.get_param('~data_path_prediction') + 'semantic_rasters/drivable_area_mask_updated.png')), 0, 1)

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
        ## TODO: Implement better agnostic driving area detection
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
