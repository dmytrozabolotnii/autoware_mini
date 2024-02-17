import numpy as np
import rospy
import threading

from autoware_msgs.msg import Lane, DetectedObjectArray, DetectedObject, Waypoint
from geometry_msgs.msg import TwistStamped
from visualization_msgs.msg import MarkerArray, Marker

from common_utils import color_points, calculate_danger_value


class NetSubscriber:
    def __init__(self):
        rospy.loginfo(self.__class__.__name__ + " - Initializing")
        self.lock = threading.Lock()
        self.self_traj_sub = rospy.Subscriber("/planning/local_path", Lane, self.self_traj_callback)
        self.velocity_sub = rospy.Subscriber("/localization/current_velocity", TwistStamped, self.velocity_callback)
        self.objects_sub = rospy.Subscriber("tracked_objects",
                                            DetectedObjectArray, self.detected_objects_sub_callback)
        self.marker_pub = rospy.Publisher('predicted_behaviours_net', MarkerArray, queue_size=1, tcp_nodelay=True)
        self.objects_pub = rospy.Publisher('predicted_behaviours_net_objects', DetectedObjectArray, queue_size=1,
                                           tcp_nodelay=True)
        # Caching structure
        self.self_new_traj = []
        self.self_traj_history = []
        self.all_raw_trajectories = {}
        self.all_raw_velocities = {}
        self.all_predictions_history = {}
        self.all_predictions_history_danger_value = {}
        self.last_messages = {}
        self.raw_trajectories = {}
        self.raw_velocities = {}
        self.self_traj = []
        self.self_traj_exists = False
        self.velocity = 0.0
        self.all_endpoints = {}
        self.current_endpoints = {}
        self.active_keys = set()
        # Basic inference values
        self.inference_timer_duration = 0.5
        self.model = None
        self.predictions_amount = 1


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
            if detectedobject.label == 'pedestrian':
                position = np.array([detectedobject.pose.position.x, detectedobject.pose.position.y])
                velocity = np.array([detectedobject.velocity.linear.x, detectedobject.velocity.linear.y])
                _id = detectedobject.id
                active_keys.add(_id)

                if not _id in self.all_raw_trajectories:
                    self.all_raw_trajectories[_id] = [position]
                    self.all_raw_velocities[_id] = [velocity]
                    self.all_endpoints[_id] = 0
                    self.all_predictions_history[_id] = [[]]
                    self.all_predictions_history_danger_value[_id] = []
                    self.last_messages[_id] = detectedobject
                else:
                    self.all_raw_trajectories[_id][len(self.all_raw_trajectories[_id]) - 1] = position
                    self.all_raw_velocities[_id][len(self.all_raw_velocities[_id]) - 1] = velocity
                    self.last_messages[_id] = detectedobject
        with self.lock:
            self.active_keys = active_keys

    def calculate_danger_values_and_publish(self, inference_dataset, inference_result, temp_active_keys,
                                            future_horizon=12, past_horizon=8):
        if self.self_traj_exists:
            trajectory_length = self.velocity * self.inference_timer_duration * future_horizon
            self.self_traj = self.self_new_traj.copy()
            # Calculating the danger value of agent intersecting with ego-vehicle
            # TODO: decide on how to integrate this with planning module or move out in differrent node
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
                self.all_predictions_history_danger_value[_id].append(avg_danger_values[j])
            self.self_traj_history.append(self.self_traj[0])

            self.publish_markers(inference_dataset.traj_flat[:, past_horizon - 1],
                                 inference_result, inference_colors, endpoint_colors, avg_danger_values,
                                 self.predictions_amount)

    # TODO: phase out in favour of existing markers visualization
    def publish_markers(self, endpoints, inference_result, inference_colors, endpoint_colors, avg_danger_values, predictions_amount):
        marker_array = MarkerArray()
        marker_array.markers = []
        # Endpoint markers and danger values
        for i, endpoint in enumerate(endpoints):

            # # Endpoint markers
            # marker = Marker()
            # marker.id = 2 * i
            # marker.header.frame_id = 'map'
            # marker.header.stamp = rospy.Time.now()
            # marker.type = 2
            #
            # marker.scale.x = 3.0
            # marker.scale.y = 3.0
            # marker.scale.z = 3.0
            # marker.color.a = 3.0
            # if endpoint_colors[i][0] == 'g':
            #     marker.color.g = 1.0
            # elif endpoint_colors[i][0] == 'y':
            #     marker.color.r = 1.0
            #     marker.color.g = 1.0
            # elif endpoint_colors[i][0] == 'r':
            #     marker.color.r = 1.0
            # else:
            #     marker.color.b = 1.0
            # marker.text = 'dng: ' + str(avg_danger_values[i])
            #
            # marker.pose.position.x = endpoint[0]
            # marker.pose.position.y = endpoint[1]
            #
            # marker.pose.orientation.x = 0.0
            # marker.pose.orientation.y = 0.0
            # marker.pose.orientation.z = 0.0
            # marker.pose.orientation.w = 1.0
            #
            # marker_array.markers.append(marker)

            # Danger values
            marker_text = Marker()
            marker_text.id = 2 * i + 1
            marker_text.header.frame_id = 'map'
            marker_text.header.stamp = rospy.Time.now()
            marker_text.type = 9

            marker_text.scale.x = 3.0
            marker_text.scale.y = 3.0
            marker_text.scale.z = 3.0
            marker_text.color.a = 3.0
            if endpoint_colors[i][0] == 'g':
                marker_text.color.g = 1.0
            elif endpoint_colors[i][0] == 'y':
                marker_text.color.r = 1.0
                marker_text.color.g = 1.0
            elif endpoint_colors[i][0] == 'r':
                marker_text.color.r = 1.0
            else:
                marker_text.color.b = 1.0

            marker_text.text = 'dng: ' + str(round(avg_danger_values[i], 2))

            marker_text.pose.position.x = endpoint[0] - 5
            marker_text.pose.position.y = endpoint[1]

            marker_text.pose.orientation.x = 0.0
            marker_text.pose.orientation.y = 0.0
            marker_text.pose.orientation.z = 0.0
            marker_text.pose.orientation.w = 1.0

            marker_array.markers.append(marker_text)

        # Prediction markers
        for j in range(predictions_amount):
            for i in range(len(endpoints)):
                marker = Marker()
                marker.id = 2 * len(endpoints) + j * len(endpoints) + i + 1
                marker.header.frame_id = 'map'
                marker.header.stamp = rospy.Time.now()
                marker.type = 2

                marker.scale.x = 1.0
                marker.scale.y = 1.0
                marker.scale.z = 1.0
                marker.color.a = 1.0
                if inference_colors[j][i][0] == 'g':
                    marker.color.g = 1.0
                elif inference_colors[j][i][0] == 'y':
                    marker.color.r = 1.0
                    marker.color.g = 1.0
                elif inference_colors[j][i][0] == 'r':
                    marker.color.r = 1.0
                else:
                    marker.color.b = 1.0
                marker.text = str(inference_colors[j][i])

                marker.pose.position.x = inference_result[j][i][0]
                marker.pose.position.y = inference_result[j][i][1]

                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0

                marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

    def publish_predicted_objects(self):
        # Construct candidate predictors from saved history of predictions
        msg_array = DetectedObjectArray()
        msg_array.header.frame_id = 'map'
        msg_array.header.stamp = rospy.Time.now()
        with self.lock:
            for _id in self.active_keys:
                msg = self.last_messages[_id]
                for predictions in self.all_predictions_history[_id][len(self.all_predictions_history[_id]) - 2]:
                    lane = Lane()
                    for j in [predictions]:
                        wp = Waypoint()
                        wp.pose.pose.position.x, wp.pose.pose.position.y = j
                        lane.waypoints.append(wp)
                    msg.candidate_trajectories.lanes.append(lane)
                msg_array.objects.append(msg)
        # Publish objects with predicted candidate trajectories
        self.objects_pub.publish(msg_array)

    def move_endpoints(self):
        # Moves end-point of cached trajectory every inference
        with self.lock:
            for _id in self.active_keys:
                self.all_endpoints[_id] += 1
                self.all_raw_trajectories[_id].append(self.all_raw_trajectories[_id][len(self.all_raw_trajectories[_id]) - 1])
                self.all_raw_velocities[_id].append(self.all_raw_velocities[_id][len(self.all_raw_velocities[_id]) - 1])

    def run(self):
        rospy.spin()

