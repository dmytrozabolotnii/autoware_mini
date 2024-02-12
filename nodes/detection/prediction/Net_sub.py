import threading

import numpy as np
import rospy
from autoware_msgs.msg import Lane, DetectedObjectArray
from geometry_msgs.msg import TwistStamped
from visualization_msgs.msg import MarkerArray, Marker
from copy import deepcopy
import datetime, os, pickle

class NetSubscriber:
    def __init__(self):
        self.lock = threading.Lock()
        self.self_traj_sub = rospy.Subscriber("/planning/local_path", Lane, self.self_traj_callback)
        self.velocity_sub = rospy.Subscriber("/localization/current_velocity", TwistStamped, self.velocity_callback)
        self.objects_sub = rospy.Subscriber("tracked_objects",
                                            DetectedObjectArray, self.sub_callback)
        self.self_traj_exists = False
        self.marker_pub = rospy.Publisher('predicted_behaviours_net', MarkerArray, queue_size=1, tcp_nodelay=True)
        self.self_new_traj = []
        self.self_traj_history = []
        self.all_raw_trajectories = {}
        self.all_raw_velocities = {}
        self.all_predictions_history = {}
        self.all_predictions_history_danger_value = {}
        self.raw_trajectories = {}
        self.raw_velocities = {}
        self.self_traj = []
        self.velocity = 0.0
        self.all_endpoints = {}
        self.current_endpoints = {}
        self.active_keys = set()

    def self_traj_callback(self, lane):
        if len(lane.waypoints) > 3:
            self.self_traj_exists = True
        new_traj = []
        for waypoint in lane.waypoints:
            new_traj.append([waypoint.pose.pose.position.x, waypoint.pose.pose.position.y])
        self.self_new_traj = new_traj.copy()

    def velocity_callback(self, twist):
        self.velocity = (twist.twist.linear.x ** 2 + twist.twist.linear.y ** 2 + twist.twist.linear.z ** 2) ** 0.5

    def sub_callback(self, detectedobjectarray):
        # cash messages every callback into last element
        # print('Detected:', detectedobjectarray.header.seq)
        active_keys = set()
        for i, detectedobject in enumerate(detectedobjectarray.objects):
            if detectedobject.label == 'pedestrian':
                position = np.array([detectedobject.pose.position.x, detectedobject.pose.position.y])
                velocity = np.array([detectedobject.velocity.linear.x, detectedobject.velocity.linear.y])
                _id = detectedobject.id
                active_keys.add(_id)
                # print('Detected object:', i, 'pose:', position)

                if not _id in self.all_raw_trajectories:
                    self.all_raw_trajectories[_id] = [position]
                    self.all_raw_velocities[_id] = [velocity]
                    self.all_endpoints[_id] = 0
                    self.all_predictions_history[_id] = [[]]
                    self.all_predictions_history_danger_value[_id] = []
                else:
                    self.all_raw_trajectories[_id][len(self.all_raw_trajectories[_id]) - 1] = position
                    self.all_raw_velocities[_id][len(self.all_raw_velocities[_id]) - 1] = velocity
        with self.lock:
            self.active_keys = active_keys

    def publish_markers(self, endpoints, inference_result, inference_colors, endpoint_colors, avg_danger_values, predictions_amount):
        marker_array = MarkerArray()
        marker_array.markers = []
        # Endpoint markers and danger values
        for i, endpoint in enumerate(endpoints):
            # Endpoint markers
            marker = Marker()
            marker.id = 2 * i
            marker.header.frame_id = 'map'
            marker.header.stamp = rospy.Time.now()
            marker.type = 2

            marker.scale.x = 3.0
            marker.scale.y = 3.0
            marker.scale.z = 3.0
            marker.color.a = 3.0
            if endpoint_colors[i][0] == 'g':
                marker.color.g = 1.0
            elif endpoint_colors[i][0] == 'y':
                marker.color.r = 1.0
                marker.color.g = 1.0
            elif endpoint_colors[i][0] == 'r':
                marker.color.r = 1.0
            else:
                marker.color.b = 1.0
            marker.text = 'dng: ' + str(avg_danger_values[i])

            marker.pose.position.x = endpoint[0]
            marker.pose.position.y = endpoint[1]

            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            marker_array.markers.append(marker)
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

    def move_endpoints(self):
        # Move end-point
        with self.lock:
            for _id in self.active_keys:
                self.all_endpoints[_id] += 1
                self.all_raw_trajectories[_id].append(self.all_raw_trajectories[_id][len(self.all_raw_trajectories[_id]) - 1])
                self.all_raw_velocities[_id].append(self.all_raw_velocities[_id][len(self.all_raw_velocities[_id]) - 1])

    def on_shutdown(self):
        # Saving things
        name_test = str(datetime.datetime.now())[:10]
        folder_test = 'testing/' + name_test + '_' + str(self.__class__.__name__) + '_' + \
                      os.path.basename(os.path.dirname(rospy.get_param('/scenario_simulator/path'))) + '_run0'
        i = 0
        while os.path.exists(folder_test):
            i = i + 1
            folder_test = folder_test[:-1] + str(i)
        os.makedirs(folder_test)

        with open(os.path.join(folder_test, 'all_raw_trajectories.pkl'), 'wb') as f:
            pickle.dump(self.all_raw_trajectories, f)
        with open(os.path.join(folder_test, 'all_predictions_history.pkl'), 'wb') as f:
            pickle.dump(self.all_predictions_history, f)
        with open(os.path.join(folder_test, 'all_predictions_history_danger_value.pkl'), 'wb') as f:
            pickle.dump(self.all_predictions_history_danger_value, f)
        with open(os.path.join(folder_test, 'self_traj_history.pkl'), 'wb') as f:
            pickle.dump(self.self_traj_history, f)

        print('Trajectories and predictions saved')



