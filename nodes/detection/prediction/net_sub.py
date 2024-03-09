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
        self.use_backpropagation = bool(rospy.get_param('~predictor_backfill'))
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
                    if _id not in self.cache:
                        self.cache[_id] = MessageCache(_id, position, velocity, header,
                                                       delta_t=self.inference_timer_duration)
                    else:
                        self.cache[_id].update_last_trajectory_velocity(position, velocity, header)
        with self.lock:
            self.active_keys = self.active_keys.union(active_keys)
        # Publish objects back retrieving candidate trajectories from history of inferences
        self.publish_predicted_objects(detectedobjectarray)

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
                    # Add prediction

                    for j in predictions:
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

