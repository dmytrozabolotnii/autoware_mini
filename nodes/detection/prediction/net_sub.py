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

        # Inference is run every these seconds:
        self.inference_timer_duration = 0.1
        # Effectively means points for trajectories for inference are taken
        # every inference_timer * (skip_points + 1) seconds:
        self.skip_points = 4
        self.model = None
        self.predictions_amount = 1
        self.use_backpropagation = bool(rospy.get_param('~predictor_backfill'))
        self.pad_past = 8

        # ROS timers/pub/sub
        self.inference_timer = rospy.Timer(rospy.Duration(self.inference_timer_duration), self.inference_callback, reset=True)
        self.objects_pub = rospy.Publisher('predicted_objects', DetectedObjectArray, queue_size=1,
                                           tcp_nodelay=True)
        self.objects_sub = rospy.Subscriber("tracked_objects",
                                            DetectedObjectArray, self.detected_objects_sub_callback)


    def detected_objects_sub_callback(self, detectedobjectarray):
        # cache objects with filter, so we can refer to them at inference time
        active_keys = set()
        for i, detectedobject in enumerate(detectedobjectarray.objects):
            if detectedobject.label == 'pedestrian' or detectedobject.label == 'unknown':
                position = np.array([detectedobject.pose.position.x, detectedobject.pose.position.y])
                velocity = np.array([detectedobject.velocity.linear.x, detectedobject.velocity.linear.y])
                acceleration = np.array([detectedobject.acceleration.linear.x, detectedobject.acceleration.linear.y])
                header = detectedobject.header
                _id = detectedobject.id
                active_keys.add(_id)
                with self.lock:
                    if _id not in self.cache:
                        self.cache[_id] = MessageCache(_id, position, velocity, acceleration, header,
                                                       delta_t=self.inference_timer_duration)
                        #if self.use_backpropagation:
                        #    self.cache[_id].backpropagate_trajectories(pad_past=self.pad_past * (self.skip_points + 1))

                    else:
                        self.cache[_id].update_last_trajectory(position, velocity, acceleration, header)
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
            # Resets active keys
            self.active_keys = set()

    def run(self):
        rospy.spin()

