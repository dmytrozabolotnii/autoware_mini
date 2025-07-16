from abc import ABCMeta, abstractmethod

import numpy as np
import rospy
import threading
import time

from autoware_mini.msg import Path, DetectedObjectArray, Waypoint

from autoware_mini.message_cache import MessageCache

from metrics_calculator import MetricsCalculator


class NetSubscriber(metaclass=ABCMeta):
    def __init__(self):
        self.lock = threading.Lock()
        # Caching structure
        # Dict of Message Cache class values
        self.cache = {}
        self.cache_cars = {}
        self.active_keys = set()
        self.active_keys_cars = set()
        self.collect_car_info = rospy.get_param('~cars_constraints', False)
        # Basic inference values

        # Inference is run every these seconds:
        self.inference_timer_duration = rospy.get_param('inference_timer')
        # Effectively means points for trajectories for inference are taken
        # every inference_timer * (skip_points + 1) seconds:
        self.skip_points = int(rospy.get_param('step_length') / rospy.get_param('inference_timer')) - 1
        self.model = None
        self.use_backpropagation = bool(rospy.get_param('predictor_backfill'))
        self.pad_past = int(rospy.get_param('prediction_history'))
        self.hide_past = int(rospy.get_param('prediction_history_hide'))

        # ROS timers/pub/sub
        self.class_init = False
        self.inference_timer = rospy.Timer(rospy.Duration(self.inference_timer_duration), self.inference_callback, reset=True)
        self.objects_pub = rospy.Publisher('predicted_objects', DetectedObjectArray, queue_size=1,
                                           tcp_nodelay=True)
        self.objects_sub = rospy.Subscriber("tracked_objects",
                                            DetectedObjectArray, self.detected_objects_sub_callback)
        self.metrics_node = MetricsCalculator()

    def detected_objects_sub_callback(self, detectedobjectarray):
        # cache objects with filter, so we can refer to them at inference time
        active_keys = set()
        active_keys_cars = set()
        for i, detectedobject in enumerate(detectedobjectarray.objects):
            if detectedobject.label == 'pedestrian' or detectedobject.label == 'unknown':
                position = np.array([detectedobject.center.x, detectedobject.center.y])
                velocity = np.array([detectedobject.velocity.x, detectedobject.velocity.y])
                acceleration = np.array([detectedobject.acceleration.x, detectedobject.acceleration.y])
                convex_hull = detectedobject.convex_hull
                header = detectedobjectarray.header
                _id = detectedobject.id
                active_keys.add(_id)
                with self.lock:
                    if _id not in self.cache:
                        self.cache[_id] = MessageCache(_id, position, velocity, acceleration, header,
                                                       pad_past=self.pad_past, hide_past=self.hide_past, delta_t=self.inference_timer_duration, convex_hull=convex_hull)

                    else:
                        self.cache[_id].move_endpoints()
                        self.cache[_id].update_last_trajectory(position, velocity, acceleration, header, convex_hull=convex_hull)
            elif self.collect_car_info and (detectedobject.label == 'bicycle' or detectedobject.label == 'car'):
                position = np.array([detectedobject.center.x, detectedobject.center.y])
                velocity = np.array([detectedobject.velocity.x, detectedobject.velocity.y])
                acceleration = np.array([detectedobject.acceleration.x, detectedobject.acceleration.y])
                convex_hull = detectedobject.convex_hull
                header = detectedobjectarray.header
                _id = detectedobject.id
                active_keys_cars.add(_id)
                with self.lock:
                    if _id not in self.cache_cars:
                        self.cache_cars[_id] = MessageCache(_id, position, velocity, acceleration, header,
                                                       pad_past=self.pad_past, hide_past=self.hide_past, delta_t=self.inference_timer_duration, convex_hull=convex_hull)

                    else:
                        self.cache_cars[_id].move_endpoints()
                        self.cache_cars[_id].update_last_trajectory(position, velocity, acceleration, header, convex_hull=convex_hull)

        with self.lock:
            self.active_keys = self.active_keys.union(active_keys)
            self.active_keys_cars = self.active_keys_cars.union(active_keys_cars)
        # Publish objects back retrieving candidate trajectories from history of inferences
        self.publish_predicted_objects(detectedobjectarray)

    @abstractmethod
    def inference_callback(self, event):
        pass

    def publish_predicted_objects(self, detectedobjectsarray):
        # Construct candidate predictors from saved history of predictions
        output_msg_array = DetectedObjectArray(header=detectedobjectsarray.header)

        for detectedobject in detectedobjectsarray.objects:
            if detectedobject.label == 'pedestrian' or detectedobject.label == 'bicycle' or detectedobject.label == 'unknown':
                with self.lock:
                    predictions = self.cache[detectedobject.id].return_last_prediction()
                    predictions_header = self.cache[detectedobject.id].return_last_prediction_header()
                for prediction in predictions:
                    lane = Path(header=predictions_header)

                    for j in prediction:
                        wp = Waypoint()
                        wp.position.x, wp.position.y = j
                        wp.position.z = detectedobject.center.z
                        # print(wp)
                        lane.waypoints.append(wp)
                    detectedobject.candidate_trajectories.paths.append(lane)

            output_msg_array.objects.append(detectedobject)
        # Publish objects with predicted candidate trajectories
        self.objects_pub.publish(output_msg_array)

    def move_endpoints(self):
        # Moves end-point of cached trajectory every inference
        with self.lock:
            # Update metrics node cache and calculate metrics

            # Calculate metrics
            if len(self.active_keys) > 0:
                self.metrics_node.calculate_metrics({key: self.cache[key] for key in self.active_keys})

            # Resets active keys
            self.active_keys = set()
            self.active_keys_cars = set()

    def run(self):
        rospy.spin()

