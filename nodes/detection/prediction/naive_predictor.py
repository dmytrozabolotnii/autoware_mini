#!/usr/bin/env python3

import rospy
import numpy as np
from helpers.detection import get_prediction_origin
from helpers.geometry import get_vector_norm_3d
from autoware_mini.msg import DetectedObjectArray, Path, Waypoint

class NaivePredictor:
    def __init__(self):
        # Parameters
        self.prediction_horizon = rospy.get_param('~prediction_horizon')
        self.prediction_interval = rospy.get_param('~prediction_interval')
        self.prediction_min_speed = rospy.get_param('~prediction_min_speed')
        self.use_object_width = rospy.get_param('/planning/use_object_width')

        # Publishers
        self.predicted_objects_pub = rospy.Publisher('predicted_objects', DetectedObjectArray, queue_size=1, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('tracked_objects', DetectedObjectArray, self.tracked_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)

    def tracked_objects_callback(self, msg):
        num_objects = len(msg.objects)

        # Convert tracked objects to numpy array
        tracked_objects_array = np.zeros(num_objects, dtype=[
            ('position', np.float32, (2,)),
            ('velocity', np.float32, (2,)),
            ('acceleration', np.float32, (2,)),
            ('width', np.float32),
        ])

        valid_indices = []  # Keep track of indices for objects that need naive prediction

        for i, obj in enumerate(msg.objects):
            # Skip objects that do not need predictions
            if get_vector_norm_3d(obj.velocity) < self.prediction_min_speed or len(obj.candidate_trajectories.paths) > 0:
                continue

            # calculate object width and origin for prediction
            if self.use_object_width:
                origin, width, _ = get_prediction_origin(obj)
                tracked_objects_array[i]['position'] = origin
                tracked_objects_array[i]['width'] = width
            else:
                tracked_objects_array[i]['position'] = (obj.position.x, obj.position.y)

            tracked_objects_array[i]['velocity'] = (obj.velocity.x, obj.velocity.y)
            tracked_objects_array[i]['acceleration'] = (obj.acceleration.x, obj.acceleration.y)

            valid_indices.append(i)

        # Predict future positions and velocities - includes also initial step, thus + 1
        num_timesteps = int(self.prediction_horizon // self.prediction_interval) + 1
        timesteps = np.arange(num_timesteps) * self.prediction_interval
        timesteps = timesteps[:, None, None]  # Reshape for broadcasting

        predicted_positions = (
            tracked_objects_array['position'][None, :, :]  # Shape (1, num_objects, 2)
            + tracked_objects_array['velocity'][None, :, :] * timesteps  # v0 * t
            + 0.5 * tracked_objects_array['acceleration'][None, :, :] * timesteps**2  # (1/2) * a * t^2
        )
        predicted_velocities = (
            tracked_objects_array['velocity'][None, :, :]  # Shape (1, num_objects, 2)
            + tracked_objects_array['acceleration'][None, :, :] * timesteps  # v0 + a * t
        )

        # Create candidate trajectories
        for i in valid_indices:
            obj = msg.objects[i]
            path = Path()

            for t in range(num_timesteps):
                wp = Waypoint()
                wp.position.x, wp.position.y = predicted_positions[t, i]
                wp.position.z = obj.position.z
                wp.speed = np.linalg.norm(predicted_velocities[t, i])
                wp.left_width = tracked_objects_array[i]['width']
                wp.right_width = tracked_objects_array[i]['width']
                path.waypoints.append(wp)
            obj.candidate_trajectories.paths.append(path)

        # Publish predicted objects
        self.predicted_objects_pub.publish(msg)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('naive_predictor', log_level=rospy.INFO)
    node = NaivePredictor()
    node.run()