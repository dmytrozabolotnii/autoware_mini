#!/usr/bin/env python3

import math
import rospy
import numpy as np

from helpers.geometry import get_vector_norm_3d
from autoware_mini.msg import DetectedObjectArray, Path, Waypoint

class NaivePredictor:
    def __init__(self):
        # Parameters
        self.prediction_horizon = rospy.get_param('~prediction_horizon')
        self.prediction_interval = rospy.get_param('~prediction_interval')
        self.prediction_min_speed = rospy.get_param('~prediction_min_speed')

        # Publishers
        self.predicted_objects_pub = rospy.Publisher('predicted_objects', DetectedObjectArray, queue_size=1, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('tracked_objects', DetectedObjectArray, self.tracked_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)

    def tracked_objects_callback(self, msg):
        # Convert tracked objects to numpy array
        tracked_objects_array = np.empty((len(msg.objects)), dtype=[
            ('centroid', np.float32, (2,)),
            ('velocity', np.float32, (2,)),
            ('acceleration', np.float32, (2,)),
        ])
        for i, obj in enumerate(msg.objects):
            tracked_objects_array[i]['centroid'] = (obj.pose.position.x, obj.pose.position.y)
            tracked_objects_array[i]['velocity'] = (obj.velocity.x, obj.velocity.y) 
            tracked_objects_array[i]['acceleration'] = (obj.acceleration.x, obj.acceleration.y)

        # Predict future positions and velocities - includes also initial step, thus + 1
        num_timesteps = int(self.prediction_horizon // self.prediction_interval) + 1
        predicted_objects_array = np.empty((num_timesteps, len(msg.objects)), dtype=[
            ('centroid', np.float32, (2,)),
            ('velocity', np.float32, (2,)),
        ])
        predicted_objects_array[0] = tracked_objects_array[['centroid', 'velocity']]
        for i in range(1, num_timesteps):
            predicted_objects_array[i]['centroid'] = predicted_objects_array[i-1]['centroid'] + predicted_objects_array[i-1]['velocity'] * self.prediction_interval
            predicted_objects_array[i]['velocity'] = predicted_objects_array[i-1]['velocity'] + tracked_objects_array['acceleration'] * self.prediction_interval

        # Create candidate trajectories
        for i, obj in enumerate(msg.objects):
            # Skip prediction for near stationary objects
            if get_vector_norm_3d(obj.velocity) < self.prediction_min_speed:
                continue

            # Skip prediction if candidate trajectories already exist
            if len(obj.candidate_trajectories.paths) > 0:
                continue

            path = Path()
            for j in range(num_timesteps):
                wp = Waypoint()
                wp.position.x, wp.position.y = predicted_objects_array[j][i]['centroid']
                wp.position.z = obj.pose.position.z
                wp.speed = (predicted_objects_array[j][i]['velocity'][0]**2 + predicted_objects_array[j][i]['velocity'][1]**2)**0.5
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