# Adapted naive_predictor for pedestrian prediction experiments

#!/usr/bin/env python3

import rospy
import numpy as np

from autoware_msgs.msg import DetectedObjectArray, Lane, Waypoint
from net_sub import NetSubscriber

class NaivePredictor(NetSubscriber):
    def __init__(self):
        super().__init__()
        # Parameters
        self.prediction_horizon = rospy.get_param('prediction_horizon')
        self.prediction_interval = rospy.get_param('step_length')
        self.constant_velocity_mode = bool(rospy.get_param('~constant_velocity'))

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
                for i, key in enumerate(temp_active_keys):
                    tracked_objects_array[i]['centroid'] = (self.cache[key].raw_trajectories[-1][0], self.cache[key].raw_trajectories[-1][1])
                    tracked_objects_array[i]['velocity'] = (self.cache[key].raw_velocities[-1][0], self.cache[key].raw_velocities[-1][1])
                    if self.constant_velocity_mode:
                        tracked_objects_array[i]['acceleration'] = 0
                    else:
                        tracked_objects_array[i]['acceleration'] = (
                        self.cache[key].raw_accelerations[-1][0], self.cache[key].raw_accelerations[-1][1])
                temp_headers = [self.cache[key].return_last_header() for key in temp_active_keys]


            # Predict future positions and velocities
            num_timesteps = self.prediction_horizon + 1
            predicted_objects_array = np.empty((num_timesteps, len(temp_active_keys)), dtype=[
                ('centroid', np.float32, (2,)),
                ('velocity', np.float32, (2,)),
            ])
            predicted_objects_array[0] = tracked_objects_array[['centroid', 'velocity']]
            for i in range(1, num_timesteps):
                predicted_objects_array[i]['centroid'] = predicted_objects_array[i - 1]['centroid'] + \
                                                         predicted_objects_array[i - 1][
                                                             'velocity'] * self.prediction_interval
                predicted_objects_array[i]['velocity'] = predicted_objects_array[i - 1]['velocity'] + \
                                                         tracked_objects_array[
                                                             'acceleration'] * self.prediction_interval
            with self.lock:
                # Create candidate trajectories
                for i, _id in enumerate(temp_active_keys):
                    self.cache[_id].extend_prediction_history([predicted_objects_array[:, i]['centroid']])
                    self.cache[_id].extend_prediction_header_history(temp_headers[i])
            self.move_endpoints()


    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('naive_predictor', log_level=rospy.INFO)
    node = NaivePredictor()
    node.run()