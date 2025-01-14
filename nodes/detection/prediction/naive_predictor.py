#!/usr/bin/env python3

import rospy
import numpy as np
import shapely
from helpers.shapely import get_polygon_width_and_prediction_origin
from helpers.geometry import get_vector_norm_3d, get_heading_from_vector
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
            ('prediction_origin', np.float32, (2,)),
            ('velocity', np.float32, (2,)),
            ('acceleration', np.float32, (2,)),
        ])

        buffer_widths = []
        valid_indices = []  # Keep track of indices for objects that need naive prediction

        for i, obj in enumerate(msg.objects):
            # Skip objects that do not need predictions
            if get_vector_norm_3d(obj.velocity) < self.prediction_min_speed or len(obj.candidate_trajectories.paths) > 0:
                buffer_widths.append(None)  # Placeholder for skipped objects
                continue

            # calculate objcet width and origin for prediction
            object_polygon = shapely.Polygon([(p.x, p.y) for p in obj.convex_hull.points])
            object_heading = get_heading_from_vector(obj.velocity)
            if self.use_object_width:
                buffer_width, center_front, _ = get_polygon_width_and_prediction_origin(object_polygon, object_heading)
                tracked_objects_array[i]['prediction_origin'] = (center_front.x, center_front.y)
            else:
                buffer_width = 0.0
                tracked_objects_array[i]['prediction_origin'] = (obj.position.x, obj.position.y)

            tracked_objects_array[i]['velocity'] = (obj.velocity.x, obj.velocity.y)
            tracked_objects_array[i]['acceleration'] = (obj.acceleration.x, obj.acceleration.y)

            buffer_widths.append(buffer_width)
            valid_indices.append(i)

        # Predict future positions and velocities - includes also initial step, thus + 1
        num_timesteps = int(self.prediction_horizon // self.prediction_interval) + 1
        predicted_objects_array = np.empty((num_timesteps, num_objects), dtype=[
            ('prediction_origin', np.float32, (2,)),
            ('velocity', np.float32, (2,)),
        ])
        predicted_objects_array[0] = tracked_objects_array[['prediction_origin', 'velocity']]
        for t in range(1, num_timesteps):
            predicted_objects_array[t]['velocity'] = predicted_objects_array[t - 1]['velocity'] + tracked_objects_array['acceleration'] * self.prediction_interval
            predicted_objects_array[t]['prediction_origin'] = predicted_objects_array[t - 1]['prediction_origin'] + predicted_objects_array[t - 1]['velocity'] * self.prediction_interval

        # Create candidate trajectories
        for i in valid_indices:
            obj = msg.objects[i]
            path = Path()

            for t in range(num_timesteps):
                wp = Waypoint()
                wp.position.x, wp.position.y = predicted_objects_array[t][i]['prediction_origin']
                wp.position.z = obj.position.z
                wp.speed = np.linalg.norm(predicted_objects_array[t][i]['velocity'])
                wp.left_width = buffer_widths[i]
                wp.right_width = buffer_widths[i]
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