# Class for caching info from messages of type DetectedObject for inference with separate timer
import numpy as np
from shapely import LineString, Point, LinearRing, prepare
import time

class MessageCache:
    def __init__(self, id,
                 initial_trajectory_point,
                 initial_velocity, initial_acceleration, initial_header, pad_past=8, hide_past=0, delta_t=0.5):
        self.id = id
        self.endpoints_count = 0
        self.raw_trajectories = [initial_trajectory_point]
        self.raw_velocities = [initial_velocity]
        self.raw_accelerations = [initial_acceleration]
        self.headers = [initial_header]
        self.prediction_history = [[]]
        self.predictions_history_headers = [None]
        # Approximate time between points
        self.pad_past = pad_past
        self.hide_past = hide_past
        self.delta_t = delta_t

    def backpropagate_trajectories(self, ca=False):
        pad_past = self.pad_past
        # Create fake history of movement depending on velocity and acceleration vector we receive
        if ca:
            # Constance acceleration mode
            self.raw_accelerations = [self.raw_accelerations[0]] * pad_past + self.raw_accelerations
        else:
            # Constance velocity mode
            self.raw_accelerations = [np.array([0, 0])] * pad_past + self.raw_accelerations
        new_velocities = [self.raw_velocities[0] + self.raw_accelerations[0] * i * self.delta_t
                          for i in range(-1 * pad_past, 0)]
        self.raw_velocities = new_velocities + self.raw_velocities
        velocities_cumsum = np.cumsum(new_velocities[::-1], axis=0)[::-1] * self.delta_t
        self.raw_trajectories = list(self.raw_trajectories[0] - velocities_cumsum) + self.raw_trajectories

        self.endpoints_count = pad_past

    def update_last_trajectory(self, trajectory, velocity, acceleration, header):
        self.raw_trajectories[len(self.raw_trajectories) - 1] = trajectory
        self.raw_velocities[len(self.raw_velocities) - 1] = velocity
        self.raw_accelerations[len(self.raw_accelerations) - 1] = acceleration
        self.headers[len(self.headers) - 1] = header

    def return_last_header(self):
        return self.headers[len(self.headers) - 1]

    def return_last_prediction(self):
        return self.prediction_history[len(self.prediction_history) - 1]

    def return_last_prediction_header(self):
        return self.predictions_history_headers[len(self.predictions_history_headers) - 1]

    def extend_prediction_history(self, prediction):
        self.prediction_history.append(list(prediction))

    def extend_prediction_header_history(self, header):
        self.predictions_history_headers.append(header)

    def move_endpoints(self):
        self.endpoints_count += 1
        self.raw_trajectories.append(self.raw_trajectories[len(self.raw_trajectories) - 1])
        self.raw_velocities.append(self.raw_velocities[len(self.raw_velocities) - 1])
        self.raw_accelerations.append(self.raw_accelerations[len(self.raw_accelerations) - 1])
        self.headers.append(self.headers[len(self.headers) - 1])

    def return_last_interpolated_trajectory(self, length=8, delta=0.4, hide_past=0):
        if self.endpoints_count == 0:
            return np.array([self.raw_trajectories[-1]] * length)

        cum_time = 0
        past_horizon = length * delta
        popped_points = 0
        points = []
        dist_between_points = [0]
        time_between_points = [0]
        while cum_time < past_horizon and popped_points < self.endpoints_count:
            points.append(self.raw_trajectories[-1 - popped_points])
            dist_between_points.append(np.linalg.norm(self.raw_trajectories[-1 - popped_points] - self.raw_trajectories[-1 - popped_points - 1]))
            time_between_points.append((self.headers[-1 - popped_points].stamp - self.headers[-1 - popped_points - 1].stamp).to_sec())
            cum_time += time_between_points[-1]
            popped_points += 1
        points.append(self.raw_trajectories[-1 - popped_points])
        linepoints = LineString([(point[0], point[1]) for point in points])
        prepare(linepoints)
        dist_between_points_cumsum = np.cumsum(dist_between_points)
        time_between_points_cumsum = np.cumsum(time_between_points)
        past_trajectory_normalized_times = np.arange(0, delta * length, delta)
        distance_interpolated_values = np.interp(past_trajectory_normalized_times, time_between_points_cumsum, dist_between_points_cumsum)
        past_trajectory = LineString([linepoints.interpolate(distance_interpolated_values[i]) for i in range(length)])

        return np.pad(np.flip(np.asarray(past_trajectory.coords[:length-hide_past]), axis=0), ((hide_past, 0), (0, 0)), mode='edge')

    def return_last_interpolated_velocities(self, trajectory, delta=0.4):
        return np.gradient(trajectory, delta)

    def return_last_interpolated_acceleration(self, velocities, delta=0.4):
        return np.gradient(velocities, delta)
