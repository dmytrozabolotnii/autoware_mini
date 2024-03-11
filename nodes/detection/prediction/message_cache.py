# Class for caching info from messages of type DetectedObject for inference with separate timer
import numpy as np

class MessageCache:
    def __init__(self, id,
                 initial_trajectory_point,
                 initial_velocity, initial_acceleration, initial_header, delta_t=0.5):
        self.id = id
        self.endpoints_count = 0
        self.raw_trajectories = [initial_trajectory_point]
        self.raw_velocities = [initial_velocity]
        self.raw_accelerations = [initial_acceleration]
        self.prediction_history = [[]]
        self.predictions_history_headers = [initial_header]
        # Approximate time between points
        self.delta_t = delta_t

    def backpropagate_trajectories(self, pad_past=8):
        # Create fake history of movement depending on velocity and acceleration vector we receive
        self.raw_accelerations = [self.raw_accelerations[0]] * pad_past + self.raw_accelerations
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
        self.predictions_history_headers[len(self.predictions_history_headers) - 1] = header

    def return_last_prediction(self):
        return self.prediction_history[len(self.prediction_history) - 1]

    def return_last_header(self):
        return self.predictions_history_headers[len(self.predictions_history_headers) - 1]

    def extend_prediction_history(self, prediction):
        self.prediction_history.append(list(prediction))

    def move_endpoints(self):
        self.endpoints_count += 1
        self.raw_trajectories.append(self.raw_trajectories[len(self.raw_trajectories) - 1])
        self.raw_velocities.append(self.raw_velocities[len(self.raw_velocities) - 1])
        self.raw_accelerations.append(self.raw_accelerations[len(self.raw_accelerations) - 1])
        self.predictions_history_headers.append(self.predictions_history_headers
                                                [len(self.predictions_history_headers) - 1])
