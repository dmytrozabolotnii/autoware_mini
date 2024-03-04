# Class for caching info from messages of type DetectedObject for inference with separate timer

class MessageCache:
    def __init__(self, id,
                 initial_trajectory_point,
                 initial_velocity):
        self.id = id
        self.endpoints_count = 0
        self.raw_trajectories = [initial_trajectory_point]
        self.raw_velocities = [initial_velocity]
        self.prediction_history = [[]]
        self.predictions_history_danger_value = []

    def backpropagate_trajectories(self, pad_past=8):
        # Create fake history of movement depending on velocity vector we receive
        self.raw_trajectories = [self.raw_trajectories[0] + self.raw_velocities[0] * i
                                 for i in range(-1 * pad_past, 0)] + self.raw_trajectories
        self.raw_velocities = [self.raw_velocities[0]] * pad_past + self.raw_velocities
        self.endpoints_count = pad_past + 1

    def update_last_trajectory_velocity(self, trajectory, velocity):
        self.raw_trajectories[len(self.raw_trajectories) - 1] = trajectory
        self.raw_velocities[len(self.raw_velocities) - 1] = velocity

    def return_last_prediction(self):
        return self.prediction_history[len(self.prediction_history) - 1]

    def extend_prediction_history(self, prediction):
        self.prediction_history.append(list(prediction))

    def extend_prediction_history_danger_value(self, danger_value):
        self.predictions_history_danger_value.append(danger_value)

    def move_endpoints(self):
        self.endpoints_count += 1
        self.raw_trajectories.append(self.raw_trajectories[len(self.raw_trajectories) - 1])
        self.raw_velocities.append(self.raw_velocities[len(self.raw_velocities) - 1])
