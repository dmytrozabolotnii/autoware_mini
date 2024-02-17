#!/usr/bin/env python

import rospy
import numpy as np
import torch
from PECNet_utils import PECNet, PECNetDatasetInit, pecnet_iter
from common_utils import color_points, calculate_danger_value
from Net_sub import NetSubscriber
from copy import deepcopy

class PECNetSubscriber(NetSubscriber):

    def __init__(self):
        super().__init__()
        rospy.loginfo(self.__class__.__name__ + " - Initializing")
        # initialize network
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.checkpoint = torch.load(rospy.get_param('~data_path_prediction') + 'PECNet/PECNET_social_model2.pt', map_location=self.device)
        self.hyper_params = self.checkpoint["hyper_params"]
        self.model = PECNet(self.hyper_params["enc_past_size"],
                            self.hyper_params["enc_dest_size"],
                            self.hyper_params["enc_latent_size"],
                            self.hyper_params["dec_size"],
                            self.hyper_params["predictor_hidden_size"],
                            self.hyper_params['non_local_theta_size'],
                            self.hyper_params['non_local_phi_size'],
                            self.hyper_params['non_local_g_size'],
                            self.hyper_params["fdim"],
                            self.hyper_params["zdim"],
                            self.hyper_params["nonlocal_pools"],
                            self.hyper_params['non_local_dim'],
                            self.hyper_params["sigma"],
                            self.hyper_params["past_length"],
                            self.hyper_params["future_length"], verbose=True)
        self.inference_timer_duration = 0.5
        self.model = self.model.double().to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.predictions_amount = 5
        # initialize the subscriber nodes and inference timer
        self.inference_timer = rospy.Timer(rospy.Duration(self.inference_timer_duration), self.inference_callback)
        print("Initializing the instance!")

    def inference_callback(self, event):
        if len(self.active_keys):
            # Run inference
            with self.lock:
                temp_raw_trajectories = [self.all_raw_trajectories[key][:] for key in self.all_raw_trajectories if key in self.active_keys]
                temp_endpoints = [self.all_endpoints[key] for key in self.all_endpoints if key in self.active_keys]
                temp_active_keys = set(self.active_keys)
            # print(self.active_keys)

            inference_dataset = PECNetDatasetInit(temp_raw_trajectories,
                                                  end_points=temp_endpoints,
                                                  pad_past=self.hyper_params["past_length"],
                                                  pad_future=self.hyper_params["future_length"],
                                                  set_name="test",
                                                  verbose=True,
                                                      )
            inference_result = pecnet_iter(inference_dataset, self.model, self.device, self.hyper_params, n=self.predictions_amount)
            # Update history of inferences
            for j, _id in enumerate(temp_active_keys):
                for i in range(len(inference_result)):
                    self.all_predictions_history[_id][len(self.all_predictions_history[_id]) - 1]\
                        .append(inference_result[i][j])
                self.all_predictions_history[_id].append([])
            # print(self.all_predictions_history)
            # print(list(self.all_predictions_history.values())[0])
            # print('Timestamp:', event.current_real, 'Results:', inference_result, 'Endpoints:', temp_endpoints)
            # Update trajectory and process points
            if self.self_traj_exists:
                trajectory_length = self.velocity * self.inference_timer_duration * self.hyper_params["future_length"]
                self.self_traj = self.self_new_traj.copy()
                inference_colors = (list(map(color_points, inference_result, [self.self_traj] * len(inference_result),
                                             [trajectory_length] * len(inference_result))))
                # print('Inference_colors:', inference_colors)
                endpoint_colors = color_points(inference_dataset.traj[:, self.hyper_params['past_length'] - 1],
                                               self.self_traj, trajectory_length)
                # print('Endpoint_colors:', endpoint_colors)
                danger_values = np.zeros((len(endpoint_colors), len(inference_colors)))
                for i in range(len(endpoint_colors)):
                    for j in range(self.predictions_amount):
                        danger_values[i, j] = calculate_danger_value(endpoint_colors[i], inference_colors[j][i])
                # print(danger_values)
                avg_danger_values = np.mean(danger_values, axis=1)

                for j, _id in enumerate(temp_active_keys):
                    self.all_predictions_history_danger_value[_id].append(avg_danger_values[j])
                self.self_traj_history.append(self.self_traj[0])

                # print('Danger_values:\n', avg_danger_values)
                self.publish_markers(inference_dataset.traj[:, self.hyper_params['past_length'] - 1],
                                     inference_result, inference_colors, endpoint_colors, avg_danger_values, self.predictions_amount)
            self.publish_predicted_objects()
            self.move_endpoints()


def main():
    # create a subscriber instance
    rospy.init_node('PECNetPredictor', anonymous=True)

    sub = PECNetSubscriber()

    # initializing the subscriber node
    rospy.on_shutdown(sub.on_shutdown)
    rospy.spin()


if __name__ == '__main__':
    try:
        print('Cuda:', torch.cuda.is_available())
        main()
    except rospy.ROSInterruptException:
        pass
