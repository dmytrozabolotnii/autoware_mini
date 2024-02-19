#!/usr/bin/env python

import rospy
import torch
from pecnet_utils import PECNet, PECNetDatasetInit, pecnet_iter
from net_sub import NetSubscriber

class PECNetSubscriber(NetSubscriber):

    def __init__(self):
        super().__init__()
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
        # TODO: remove hardcoded values
        self.model = self.model.double().to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.predictions_amount = 5

    def inference_callback(self, event):
        if len(self.active_keys):
            # Run inference
            with self.lock:
                temp_raw_trajectories = [self.all_raw_trajectories[key][:] for key in self.all_raw_trajectories if key in self.active_keys]
                temp_endpoints = [self.all_endpoints[key] for key in self.all_endpoints if key in self.active_keys]
                temp_active_keys = set(self.active_keys)

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
            # Process points for danger values
            self.calculate_danger_values(inference_dataset, inference_result, temp_active_keys,
                                         self.hyper_params["future_length"], self.hyper_params['past_length'])
            self.move_endpoints()


if __name__ == '__main__':
    if not torch.cuda.is_available():
        rospy.logerr("Cuda is not available")
    rospy.init_node('pecnet_predictor', anonymous=True)

    sub = PECNetSubscriber()
    sub.run()

