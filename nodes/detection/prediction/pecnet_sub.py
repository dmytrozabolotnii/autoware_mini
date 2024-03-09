#!/usr/bin/env python

import rospy
import torch
import time
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
        self.model = self.model.double().to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.predictions_amount = 1

        rospy.loginfo(rospy.get_name() + " - initialized")

    def inference_callback(self, event):
        if len(self.active_keys):
            # Run inference
            with self.lock:
                temp_active_keys = set(self.active_keys)
                if self.use_backpropagation:
                    [self.cache[key].backpropagate_trajectories(pad_past=self.hyper_params["past_length"] *
                                                                         (self.skip_points + 1))
                     for key in temp_active_keys if self.cache[key].endpoints_count == 0]
                temp_raw_trajectories = [self.cache[key].raw_trajectories[-1::-1 * (self.skip_points + 1)][::-1]
                                         for key in temp_active_keys]
                temp_endpoints = [self.cache[key].endpoints_count // (self.skip_points + 1)
                                  for key in temp_active_keys]
            self.move_endpoints()
            inference_dataset = PECNetDatasetInit(temp_raw_trajectories,
                                                  end_points=temp_endpoints,
                                                  pad_past=self.hyper_params["past_length"],
                                                  pad_future=self.hyper_params["future_length"],
                                                  dist_thresh=self.hyper_params["dist_thresh"]
                                                      )
            inference_result = pecnet_iter(inference_dataset, self.model, self.device, self.hyper_params, n=self.predictions_amount)
            # Update history of inferences
            for j, _id in enumerate(temp_active_keys):
                self.cache[_id].extend_prediction_history(inference_result[i][j] for i in range(len(inference_result)))


if __name__ == '__main__':
    if not torch.cuda.is_available():
        rospy.logerr("Cuda is not available")
    rospy.init_node('pecnet_predictor', anonymous=True)

    sub = PECNetSubscriber()
    sub.run()

