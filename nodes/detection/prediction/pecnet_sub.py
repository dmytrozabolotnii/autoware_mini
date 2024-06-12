#!/usr/bin/env python

import numpy as np
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
        self.checkpoint = torch.load(rospy.get_param('data_path_prediction') + 'PECNet/PECNET_social_model1.pt', map_location=self.device)
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
        self.predictions_amount = rospy.get_param('~predictions_amount')
        self.pad_past = self.hyper_params["past_length"]
        self.class_init = True
        self.timer = time.time()

        rospy.loginfo(rospy.get_name() + " - initialized")

    def inference_callback(self, event):
        # print('Time of callback', time.time() - self.timer)
        self.timer = time.time()

        if len(self.active_keys) and self.model is not None and next(self.model.parameters()).is_cuda and self.class_init:
            # Run inference
            with self.lock:
                temp_active_keys = set(self.active_keys)
                if self.use_backpropagation:
                    [self.cache[key].backpropagate_trajectories(pad_past=self.hyper_params["past_length"] *
                                                                         (self.skip_points + 1))
                     for key in temp_active_keys if self.cache[key].endpoints_count == 0]

                # temp_temp_raw_trajectories = [self.cache[key].raw_trajectories[-1::-1 * (self.skip_points + 1)][::-1]
                #                          for key in temp_active_keys]
                temp_raw_trajectories = [self.cache[key].return_last_interpolated_trajectory(self.pad_past, self.inference_timer_duration) for key in temp_active_keys]
                temp_endpoints = [self.cache[key].endpoints_count // (self.skip_points + 1)
                                  for key in temp_active_keys]
                temp_headers = [self.cache[key].return_last_header() for key in temp_active_keys]

            inference_dataset = PECNetDatasetInit(temp_raw_trajectories,
                                                  end_points=temp_endpoints,
                                                  pad_past=self.hyper_params["past_length"] - 1,
                                                  pad_future=0,
                                                  dist_thresh=self.hyper_params["dist_thresh"] / 2
                                                      )
            inference_result = pecnet_iter(inference_dataset, self.model, self.device, self.hyper_params, n=self.predictions_amount)
            # Update history of inferences
            for j, _id in enumerate(temp_active_keys):
                with self.lock:
                    # Append the ego-position and header at start of inference for metrics purpose
                    self.cache[_id].extend_prediction_history(np.vstack(([temp_raw_trajectories[j][-1]], inference_result[i][j]))
                                                               for i in range(len(inference_result)))
                    self.cache[_id].extend_prediction_header_history(temp_headers[j])
            self.move_endpoints()



if __name__ == '__main__':
    if not torch.cuda.is_available():
        rospy.logerr("Cuda is not available")
    rospy.init_node('pecnet_predictor', anonymous=True)

    sub = PECNetSubscriber()
    sub.run()

