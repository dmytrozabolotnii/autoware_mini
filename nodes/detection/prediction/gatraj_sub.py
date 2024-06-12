#!/usr/bin/env python

import numpy as np
import rospy
import torch
import time
import onnxruntime
from gatraj_utils import GATraj, GATrajDatasetInit, gatraj_iter
from GATraj.GATraj_parser import get_args
from net_sub import NetSubscriber


class GATrajSubscriber(NetSubscriber):

    def __init__(self):
        super().__init__()
        # initialize network
        self.args = get_args()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = GATraj(self.args)
        self.checkpoint = torch.load(rospy.get_param('data_path_prediction') + 'GATraj/GATraj_1000.tar', map_location=self.device)

        self.model = self.model.to(self.device)
        self.model.eval()
        self.model.load_state_dict(self.checkpoint["state_dict"])
        self.predictions_amount = rospy.get_param('~predictions_amount')
        self.pad_past = self.args.min_obs
        self.timer = time.time()

        rospy.loginfo(rospy.get_name() + " - initialized")

    def inference_callback(self, event):
        # print('Time of callback', time.time() - self.timer)
        self.timer = time.time()

        if len(self.active_keys) and self.model is not None and next(self.model.parameters()).is_cuda:
            # Run inference
            with self.lock:
                temp_active_keys = set(self.active_keys)
                if self.use_backpropagation:
                    [self.cache[key].backpropagate_trajectories(pad_past=self.args.min_obs *
                                                                         (self.skip_points + 1))
                     for key in temp_active_keys if self.cache[key].endpoints_count == 0]

                # temp_raw_trajectories = [self.cache[key].raw_trajectories[-1::-1 * (self.skip_points + 1)][::-1]
                #                          for key in temp_active_keys]
                temp_raw_trajectories = [self.cache[key].return_last_interpolated_trajectory(self.pad_past, self.inference_timer_duration) for key in temp_active_keys]
                temp_endpoints = [self.cache[key].endpoints_count // (self.skip_points + 1)
                                  for key in temp_active_keys]
                temp_headers = [self.cache[key].return_last_header() for key in temp_active_keys]

            inference_dataset = GATrajDatasetInit(temp_raw_trajectories,
                                                  end_points=temp_endpoints,
                                                  pad_past=self.args.min_obs - 1,
                                                  pad_future=0,
                                                  dist_thresh=50 / 2
                                                      )
            t0 = time.time()

            inference_result = gatraj_iter(inference_dataset, self.model, self.device, self.args, n=self.predictions_amount)
            torch.cuda.synchronize()
            # print('Inference time:', time.time() - t0)
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
    rospy.init_node('gatraj_predictor', anonymous=True)

    sub = GATrajSubscriber()
    sub.run()

