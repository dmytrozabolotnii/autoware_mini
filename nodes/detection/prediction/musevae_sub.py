#!/usr/bin/env python

import numpy as np
import rospy
import torch
import os.path as osp
import time
from PIL import Image

from net_sub import NetSubscriber
from musevae_utils import parse_args, musevae_iter, MuseVAEDatasetInit, load_checkpoint, eval_mode


class MuseVAESubscriber(NetSubscriber):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = load_checkpoint(osp.join(rospy.get_param('~data_path_prediction'), 'MUSE_VAE', 'pretrained_models_sdd'), self.device)
        self.models = eval_mode(self.models)
        # ade_min, fde_min = solver.all_evaluation()

        self.past_horizon = args.obs_len
        self.future_horizon = args.pred_len
        self.global_map = Image.open(osp.join(rospy.get_param('~data_path_prediction'), 'MUSE_VAE', 'demo_route_map_local.tif'))
        # if osp.isfile(rospy.get_param('~data_path_prediction') + args.checkpoint):
        #     self.checkpoint = torch.load(rospy.get_param('~data_path_prediction') + args.checkpoint, map_location=self.device)
        #     self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.predictions_amount = rospy.get_param('~predictions_amount')
        self.pad_past = self.past_horizon
        self.class_init = True

        rospy.loginfo(rospy.get_name() + " - initialized")

    def inference_callback(self, event):
        if len(self.active_keys) and self.model is not None and next(self.model.parameters()).is_cuda and self.class_init:
            # Run inference
            with self.lock:
                temp_active_keys = set(self.active_keys)
                if self.use_backpropagation:
                    [self.cache[key].backpropagate_trajectories(pad_past=self.past_horizon *
                                                                         (self.skip_points + 1))
                     for key in temp_active_keys if self.cache[key].endpoints_count == 0]

                temp_raw_trajectories = [self.cache[key].raw_trajectories[-1::-1 * (self.skip_points + 1)][::-1]
                                         for key in temp_active_keys]
                temp_raw_velocities = [self.cache[key].raw_velocities[-1::-1 * (self.skip_points + 1)][::-1]
                                       for key in temp_active_keys]
                temp_raw_acceleration = [self.cache[key].raw_accelerations[-1::-1 * (self.skip_points + 1)][::-1]
                                         for key in temp_active_keys]

                temp_endpoints = [self.cache[key].endpoints_count // (self.skip_points + 1) for key in temp_active_keys]
                temp_headers = [self.cache[key].return_last_header() for key in temp_active_keys]


            inference_dataset = MuseVAEDatasetInit(temp_raw_trajectories, temp_raw_velocities, temp_raw_acceleration, self.global_map,
                                                   end_points=temp_endpoints,
                                                   pad_past=self.past_horizon - 1,
                                                   pad_future=0,
                                                   inference_timer_duration=self.inference_timer_duration * (self.skip_points + 1))

            inference_result = musevae_iter(inference_dataset, self.model, self.device, self.args, n=self.predictions_amount)
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
    rospy.init_node('musevae_predictor', anonymous=True)

    sub = MuseVAESubscriber(parse_args())
    sub.run()

