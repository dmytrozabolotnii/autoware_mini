#!/usr/bin/env python

import numpy as np
import rospy
import torch
import os.path as osp
from PIL import Image
import rasterio as rio

from net_sub import NetSubscriber
from musevae_utils import parse_args, musevae_iter, MuseVAEDatasetInit, load_checkpoint, eval_mode


class MuseVAESubscriber(NetSubscriber):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = load_checkpoint(osp.join(rospy.get_param('data_path_prediction'), 'MUSE_VAE', 'pretrained_models_sdd'), self.device)
        self.models = eval_mode(self.models)
        # ade_min, fde_min = solver.all_evaluation()

        self.past_horizon = args.obs_len
        self.future_horizon = args.pred_len
        self.rio_global_map = rio.open(osp.join(rospy.get_param('data_path_prediction'), 'MUSE_VAE', 'demo_route_map_large_local.tif'))
        self.local_band = self.rio_global_map.read(1)
        self.buffer = 80
        self.map_res = 0.5
        self.predictions_amount = rospy.get_param('~predictions_amount')
        self.pad_past = self.past_horizon
        self.class_init = True

        rospy.loginfo(rospy.get_name() + " - initialized")

    def inference_callback(self, event):
        if len(self.active_keys) and self.models is not None and self.class_init:
            # Run inference
            with self.lock:
                temp_active_keys = set(self.active_keys)
                if self.use_backpropagation:
                    [self.cache[key].backpropagate_trajectories(pad_past=self.past_horizon *
                                                                         (self.skip_points + 1))
                     for key in temp_active_keys if self.cache[key].endpoints_count == 0]

                temp_raw_trajectories = [self.cache[key].return_last_interpolated_trajectory(self.pad_past, self.inference_timer_duration, self.hide_past) for key in temp_active_keys]
                temp_raw_trajectories = np.array(temp_raw_trajectories)
                local_map_min_x = np.min(temp_raw_trajectories[:, :, 0]) - self.buffer
                local_map_max_x = np.max(temp_raw_trajectories[:, :, 0]) + self.buffer
                local_map_min_y = np.min(temp_raw_trajectories[:, :, 1]) - self.buffer
                local_map_max_y = np.max(temp_raw_trajectories[:, :, 1]) + self.buffer
                row_max_x, col_min_y = self.rio_global_map.index(local_map_min_x, local_map_min_y)
                row_min_x, col_max_y = self.rio_global_map.index(local_map_max_x, local_map_max_y)
                # print(row_min_x, row_max_x, col_min_y, col_max_y)
                local_band = self.local_band[row_min_x:row_max_x, col_min_y:col_max_y]
                if local_band.size < (self.buffer * 2) ** 2:
                    print('Map empty')
                    local_band = np.ones((int((local_map_max_x - local_map_min_x) / self.map_res) + 1, int((local_map_max_y - local_map_min_y) / self.map_res) + 1))
                # print(local_band)
                initial_shift = np.array([local_map_min_x, local_map_min_y])

                temp_raw_trajectories = (temp_raw_trajectories - initial_shift) / self.map_res
                temp_raw_velocities = [np.gradient(trajectory, self.inference_timer_duration)[0] for trajectory in temp_raw_trajectories]
                temp_raw_acceleration = [np.gradient(velocities, self.inference_timer_duration)[0] for velocities in temp_raw_velocities]

                temp_endpoints = [self.cache[key].endpoints_count // (self.skip_points + 1) for key in temp_active_keys]
                temp_headers = [self.cache[key].return_last_header() for key in temp_active_keys]


            inference_dataset = MuseVAEDatasetInit(temp_raw_trajectories, temp_raw_velocities, temp_raw_acceleration, local_band,
                                                   end_points=temp_endpoints,
                                                   pad_past=self.past_horizon - 1,
                                                   pad_future=0,
                                                   inference_timer_duration=self.inference_timer_duration * (self.skip_points + 1),
                                                   device=self.device)
            inference_result = musevae_iter(inference_dataset, self.models, self.device, self.args, n=self.predictions_amount)
            inference_result = np.array(inference_result) * self.map_res + initial_shift
            temp_raw_trajectories = temp_raw_trajectories * self.map_res + initial_shift
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

