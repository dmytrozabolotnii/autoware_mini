#!/usr/bin/env python

import rospy
import torch
import os.path as osp

from net_sub import NetSubscriber
from sgnet_utils import SGNet, parse_sgnet_args, sgnet_iter, SGNetDatasetInit


class SGNetSubscriber(NetSubscriber):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # utl.set_seed(int(args.seed))
        self.model = SGNet(args)
        # TODO: remove hardcoded values
        self.pad_past = 8
        self.future_horizon = 12
        self.predictions_amount = 1
        self.model = self.model.double().to(self.device)
        if osp.isfile(rospy.get_param('~data_path_prediction') + args.checkpoint):
            self.checkpoint = torch.load(rospy.get_param('~data_path_prediction') + args.checkpoint, map_location=self.device)
            self.model.load_state_dict(self.checkpoint['model_state_dict'])

    def inference_callback(self, event):
        if len(self.active_keys):
            # Run inference
            with self.lock:
                temp_raw_trajectories = [self.all_raw_trajectories[key][:] for key in self.all_raw_trajectories if key in self.active_keys]
                temp_velocities = [self.all_raw_velocities[key][:] for key in self.all_raw_velocities if key in self.active_keys]
                temp_endpoints = [self.all_endpoints[key] for key in self.all_endpoints if key in self.active_keys]
                temp_active_keys = set(self.active_keys)

            inference_dataset = SGNetDatasetInit(temp_raw_trajectories, temp_velocities,
                                                 end_points=temp_endpoints,
                                                 pad_past=8,
                                                 pad_future=0,
                                                 inference_timer_duration=self.inference_timer_duration)

            inference_result = sgnet_iter(inference_dataset, self.model, self.device, n=self.predictions_amount)
            # Update history of inferences
            for j, _id in enumerate(temp_active_keys):
                for i in range(len(inference_result)):
                    self.all_predictions_history[_id][len(self.all_predictions_history[_id]) - 1]\
                        .append(inference_result[i][j])
                self.all_predictions_history[_id].append([])
            # Process points for danger values
            self.calculate_danger_values(inference_dataset, inference_result, temp_active_keys,
                                         self.future_horizon, self.pad_past)
            self.move_endpoints()


if __name__ == '__main__':
    if not torch.cuda.is_available():
        rospy.logerr("Cuda is not available")
    rospy.init_node('sgnet_predictor', anonymous=True)

    sub = SGNetSubscriber(parse_sgnet_args())
    sub.run()

