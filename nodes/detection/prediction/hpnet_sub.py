#!/usr/bin/env python

import numpy as np
import rospy
import torch
import time
from hpnet_ulils import HPNet
from argparse import ArgumentParser
from net_sub import NetSubscriber
import pytorch_lightning as pl

class HPNetSubscriber(NetSubscriber):

    def __init__(self):
        super().__init__()
        # initialize network
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        parser = ArgumentParser()
        parser.add_argument('--root', type=str, required=True)
        parser.add_argument('--test_batch_size', type=int, required=True)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--pin_memory', type=bool, default=True)
        parser.add_argument('--persistent_workers', type=bool, default=True)
        parser.add_argument('--devices', type=int, default=self.device)
        # parser.add_argument('--ckpt_path', type=str, required=True)
        HPNet.add_model_specific_args(parser)
        args = parser.parse_args()

        model = HPNet.load_from_checkpoint(checkpoint_path='HPNet/HPNet_interaction.ckpt')
        trainer = pl.Trainer(devices=args.devices, accelerator='gpu')
        test_dataset = INTERACTIONDataset(args.root, 'test', transform=LaneRandomOcclusion(0.0))
        dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=args.pin_memory,
                                persistent_workers=args.persistent_workers)
        trainer.test(model, dataloader)


        self.predictions_amount = rospy.get_param('~predictions_amount')
        self.pad_past = self.hyper_params["past_length"]
        self.class_init = True

        rospy.loginfo(rospy.get_name() + " - initialized")

    def inference_callback(self, event):
        if len(self.active_keys) and self.model is not None and next(self.model.parameters()).is_cuda and self.class_init:
            # Run inference
            with self.lock:
                temp_active_keys = set(self.active_keys)
                if self.use_backpropagation:
                    [self.cache[key].backpropagate_trajectories(pad_past=self.hyper_params["past_length"] *
                                                                         (self.skip_points + 1))
                     for key in temp_active_keys if self.cache[key].endpoints_count == 0]

                temp_raw_trajectories = [self.cache[key].return_last_interpolated_trajectory(self.pad_past, self.inference_timer_duration, self.hide_past) for key in temp_active_keys]
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
    rospy.init_node('hpnet_predictor', anonymous=True)

    sub = HPNetSubscriber()
    sub.run()

