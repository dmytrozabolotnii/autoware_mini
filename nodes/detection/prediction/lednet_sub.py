#!/usr/bin/env python

import numpy as np
import rospy
import torch
import time
import os.path as osp
import yaml
from lednet_utils import make_beta_schedule, lednet_iter, LEDNetDatasetInit
from LED.models.model_led_initializer import LEDInitializer as InitializationModel
from LED.models.model_diffusion import TransformerDenoisingModel as CoreDenoisingModel

from net_sub import NetSubscriber


class LEDNetSubscriber(NetSubscriber):

    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # load config for LED
        self.cfg = yaml.safe_load(open(osp.join(rospy.get_param('~data_path_prediction'), 'LED', 'led_augment.yaml'), 'r'))

        # intialize base denoising network
        self.model = CoreDenoisingModel().cuda()
        self.model_cp = torch.load(osp.join(rospy.get_param('~data_path_prediction'), 'LED', 'base_diffusion_model.p'), map_location='cpu')
        self.model.load_state_dict(self.model_cp['model_dict'])

        # initialize initializer network
        self.model_initializer = InitializationModel(t_h=10, d_h=6, t_f=20, d_f=2, k_pred=20).cuda()
        self.model_initializer_cp = torch.load(osp.join(rospy.get_param('~data_path_prediction'), 'LED', 'led_new.p'), map_location=torch.device('cpu'))
        self.model_initializer.load_state_dict(self.model_initializer_cp['model_initializer_dict'])

        # initialize diffusion parameters
        self.n_steps = self.cfg['diffusion']['steps']  # define total diffusion steps

        # make beta schedule and calculate the parameters used in denoising process.
        self.betas = make_beta_schedule(
            schedule=self.cfg['diffusion']['beta_schedule'], n_timesteps=self.n_steps,
            start=self.cfg['diffusion']['beta_start'], end=self.cfg['diffusion']['beta_end']).cuda()

        self.alphas = 1 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)
        self.pad_past = self.cfg["past_frames"]
        self.predictions_amount = rospy.get_param('~predictions_amount')

        rospy.loginfo(rospy.get_name() + " - initialized")

    def inference_callback(self, event):
        if len(self.active_keys) and self.model is not None and next(self.model.parameters()).is_cuda:
            # Run inference
            with self.lock:
                temp_active_keys = set(self.active_keys)
                if self.use_backpropagation:
                    [self.cache[key].backpropagate_trajectories(pad_past=self.pad_past *
                                                                         (self.skip_points + 1))
                     for key in temp_active_keys if self.cache[key].endpoints_count == 0]

                temp_raw_trajectories = [self.cache[key].raw_trajectories[-1::-1 * (self.skip_points + 1)][::-1]
                                         for key in temp_active_keys]
                temp_endpoints = [self.cache[key].endpoints_count // (self.skip_points + 1)
                                  for key in temp_active_keys]
                temp_headers = [self.cache[key].return_last_header() for key in temp_active_keys]

            inference_dataset = LEDNetDatasetInit(temp_raw_trajectories,
                                                  end_points=temp_endpoints,
                                                  pad_past=self.pad_past - 1,
                                                  pad_future=0,
                                                  inference_timer_duration=self.inference_timer_duration * (self.skip_points + 1)
                                                      )
            print('Dataset length:', len(inference_dataset))
            t0 = time.time()
            # make beta schedule and calculate the parameters used in denoising process.
            self.betas = make_beta_schedule(
                schedule=self.cfg['diffusion']['beta_schedule'], n_timesteps=self.n_steps,
                start=self.cfg['diffusion']['beta_start'], end=self.cfg['diffusion']['beta_end']).cuda()

            self.alphas = 1 - self.betas
            self.alphas_prod = torch.cumprod(self.alphas, 0)
            self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
            self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)
            print('Init betas time:', time.time() - t0)
            t0 = time.time()
            inference_result = lednet_iter(inference_dataset, self.model_initializer, self.model, self.betas, self.alphas, self.one_minus_alphas_bar_sqrt, n=self.predictions_amount)
            print('Inference time:', time.time() - t0)
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
    rospy.init_node('lednet_predictor', anonymous=True)

    sub = LEDNetSubscriber()
    sub.run()

