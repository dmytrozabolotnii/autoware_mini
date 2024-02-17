#!/usr/bin/env python

import rospy
import numpy as np
import torch
import os
import os.path as osp
from common_utils import color_points, calculate_danger_value
from Net_sub import NetSubscriber
from SGNet_utils import SGNet, parse_sgnet_args, sgnet_iter, SGNetDatasetInit


class SGNetSubscriber(NetSubscriber):
    def __init__(self, args):
        super().__init__()
        rospy.loginfo(self.__class__.__name__ + " - Initializing")
        # this_dir = osp.dirname(__file__)
        # model_name = args.model
        # save_dir = osp.join(this_dir, 'checkpoints', args.dataset, model_name, str(args.dropout), str(args.seed))
        # if not osp.isdir(save_dir):
        #     os.makedirs(save_dir)

        # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # utl.set_seed(int(args.seed))
        self.model = SGNet(args)
        self.inference_timer_duration = 0.5
        self.pad_past = 8
        self.future_horizon = 12
        self.predictions_amount = 1
        self.model = self.model.double().to(self.device)
        print(rospy.get_param('~data_path_prediction') + args.checkpoint)
        if osp.isfile(rospy.get_param('~data_path_prediction') + args.checkpoint):
            self.checkpoint = torch.load(rospy.get_param('~data_path_prediction') + args.checkpoint, map_location=self.device)
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
        # criterion = rmse_loss().to(device)

        # test_gen = utl.build_data_loader(args, 'test', batch_size=1)
        # print("Number of test samples:", test_gen.__len__())

        # test
        # test_loss, ADE_08, FDE_08, ADE_12, FDE_12 = test(model, test_gen, criterion, device)
        self.inference_timer = rospy.Timer(rospy.Duration(self.inference_timer_duration), self.inference_callback)
        print("Initializing the instance!")

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
            # print(len(inference_result))
            # Update history of inferences
            for j, _id in enumerate(temp_active_keys):
                for i in range(len(inference_result)):
                    self.all_predictions_history[_id][len(self.all_predictions_history[_id]) - 1]\
                        .append(inference_result[i][j])
                self.all_predictions_history[_id].append([])

            # print('Timestamp:', event.current_real, 'Results:', inference_result, 'Endpoints:', self.all_endpoints)
            # Update trajectory and process points
            if self.self_traj_exists:
                trajectory_length = self.velocity * self.inference_timer_duration * self.future_horizon
                self.self_traj = self.self_new_traj.copy()
                inference_colors = (list(map(color_points, inference_result, [self.self_traj] * len(inference_result),
                                             [trajectory_length] * len(inference_result))))
                # print('Inference_colors:', inference_colors)
                endpoint_colors = color_points(inference_dataset.traj_flat[:, self.pad_past - 1],
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
                self.publish_markers(inference_dataset.traj_flat[:, self.pad_past - 1],
                                     inference_result, inference_colors, endpoint_colors, avg_danger_values, self.predictions_amount)
            self.move_endpoints()
            self.publish_predicted_objects()



def main(args):
    # create a subscriber instance
    rospy.init_node('SGNetPredictor', anonymous=True)
    sub = SGNetSubscriber(args)

    # initializing the subscriber node
    rospy.on_shutdown(sub.on_shutdown)
    rospy.spin()


if __name__ == '__main__':
    try:
        print('Cuda:', torch.cuda.is_available())
        main(parse_sgnet_args())
    except rospy.ROSInterruptException:
        pass
