# Based on https://github.com/ChuhuaW/SGNet.pytorch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils import data
from typing import List

def parse_base_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='SGNet/SGNet_eth_checkpoint.pth', type=str)
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--weight_decay', default=5e-04, type=float)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--phases', default=['train', 'test'], type=list)
    parser.add_argument('--shuffle', default=True, type=bool)

    return parser


def parse_sgnet_args():
    parser = parse_base_args()
    parser.add_argument('--dataset', default='ETH', type=str)
    parser.add_argument('--lr', default=5e-04, type=float) # ETH 0.0005，HOTEL 0.0001, UNIV 0.0001, ZARA1 0.0001, ZARA2 0.0001
    parser.add_argument('--eth_root', default='data/ETHUCY', type=str)
    parser.add_argument('--model', default='SGNet', type=str)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--enc_steps', default=8, type=int)
    parser.add_argument('--dec_steps', default=12, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--nu', default=0.0, type=float)
    parser.add_argument('--sigma', default=1.0, type=float)
    parser.add_argument('--ETH_CONFIG', default='SGNet_ETH_UCY.json', type=str)
    parser.add_argument('--augment', default=False, type=bool)
    parser.add_argument('--DEC_WITH_Z', default=True, type=bool)
    parser.add_argument('--LATENT_DIM', default=32, type=int)
    parser.add_argument('--pred_dim', default=2, type=int)
    parser.add_argument('--input_dim', default=6, type=int)
    parser.add_argument('--K', default=20, type=int)

    args, _ = parser.parse_known_args()
    return args


class ETHUCYFeatureExtractor(nn.Module):

    def __init__(self, args):
        super(ETHUCYFeatureExtractor, self).__init__()
        self.embbed_size = args.hidden_size
        self.embed = nn.Sequential(nn.Linear(6, self.embbed_size),
                                        nn.ReLU())


    def forward(self, inputs):
        box_input = inputs

        embedded_box_input= self.embed(box_input)

        return embedded_box_input


class SGNet(nn.Module):
    def __init__(self, args):
        super(SGNet, self).__init__()

        self.hidden_size = args.hidden_size
        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps
        self.dataset = args.dataset
        self.dropout = args.dropout
        self.feature_extractor = ETHUCYFeatureExtractor(args)
        if self.dataset in ['JAAD', 'PIE']:
            self.pred_dim = 4
            self.regressor = nn.Sequential(nn.Linear(self.hidden_size,
                                                     self.pred_dim),
                                           nn.Tanh())
            self.flow_enc_cell = nn.GRUCell(self.hidden_size * 2, self.hidden_size)
        elif self.dataset in ['ETH', 'HOTEL', 'UNIV', 'ZARA1', 'ZARA2']:
            self.pred_dim = 2
            self.regressor = nn.Sequential(nn.Linear(self.hidden_size,
                                                     self.pred_dim))

        self.enc_goal_attn = nn.Sequential(nn.Linear(self.hidden_size // 4,
                                                     1),
                                           nn.ReLU(inplace=True))
        self.dec_goal_attn = nn.Sequential(nn.Linear(self.hidden_size // 4,
                                                     1),
                                           nn.ReLU(inplace=True))

        self.enc_to_goal_hidden = nn.Sequential(nn.Linear(self.hidden_size,
                                                          self.hidden_size // 4),
                                                nn.ReLU(inplace=True))
        self.enc_to_dec_hidden = nn.Sequential(nn.Linear(self.hidden_size,
                                                         self.hidden_size),
                                               nn.ReLU(inplace=True))

        self.goal_hidden_to_input = nn.Sequential(nn.Linear(self.hidden_size // 4,
                                                            self.hidden_size // 4),
                                                  nn.ReLU(inplace=True))
        self.dec_hidden_to_input = nn.Sequential(nn.Linear(self.hidden_size,
                                                           self.hidden_size),
                                                 nn.ReLU(inplace=True))
        self.goal_hidden_to_traj = nn.Sequential(nn.Linear(self.hidden_size // 4,
                                                           self.hidden_size),
                                                 nn.ReLU(inplace=True))
        self.goal_to_enc = nn.Sequential(nn.Linear(self.hidden_size // 4,
                                                   self.hidden_size // 4),
                                         nn.ReLU(inplace=True))
        self.goal_to_dec = nn.Sequential(nn.Linear(self.hidden_size // 4,
                                                   self.hidden_size // 4),
                                         nn.ReLU(inplace=True))
        self.enc_drop = nn.Dropout(self.dropout)
        self.goal_drop = nn.Dropout(self.dropout)
        self.dec_drop = nn.Dropout(self.dropout)

        self.traj_enc_cell = nn.GRUCell(self.hidden_size + self.hidden_size // 4, self.hidden_size)
        self.goal_cell = nn.GRUCell(self.hidden_size // 4, self.hidden_size // 4)
        self.dec_cell = nn.GRUCell(self.hidden_size + self.hidden_size // 4, self.hidden_size)

    def SGE(self, goal_hidden):
        goal_input = goal_hidden.new_zeros((goal_hidden.size(0), self.hidden_size // 4))
        goal_traj = goal_hidden.new_zeros(goal_hidden.size(0), self.dec_steps, self.pred_dim)
        goal_list = []
        for dec_step in range(self.dec_steps):
            goal_hidden = self.goal_cell(self.goal_drop(goal_input), goal_hidden)
            goal_input = self.goal_hidden_to_input(goal_hidden)
            goal_list.append(goal_hidden)
            goal_traj_hidden = self.goal_hidden_to_traj(goal_hidden)
            # regress goal traj for loss
            goal_traj[:, dec_step, :] = self.regressor(goal_traj_hidden)
        # get goal for decoder and encoder
        goal_for_dec = [self.goal_to_dec(goal) for goal in goal_list]
        goal_for_enc = torch.stack([self.goal_to_enc(goal) for goal in goal_list], dim=1)
        enc_attn = self.enc_goal_attn(torch.tanh(goal_for_enc)).squeeze(-1)
        enc_attn = F.softmax(enc_attn, dim=1).unsqueeze(1)
        goal_for_enc = torch.bmm(enc_attn, goal_for_enc).squeeze(1)
        return goal_for_dec, goal_for_enc, goal_traj

    def decoder(self, dec_hidden, goal_for_dec: List[torch.Tensor]):
        # initial trajectory tensor
        dec_traj = dec_hidden.new_zeros(dec_hidden.size(0), self.dec_steps, self.pred_dim)
        for dec_step in range(self.dec_steps):
            goal_dec_input = dec_hidden.new_zeros(dec_hidden.size(0), self.dec_steps, self.hidden_size // 4)
            goal_dec_input_temp = torch.stack(goal_for_dec[dec_step:], dim=1)
            goal_dec_input[:, dec_step:, :] = goal_dec_input_temp
            dec_attn = self.dec_goal_attn(torch.tanh(goal_dec_input)).squeeze(-1)
            dec_attn = F.softmax(dec_attn, dim=1).unsqueeze(1)
            goal_dec_input = torch.bmm(dec_attn, goal_dec_input).squeeze(
                1)  # .view(goal_hidden.size(0), self.dec_steps, self.hidden_size//4).sum(1)

            dec_dec_input = self.dec_hidden_to_input(dec_hidden)
            dec_input = self.dec_drop(torch.cat((goal_dec_input, dec_dec_input), dim=-1))
            dec_hidden = self.dec_cell(dec_input, dec_hidden)
            # regress dec traj for loss
            dec_traj[:, dec_step, :] = self.regressor(dec_hidden)
        return dec_traj

    def encoder(self, traj_input):
        # initial output tensor
        all_goal_traj = traj_input.new_zeros(traj_input.size(0), self.enc_steps, self.dec_steps, self.pred_dim)
        all_dec_traj = traj_input.new_zeros(traj_input.size(0), self.enc_steps, self.dec_steps, self.pred_dim)
        # initial encoder goal with zeros
        goal_for_enc = traj_input.new_zeros((traj_input.size(0), self.hidden_size // 4))
        # initial encoder hidden with zeros
        traj_enc_hidden = traj_input.new_zeros((traj_input.size(0), self.hidden_size))
        for enc_step in range(0, self.enc_steps):

            traj_enc_hidden = self.traj_enc_cell(
                self.enc_drop(torch.cat((traj_input[:, enc_step, :], goal_for_enc), 1)), traj_enc_hidden)
            # if self.dataset in ['JAAD', 'PIE', 'ETH', 'HOTEL', 'UNIV', 'ZARA1', 'ZARA2']:
            #     enc_hidden = traj_enc_hidden
            enc_hidden = traj_enc_hidden
            # generate hidden states for goal and decoder
            goal_hidden = self.enc_to_goal_hidden(enc_hidden)
            dec_hidden = self.enc_to_dec_hidden(enc_hidden)

            goal_for_dec, goal_for_enc, goal_traj = self.SGE(goal_hidden)
            dec_traj = self.decoder(dec_hidden, goal_for_dec)

            # output
            all_goal_traj[:, enc_step, :, :] = goal_traj
            all_dec_traj[:, enc_step, :, :] = dec_traj

        return all_goal_traj, all_dec_traj

    def forward(self, inputs):
        if self.dataset in ['JAAD', 'PIE']:
            traj_input = self.feature_extractor(inputs)
            all_goal_traj, all_dec_traj = self.encoder(traj_input)
            return all_goal_traj, all_dec_traj
        elif self.dataset in ['ETH', 'HOTEL', 'UNIV', 'ZARA1', 'ZARA2']:
            traj_input_temp = self.feature_extractor(inputs[:, 0:, :])
            traj_input = traj_input_temp.new_zeros((inputs.size(0), inputs.size(1), traj_input_temp.size(-1)))
            traj_input[:, 0:, :] = traj_input_temp
            all_goal_traj, all_dec_traj = self.encoder(traj_input)
            return all_goal_traj, all_dec_traj


def sgnet_iter(dataset, model, device, n=5):

    dataloader = data.DataLoader(
        dataset, batch_size=len(dataset), shuffle=False, num_workers=0)

    model.eval()
    batch_by_batch_guesses = []
    with torch.set_grad_enabled(False):
        for batch_idx, traj in enumerate(dataloader):
            batch_by_batch_guesses.append([])
            traj = torch.DoubleTensor(traj).to(device)

            shift = traj[:, -1:, :2].to('cpu').numpy()
            input_normalized = torch.cat([traj[:, :, :2] - traj[:, -1:, :2], traj[:, :, 2:]], dim=2)
            # input_normalized = torch.clone(traj).to(device)
            # reduce = torch.max(torch.abs(torch.min(input_normalized[:, :, :2])), torch.abs(torch.max(input_normalized[:, :, :2])))
            # for i in range(2):
            #     input_normalized[:, :, i] = (input_normalized[:, :, i] - torch.mean(input_normalized[:, :, i])) / reduce

            input_traj_torch = input_normalized

            for j in range(n):
                all_goal_traj, all_dec_traj = model(input_traj_torch)
                all_goal_traj_np = all_goal_traj.to('cpu').numpy()
                all_dec_traj_np = all_dec_traj.to('cpu').numpy()
                dest_path = all_dec_traj_np[:, -1, :, :]
                dest_path = dest_path + shift

                batch_by_batch_guesses[len(batch_by_batch_guesses) - 1].append(dest_path)

    true_guesses = [[] for _ in range(n)]
    for batch_guess in batch_by_batch_guesses:
        for i in range(n):
            true_guesses[i].extend(batch_guess[i])

    return true_guesses


class SGNetDatasetInit(data.Dataset):
    def __init__(self, detected_object_trajs, velocity_objects, acceleration_objects, end_points, pad_past=8, pad_future=0, inference_timer_duration=0.5):
        # self.traj_flat = np.array([np.pad(np.array(traj), ((pad_past, pad_future), (0, 0)),
        #                              mode='edge')[end_points[i]:end_points[i] + pad_past + pad_future + 1] for i, traj in enumerate(detected_object_trajs)])
        self.traj_flat = detected_object_trajs
        traj = np.copy(self.traj_flat)
        # self.velocitytraj = np.array([np.pad(np.array(traj), ((pad_past, pad_future), (0, 0)),
        #                              mode='edge')[end_points[i]:end_points[i] + pad_past + pad_future + 1] for i, traj in enumerate(velocity_objects)]) / inference_timer_duration
        # self.acceltraj = np.array([np.pad(np.array(traj), ((pad_past, pad_future), (0, 0)),
        #                              mode='edge')[end_points[i]:end_points[i] + pad_past + pad_future + 1] for i, traj in enumerate(acceleration_objects)]) / inference_timer_duration

        self.velocitytraj = velocity_objects
        self.acceltraj = acceleration_objects

        self.traj = np.concatenate((traj, self.velocitytraj,
                                    self.acceltraj), axis=2)


    def __len__(self):
        return len(self.traj)

    def __getitem__(self, idx):
        return self.traj[idx]
