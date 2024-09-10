# Based on https://github.com/mengmengliu1998/GATraj

import os
import torch
import time
import torch.nn as nn
from torch.utils import data
import onnxruntime
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch.nn.functional as F
import ast

from scipy.spatial.distance import pdist, squareform
from GATraj.GATraj_parser import get_args
from GATraj.GATraj_laplace_decoder import *


def initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)  # initializing the lstm bias with zeros
        else:
            return


class LayerNorm(nn.Module):
    r"""
    Layer normalization.
    """

    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MLP_gate(nn.Module):
    def __init__(self, hidden_size, out_features=None):
        super(MLP_gate, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = F.sigmoid(hidden_states)
        return hidden_states


class MLP(nn.Module):
    def __init__(self, hidden_size, out_features=None):
        super(MLP, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states


class Temperal_Encoder(nn.Module):
    """Construct the sequence model"""

    def __init__(self, args):
        super(Temperal_Encoder, self).__init__()
        self.args = args
        self.hidden_size = self.args.hidden_size
        if args.input_mix:
            self.conv1d = nn.Conv1d(4, self.hidden_size, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1d = nn.Conv1d(2, self.hidden_size, kernel_size=3, stride=1, padding=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.args.x_encoder_head, \
                                                   dim_feedforward=self.hidden_size, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.args.x_encoder_layers)
        self.mlp1 = MLP(self.hidden_size)
        self.mlp = MLP(self.hidden_size)
        self.lstm = nn.LSTM(input_size=self.hidden_size,
                            hidden_size=self.hidden_size,
                            num_layers=1,
                            bias=True,
                            batch_first=True,
                            dropout=0,
                            bidirectional=False)
        initialize_weights(self.conv1d.modules())

    def forward(self, x):
        self.x_dense = self.conv1d(x).permute(0, 2, 1)  # [N, H, dim]
        self.x_dense = self.mlp1(self.x_dense) + self.x_dense  # [N, H, dim]
        self.x_dense_in = self.transformer_encoder(self.x_dense) + self.x_dense  # [N, H, D]
        output, (hn, cn) = self.lstm(self.x_dense_in)
        self.x_state, cn = hn.squeeze(0), cn.squeeze(0)  # [N, D]
        self.x_endoced = self.mlp(self.x_state) + self.x_state  # [N, D]
        return self.x_endoced, self.x_state, cn


class Global_interaction(nn.Module):
    def __init__(self, args):
        super(Global_interaction, self).__init__()
        self.args = args
        self.hidden_size = self.args.hidden_size
        # Motion gate
        self.ngate = MLP_gate(self.hidden_size * 3, self.hidden_size)  # sigmoid
        # Relative spatial embedding layer
        self.relativeLayer = MLP(2, self.hidden_size)
        # Attention
        self.WAr = MLP(self.hidden_size * 3, 1)
        self.weight = MLP(self.hidden_size)

    def forward(self, corr_index, nei_index, nei_num, hidden_state, cn):
        '''
        States Refinement process
        Params:
            corr_index: relative coords of each pedestrian pair [N, N, D]
            nei_index: neighbor exsists flag [N, N]
            nei_num: neighbor number [N]
            hidden_state: output states of GRU [N, D]
        Return:
            Refined states
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self_h = hidden_state
        self.N = corr_index.shape[0]
        self.D = self.hidden_size
        nei_inputs = self_h.repeat(self.N, 1)  # [N, N, D]
        nei_index_t = nei_index.view(self.N * self.N)  # [N*N]
        corr_t = corr_index.contiguous().view((self.N * self.N, -1))  # [N*N, D]
        if corr_t[nei_index_t > 0].shape[0] == 0:
            # Ignore when no neighbor in this batch
            return hidden_state, cn
        r_t = self.relativeLayer(corr_t[nei_index_t > 0])  # [N*N, D]
        inputs_part = nei_inputs[nei_index_t > 0].float()
        hi_t = nei_inputs.view((self.N, self.N, self.hidden_size)).permute(1, 0, 2).contiguous().view(-1,
                                                                                                      self.hidden_size)  # [N*N, D]
        tmp = torch.cat((r_t, hi_t[nei_index_t > 0], nei_inputs[nei_index_t > 0]), 1)  # [N*N, 3*D]
        # Motion Gate
        nGate = self.ngate(tmp).float()  # [N*N, D]
        # Attention
        Pos_t = torch.full((self.N * self.N, 1), 0, device=device).view(-1).float()
        tt = self.WAr(torch.cat((r_t, hi_t[nei_index_t > 0], nei_inputs[nei_index_t > 0]), 1)).view(
            -1).float()  # [N*N, 1]
        # have bug if there's any zero value in tt
        Pos_t[nei_index_t > 0] = tt
        Pos = Pos_t.view((self.N, self.N))
        Pos[Pos == 0] = -10000
        Pos = torch.softmax(Pos, dim=1)
        Pos_t = Pos.view(-1)
        # Message Passing
        H = torch.full((self.N * self.N, self.D), 0, device=device).float()
        H[nei_index_t > 0] = inputs_part * nGate
        H[nei_index_t > 0] = H[nei_index_t > 0] * Pos_t[nei_index_t > 0].repeat(self.D, 1).transpose(0, 1)
        H = H.view(self.N, self.N, -1)  # [N, N, D]
        H_sum = self.weight(torch.sum(H, 1))  # [N, D]
        # Update hidden states
        C = H_sum + cn  # [N, D]
        H = hidden_state + F.tanh(C)  # [N, D]
        return H, C


class Laplacian_Decoder(nn.Module):

    def __init__(self, args):
        super(Laplacian_Decoder, self).__init__()
        self.args = args
        if args.mlp_decoder:
            self._decoder = MLPDecoder(args)
        else:
            self._decoder = GRUDecoder(args)

    def forward(self, x_encode, hidden_state, cn, epoch):
        mdn_out = self._decoder(x_encode, hidden_state, cn)
        loc, scale, pi = mdn_out  # [F, N, H, 2], [F, N, H, 2], [N, F]
        return (loc, scale, pi)


class GATraj(nn.Module):
    def __init__(self, args):
        super(GATraj, self).__init__()
        self.args = args
        self.Temperal_Encoder=Temperal_Encoder(self.args)
        self.Laplacian_Decoder=Laplacian_Decoder(self.args)
        if self.args.SR:
            message_passing = []
            for i in range(self.args.pass_time):
                message_passing.append(Global_interaction(args))
            self.Global_interaction = nn.ModuleList(message_passing)

    def forward(self, batch_abs_gt, batch_norm_gt, nei_list_batch, nei_num_batch, batch_split, epoch=0, iftest=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # batch_abs_gt, batch_norm_gt, nei_list_batch, nei_num_batch, batch_split = inputs # #[H, N, 2], [H, N, 2], [B, H, N, N], [N, H], [B, 2]
        self.batch_norm_gt = batch_norm_gt
        if self.args.input_offset:
            train_x = batch_norm_gt[1:self.args.obs_length, :, :] - batch_norm_gt[:self.args.obs_length-1, :, :] #[H, N, 2]
        elif self.args.input_mix:
            offset = batch_norm_gt[1:self.args.obs_length, :, :] - batch_norm_gt[:self.args.obs_length-1, :, :] #[H, N, 2]
            position = batch_norm_gt[:self.args.obs_length, :, :] #[H, N, 2]
            pad_offset = torch.zeros_like(position).to(device)
            pad_offset[1:, :, :] = offset
            train_x = torch.cat((position, pad_offset), dim=2)
        elif self.args.input_position:
            train_x = batch_norm_gt[:self.args.obs_length, :, :] #[H, N, 2]
        train_x = train_x.permute(1, 2, 0) #[N, 2, H]
        # train_y = batch_norm_gt[self.args.obs_length:, :, :].permute(1, 2, 0) #[N, 2, H]
        self.pre_obs=batch_norm_gt[1:self.args.obs_length]
        self.x_encoded_dense, self.hidden_state_unsplited, cn=self.Temperal_Encoder.forward(train_x)  #[N, D], [N, D]
        # self.hidden_state_global = torch.ones_like(self.hidden_state_unsplited, device=device)
        # cn_global = torch.ones_like(cn, device=device)
        # if self.args.SR:
        #     for b in range(len(nei_list_batch)):
        #         left, right = batch_split[b][0], batch_split[b][1]
        #         element_states = self.hidden_state_unsplited[left: right] #[N, D]
        #         cn_state = cn[left: right] #[N, D]
        #         if element_states.shape[0] != 1:
        #             corr = batch_abs_gt[self.args.obs_length-1, left: right, :2].repeat(element_states.shape[0], 1, 1) #[N, N, D]
        #             corr_index = corr.transpose(0,1)-corr  #[N, N, D]
        #             nei_num = nei_num_batch[left:right, self.args.obs_length-1] #[N]
        #             nei_index = torch.tensor(nei_list_batch[b][self.args.obs_length-1], device=device) #[N, N]
        #             for i in range(self.args.pass_time):
        #                 element_states, cn_state = self.Global_interaction[i](corr_index, nei_index, nei_num, element_states, cn_state)
        #             self.hidden_state_global[left: right] = element_states
        #             cn_global[left: right] = cn_state
        #         else:
        #             self.hidden_state_global[left: right] = element_states
        #             cn_global[left: right] = cn_state
        # else:
        self.hidden_state_global = self.hidden_state_unsplited
        cn_global = cn
        mdn_out = self.Laplacian_Decoder.forward(self.x_encoded_dense, self.hidden_state_global, cn_global, epoch)

        return mdn_out




def gatraj_iter(dataset, model, device, args, n):
    with torch.inference_mode():
        batch_by_batch_guesses = []
        batch_by_batch_guesses.append([])

        inputs_gt, batch_split, nei_lists = dataset.get_all()
        inputs_gt = tuple([torch.Tensor(i) if i is not None else None for i in inputs_gt ])
        inputs_gt = tuple([i.cuda() if i is not None else None for i in inputs_gt])
        batch_abs_gt, batch_norm_gt, shift_value_gt, seq_list_gt, nei_num = inputs_gt
        out_mu, out_sigma, out_pi = model.forward(batch_abs_gt, batch_norm_gt, nei_lists, nei_num, batch_split)
        pred_traj = out_mu
        pred_traj = pred_traj.cpu().numpy()
        out_pi = out_pi.cpu().numpy()
        argsort_pi = np.argsort(out_pi, axis=1)
        for j in range(n):
            candidate_traj = np.zeros((pred_traj.shape[1], pred_traj.shape[2], pred_traj.shape[3]))
            for i in range(pred_traj.shape[1]):
                candidate_traj[i] = pred_traj[np.argwhere(argsort_pi[i] == j)[0][0], i]
            candidate_traj = np.swapaxes(candidate_traj, 0, 1)
            candidate_traj = candidate_traj + dataset.shift + dataset.initial_shift
            candidate_traj = np.swapaxes(candidate_traj, 0, 1)
            batch_by_batch_guesses[len(batch_by_batch_guesses) - 1].append(candidate_traj)

    true_guesses = [[] for _ in range(n)]
    for batch_guess in batch_by_batch_guesses:
        for i in range(n):
            true_guesses[i].extend(batch_guess[i])

    return true_guesses


class GATrajDatasetInit(data.Dataset):
    def __init__(self, detected_object_trajs, end_points, pad_past=7, pad_future=0, dist_thresh=10, proximity=False):
        self.proximity = proximity

        self.traj = detected_object_trajs
        self.traj = np.swapaxes(self.traj, 0, 1)
        self.initial_shift = np.min(self.traj)

        self.traj_flat = np.copy(self.traj)

        self.traj = self.traj - self.initial_shift
        self.shift = self.traj[-1:, :, :]
        self.traj_norm = self.traj - self.shift
        if self.proximity:
            self.batch_split = np.array([[0, self.traj.shape[1]]])
            self.masks_ped = np.ones((1, self.traj.shape[0], self.traj.shape[1]))
            self.masks = np.zeros((1, self.traj.shape[0], self.traj.shape[1], self.traj.shape[1]))
            for i in range(self.traj.shape[0]):
                distances = squareform(pdist(self.traj[i, :]))
                mask = np.where(distances < dist_thresh, 1.0, 0.0)
                np.fill_diagonal(mask, 0)
                self.masks[0, i] = mask
            self.masks_num = np.zeros((self.traj.shape[0], self.traj.shape[1]))
            for i in range(self.masks.shape[1]):
                self.masks_num[:, i] = np.sum(self.masks[0, :, i, :], axis=1)

    def __len__(self):
        return len(self.traj)

    def __getitem__(self, idx):
        return self.traj[idx], self.masks[idx]

    def get_all(self):
        if self.proximity:
            return (self.traj, self.traj_norm, self.shift, self.masks_ped, self.masks_num), self.batch_split, self.masks
        else:
            return (self.traj, self.traj_norm, self.shift, None, None), None, None
