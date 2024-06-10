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

# class SoftTargetCrossEntropyLoss(nn.Module):
#
#     def __init__(self, reduction: str = 'mean') -> None:
#         super(SoftTargetCrossEntropyLoss, self).__init__()
#         self.reduction = reduction
#
#     def forward(self,
#                 pred: torch.Tensor,
#                 target: torch.Tensor) -> torch.Tensor:
#         cross_entropy = torch.sum(-target * F.log_softmax(pred, dim=-1), dim=-1)
#         if self.reduction == 'mean':
#             return cross_entropy.mean()
#         elif self.reduction == 'sum':
#             return cross_entropy.sum()
#         elif self.reduction == 'none':
#             return cross_entropy
#         else:
#             raise ValueError('{} is not a valid value for reduction'.format(self.reduction))
#
# class LaplaceNLLLoss(nn.Module):
#
#     def __init__(self,
#                  eps: float = 1e-6,
#                  reduction: str = 'mean') -> None:
#         super(LaplaceNLLLoss, self).__init__()
#         self.eps = eps
#         self.reduction = reduction
#
#     def forward(self,
#                 pred: torch.Tensor,
#                 target: torch.Tensor) -> torch.Tensor:
#         loc, scale = pred.chunk(2, dim=-1)
#         scale = scale.clone()
#         # print("scale",scale.shape,"loc",loc.shape)
#         with torch.no_grad():
#             scale.clamp_(min=self.eps)
#         nll = torch.log(2 * scale) + torch.abs(target - loc) / scale
#         # print("nll", nll.shape)
#         if self.reduction == 'mean':
#             return nll.mean()
#         elif self.reduction == 'sum':
#             return nll.sum()
#         elif self.reduction == 'none':
#             return nll
#         else:
#             raise ValueError('{} is not a valid value for reduction'.format(self.reduction))
#
# class GaussianNLLLoss(nn.Module):
#     """https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html
#     """
#     def __init__(self,
#                  eps: float = 1e-6,
#                  reduction: str = 'mean') -> None:
#         super(GaussianNLLLoss, self).__init__()
#         self.eps = eps
#         self.reduction = reduction
#
#     def forward(self,
#                 pred: torch.Tensor,
#                 target: torch.Tensor) -> torch.Tensor:
#         loc, scale = pred.chunk(2, dim=-1)
#         scale = scale.clone()
#         # print("scale",scale.shape,"loc",loc.shape)
#         with torch.no_grad():
#             scale.clamp_(min=self.eps)
#         nll = 0.5*(torch.log(scale**2) + torch.abs(target - loc)**2 / scale**2)
#         # print("nll", nll.shape)
#         if self.reduction == 'mean':
#             return nll.mean()
#         elif self.reduction == 'sum':
#             return nll.sum()
#         elif self.reduction == 'none':
#             return nll
#         else:
#             raise ValueError('{} is not a valid value for reduction'.format(self.reduction))

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
            # print("LSTM------",m.named_parameters())
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)  # initializing the lstm bias with zeros
        else:
            print(m, "************")


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
        # if self.args.ifGaussian:
        #     self.reg_loss = GaussianNLLLoss(reduction='mean')
        # else:
        #     self.reg_loss = LaplaceNLLLoss(reduction='mean')
        # self.cls_loss = SoftTargetCrossEntropyLoss(reduction='mean')

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
        # GATraj_loss, full_pre_tra = self.mdn_loss(train_y.permute(2, 0, 1), mdn_out, 1, iftest)  #[K, H, N, 2]

        return mdn_out

        # return GATraj_loss, full_pre_tra

    # def mdn_loss(self, y, y_prime, goal_gt, iftest):
    #     batch_size=y.shape[1]
    #     y = y.permute(1, 0, 2)  #[N, H, 2]
    #     # [F, N, H, 2], [F, N, H, 2], [N, F]
    #     out_mu, out_sigma, out_pi = y_prime
    #     y_hat = torch.cat((out_mu, out_sigma), dim=-1)
    #     reg_loss, cls_loss = 0, 0
    #     full_pre_tra = []
    #     l2_norm = (torch.norm(out_mu - y, p=2, dim=-1) ).sum(dim=-1)   # [F, N]
    #     best_mode = l2_norm.argmin(dim=0)
    #     y_hat_best = y_hat[best_mode, torch.arange(batch_size)]
    #     reg_loss += self.reg_loss(y_hat_best, y)
    #     soft_target = F.softmax(-l2_norm / self.args.pred_length, dim=0).t().detach() # [N, F]
    #     cls_loss += self.cls_loss(out_pi, soft_target)
    #     loss = reg_loss + cls_loss
    #     #best ADE
    #     sample_k = out_mu[best_mode, torch.arange(batch_size)].permute(1, 0, 2)  #[H, N, 2]
    #     full_pre_tra.append(torch.cat((self.pre_obs,sample_k), axis=0))
    #     # best FDE
    #     l2_norm_FDE = (torch.norm(out_mu[:,:,-1,:] - y[:,-1,:], p=2, dim=-1) )  # [F, N]
    #     best_mode = l2_norm_FDE.argmin(dim=0)
    #     sample_k = out_mu[best_mode, torch.arange(batch_size)].permute(1, 0, 2)  #[H, N, 2]
    #     full_pre_tra.append(torch.cat((self.pre_obs,sample_k), axis=0))
    #     return loss, full_pre_tra


def gatraj_iter(dataset, model, device, args, n):
    # model.eval()
    # input_names = ["batch_abs_gt", "batch_norm_gt", "nei_lists", "nei_num", "batch_split"]
    #
    # output_names = ["out_mu", "out_sigma", "out_pi"]
    #
    # dynamic_axes_dict = {
    #     "batch_abs_gt": {
    #         1: "pedestrians",
    #     },
    #     "batch_norm_gt": {
    #         1: "pedestrians",
    #     },
    #     "nei_lists": {
    #         2: "pedestrians",
    #         3: "pedestrians"
    #     },
    #     "nei_num": {
    #         1: "pedestrians"
    #     },
    #     "out_mu": {
    #         1: "pedestrians"
    #     },
    #     "out_sigma": {
    #         1: "pedestrians"
    #     },
    #     "out_pi": {
    #         1: "pedestrians"
    #     }
    # }
    with torch.inference_mode():
        batch_by_batch_guesses = []
        batch_by_batch_guesses.append([])

        inputs_gt, batch_split, nei_lists = dataset.get_all()
        inputs_gt = tuple([torch.Tensor(i) if i is not None else None for i in inputs_gt ])
        inputs_gt = tuple([i.cuda() if i is not None else None for i in inputs_gt])
        # batch_split_torch = torch.IntTensor(batch_split).cuda()
        # nei_lists_torch = torch.IntTensor(nei_lists).cuda()
        batch_abs_gt, batch_norm_gt, shift_value_gt, seq_list_gt, nei_num = inputs_gt
        # inputs_fw = batch_abs_gt, batch_norm_gt, nei_lists, nei_num, batch_split  # [H, N, 2], [H, N, 2], [B, H, N, N], [N, H]
        # inputs_fw_dummy = batch_abs_gt, batch_norm_gt, nei_lists_torch, nei_num, batch_split_torch
        # session = None
        # if not os.path.isfile("GATraj/gatraj_1000.onnx"):
        #     torch.onnx.export(model,
        #                       inputs_fw_dummy,
        #                       "GATraj/gatraj_1000.onnx",
        #                       verbose=True,
        #                       input_names=input_names,
        #                       output_names=output_names,
        #                       dynamic_axes=dynamic_axes_dict,
        #                       opset_version=12,
        #                       export_params=True,
        #                       )
        # else:
        #     session = onnxruntime.InferenceSession('GATraj/gatraj_1000.onnx', providers=['CUDAExecutionProvider'])
        # t0 = time.time()
        # if session is not None:
        #     out_mu_session = session.run(['out_mu'], {
        #         "batch_abs_gt": batch_abs_gt,
        #         "batch_norm_gt": batch_norm_gt,
        #         "nei_lists": nei_lists,
        #         "nei_num": nei_num,
        #         "batch_split": batch_split
        #     })
        #     print(out_mu_session)
        # print('Session forward time', time.time() - t0)
        t0 = time.time()
        out_mu, out_sigma, out_pi = model.forward(batch_abs_gt, batch_norm_gt, nei_lists, nei_num, batch_split)
        # print('Model forward time', time.time() - t0)
        # pred_traj = torch.cat((out_mu, out_sigma), dim=-1)
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

        self.traj = np.array([np.pad(np.array(traj), ((pad_past, pad_future), (0, 0)),
                                     mode='edge')[end_points[i]:(end_points[i] + pad_past + pad_future + 1)] for i, traj in enumerate(detected_object_trajs)])
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

# class DataLoader_bytrajec2():
#     def __init__(self, args,is_gt=True):
#         self.miss=0
#         self.args=args
#         self.is_gt=is_gt
#         self.num_tra = 0
#         if self.args.dataset=='eth5':
#
#             self.data_dirs = ['./data/eth/univ', './data/eth/hotel',
#                               './data/ucy/zara/zara01', './data/ucy/zara/zara02',
#                               './data/ucy/univ/students001','data/ucy/univ/students003',
#                               './data/ucy/univ/uni_examples','./data/ucy/zara/zara03']
#
#             # Data directory where the pre-processed pickle file resides
#             self.data_dir = './data'
#             skip=[6,10,10,10,10,10,10,10]
#
#             if args.ifvalid:
#                 self.val_fraction = args.val_fraction
#             else:
#                 self.val_fraction=0
#
#             train_set=[i for i in range(len(self.data_dirs))]
#             if args.test_set==4 or args.test_set==5:
#                 self.test_set=[4,5]
#             else:
#                 self.test_set=[self.args.test_set]
#
#             for x in self.test_set:
#                 train_set.remove(x)
#             self.train_dir=[self.data_dirs[x] for x in train_set]
#             self.test_dir = [self.data_dirs[x] for x in self.test_set]
#             self.trainskip=[skip[x] for x in train_set]
#             self.testskip=[skip[x] for x in self.test_set]
#
#         self.train_data_file = os.path.join(self.args.save_dir,"train_trajectories.cpkl")
#         self.test_data_file = os.path.join(self.args.save_dir, "test_trajectories.cpkl")
#         self.train_batch_cache = os.path.join(self.args.save_dir,"train_batch_cache.cpkl")
#         self.test_batch_cache = os.path.join(self.args.save_dir, "test_batch_cache.cpkl")
#
#         print("Creating pre-processed data from raw data.")
#         self.traject_preprocess('train')
#         self.traject_preprocess('test')
#         print("Done.")
#
#         # Load the processed data from the pickle file
#         print("Preparing data batches.")
#         if not(os.path.exists(self.train_batch_cache)):
#             self.frameped_dict, self.pedtraject_dict=self.load_dict(self.train_data_file)
#             self.dataPreprocess('train')
#             print("self.num_tra", self.num_tra)
#             self.num_tra=0
#         if not(os.path.exists(self.test_batch_cache)):
#             self.test_frameped_dict, self.test_pedtraject_dict = self.load_dict(self.test_data_file)
#             self.dataPreprocess('test')
#             print("self.num_tra", self.num_tra)
#         self.trainbatch, self.trainbatchnums, \
#         self.valbatch, self.valbatchnums=self.load_cache(self.train_batch_cache)
#         self.testbatch, self.testbatchnums, _, _ = self.load_cache(self.test_batch_cache)
#         print("Done.")
#
#         print('Total number of training batches:', self.trainbatchnums)
#         print('Total number of validation batches:', self.valbatchnums)
#         print('Total number of test batches:', self.testbatchnums)
#
#
#     def traject_preprocess(self,setname):
#         '''
#         data_dirs : List of directories where raw data resides
#         data_file : The file into which all the pre-processed data needs to be stored
#         '''
#         if setname=='train':
#             data_dirs=self.train_dir
#             data_file=self.train_data_file
#         else:
#             data_dirs=self.test_dir
#             data_file=self.test_data_file
#         all_frame_data = []
#         valid_frame_data = []
#         numFrame_data = []
#
#         Pedlist_data=[]
#         frameped_dict=[]#peds id contained in a certain frame
#         pedtrajec_dict=[]#trajectories of a certain ped
#         # For each dataset
#         for seti,directory in enumerate(data_dirs):
#
#             file_path = os.path.join(directory, 'true_pos_.csv')
#             # Load the data from the csv file
#             data = np.genfromtxt(file_path, delimiter=',')
#             # Frame IDs of the frames in the current dataset
#             Pedlist = np.unique(data[1, :]).tolist()
#             numPeds = len(Pedlist)
#             # Add the list of frameIDs to the frameList_data
#             Pedlist_data.append(Pedlist)
#             # Initialize the list of numpy arrays for the current dataset
#             all_frame_data.append([])
#             # Initialize the list of numpy arrays for the current dataset
#             valid_frame_data.append([])
#             numFrame_data.append([])
#             frameped_dict.append({})
#             pedtrajec_dict.append({})
#
#             for ind, pedi in enumerate(Pedlist):
#                 if ind%100==0:
#                     print(ind,len(Pedlist))
#                 # Extract trajectories of one person
#                 FrameContainPed = data[:, data[1, :] == pedi]
#                 # Extract peds list
#                 FrameList = FrameContainPed[0, :].tolist()
#                 if len(FrameList)<2:
#                     continue
#                 # Add number of frames of this trajectory
#                 numFrame_data[seti].append(len(FrameList))
#                 # Initialize the row of the numpy array
#                 Trajectories = []
#                 # For each ped in the current frame
#                 for fi,frame in enumerate(FrameList):
#                     # Extract their x and y positions
#                     current_x = FrameContainPed[3, FrameContainPed[0, :] == frame][0]
#                     current_y = FrameContainPed[2, FrameContainPed[0, :] == frame][0]
#                     # Add their pedID, x, y to the row of the numpy array
#                     Trajectories.append([int(frame),current_x, current_y])
#                     if int(frame) not in frameped_dict[seti]:
#                         frameped_dict[seti][int(frame)]=[]
#                     frameped_dict[seti][int(frame)].append(pedi)
#                 pedtrajec_dict[seti][pedi]=np.array(Trajectories)
#
#         with open(data_file, "wb") as f:
#             pickle.dump((frameped_dict,pedtrajec_dict), f, protocol=2)
#
#     def load_dict(self,data_file):
#         f = open(data_file, 'rb')
#         raw_data = pickle.load(f)
#         f.close()
#
#         frameped_dict=raw_data[0]
#         pedtraject_dict=raw_data[1]
#
#         return frameped_dict,pedtraject_dict
#     def load_cache(self,data_file):
#         f = open(data_file, 'rb')
#         raw_data = pickle.load(f)
#         f.close()
#         return raw_data
#     def dataPreprocess(self,setname):
#         '''
#         Function to load the pre-processed data into the DataLoader object
#         '''
#         if setname=='train':
#             val_fraction=self.args.val_fraction
#             frameped_dict=self.frameped_dict
#             pedtraject_dict=self.pedtraject_dict
#             cachefile=self.train_batch_cache
#
#         else:
#             val_fraction=0
#             frameped_dict=self.test_frameped_dict
#             pedtraject_dict=self.test_pedtraject_dict
#             cachefile = self.test_batch_cache
#
#         data_index=self.get_data_index(frameped_dict,setname)
#         val_index=data_index[:,:int(data_index.shape[1]*val_fraction)]
#         train_index = data_index[:,(int(data_index.shape[1] * val_fraction)+1):]
#
#         trainbatch=self.get_seq_from_index_balance(frameped_dict,pedtraject_dict,train_index,setname)
#         valbatch = self.get_seq_from_index_balance(frameped_dict,pedtraject_dict,val_index,setname)
#
#         trainbatchnums=len(trainbatch)
#         valbatchnums=len(valbatch)
#
#         f = open(cachefile, "wb")
#         pickle.dump(( trainbatch, trainbatchnums, valbatch, valbatchnums), f, protocol=2)
#         f.close()
#
#     def get_data_index(self,data_dict,setname,ifshuffle=True):
#         '''
#         Get the dataset sampling index.
#         '''
#         set_id = []
#         frame_id_in_set = []
#         total_frame = 0
#         for seti,dict in enumerate(data_dict):
#             frames=sorted(dict)
#             maxframe=max(frames)-self.args.pred_length
#             frames = [x for x in frames if not x>maxframe]
#             total_frame+=len(frames)
#             set_id.extend(list(seti for i in range(len(frames))))
#             frame_id_in_set.extend(list(frames[i] for i in range(len(frames))))
#         all_frame_id_list = list(i for i in range(total_frame))
#
#         data_index = np.concatenate((np.array([frame_id_in_set], dtype=int), np.array([set_id], dtype=int),
#                                  np.array([all_frame_id_list], dtype=int)), 0)
#         if ifshuffle:
#             random.Random().shuffle(all_frame_id_list)
#         data_index = data_index[:, all_frame_id_list]
#
#         #to make full use of the data
#         if setname=='train':
#             data_index=np.append(data_index,data_index[:,:self.args.batch_size],1)
#         return data_index
#
#     def get_seq_from_index_balance(self,frameped_dict,pedtraject_dict,data_index,setname):
#         '''
#         Query the trajectories fragments from data sampling index.
#         Notes: Divide the scene if there are too many people; accumulate the scene if there are few people.
#                This function takes less gpu memory.
#         '''
#         batch_data_mass=[]
#         batch_data=[]
#         Batch_id=[]
#
#         if setname=='train':
#             skip=self.trainskip
#         else:
#             skip=self.testskip
#
#         ped_cnt=0
#         batch_count = 0
#         batch_data_64 =[]
#         batch_split = []
#         start, end = 0, 0
#         nei_lists = []
#         for i in range(data_index.shape[1]):
#             if i%100==0:
#                 print(i,'/number of frames of data in total',data_index.shape[1])
#             cur_frame,cur_set,_= data_index[:,i]
#             framestart_pedi=set(frameped_dict[cur_set][cur_frame])
#             try:
#                 frameend_pedi=set(frameped_dict[cur_set][cur_frame+(self.args.pred_length-1+self.args.min_obs)*skip[cur_set]])
#             except:
#                 if i == data_index.shape[1] - 1 and self.args.batch_size != 1:
#                    batch_data_mass.append((batch_data_64,batch_split,nei_lists,))
#                 continue
#             present_pedi=framestart_pedi | frameend_pedi
#             if (framestart_pedi & frameend_pedi).__len__()==0:
#                 if i == data_index.shape[1] - 1 and self.args.batch_size != 1:
#                    batch_data_mass.append((batch_data_64,batch_split,nei_lists,))
#                 continue
#             traject=()
#             for ped in present_pedi:
#                 cur_trajec, ifexistobs = self.find_trajectory_fragment(pedtraject_dict[cur_set][ped], cur_frame,
#                                                              self.args.seq_length,skip[cur_set])
#                 if len(cur_trajec) == 0:
#                     continue
#                 if ifexistobs==False:
#                     # Just ignore trajectories if their data don't exsist at the last obversed time step (easy for data shift)
#                     continue
#                 cur_trajec=(cur_trajec[:,1:].reshape(-1,1,self.args.input_size),)
#                 traject=traject.__add__(cur_trajec) # tuple of cur_trajec arrays in the same scene
#             if traject.__len__()<1:
#                 if i == data_index.shape[1] - 1 and self.args.batch_size != 1:
#                    batch_data_mass.append((batch_data_64,batch_split,nei_lists,))
#                 continue
#             self.num_tra += traject.__len__()
#             end += traject.__len__()
#             batch_split.append([start, end])
#             start = end
#             traject_batch=np.concatenate(traject,1) # ped dimension
#             cur_pednum = traject_batch.shape[1]
#             batch_id = (cur_set, cur_frame,)
#             cur_batch_data,cur_Batch_id=[],[]
#             cur_batch_data.append(traject_batch)
#             cur_Batch_id.append(batch_id)
#             cur_batch_data, nei_list=self.massup_batch(cur_batch_data)
#             nei_lists.append(nei_list)
#             ped_cnt += cur_pednum
#             batch_count += 1
#             if self.args.batch_size == 1:
#                 batch_data_mass.append((cur_batch_data,batch_split,nei_lists,))
#                 batch_split = []
#                 start, end = 0, 0
#                 nei_lists = []
#             else:
#                 if batch_count == self.args.batch_size or i == data_index.shape[1] - 1:
#                     batch_data_64 = self.merg_batch(cur_batch_data, batch_data_64)
#                     batch_data_mass.append((batch_data_64,batch_split,nei_lists,))
#                     batch_count = 0
#                     batch_split = []
#                     start, end = 0, 0
#                     nei_lists = []
#                 else:
#                     if batch_count ==1:
#                         batch_data_64 = cur_batch_data
#                     else:
#                         batch_data_64 = self.merg_batch(cur_batch_data, batch_data_64)
#         return batch_data_mass
#
#
#     def merg_batch(self, cur_batch_data, batch_data_64):
#         merge_batch_data = []
#         for cur_data, data_64 in zip(cur_batch_data, batch_data_64):
#             merge = np.concatenate([data_64, cur_data], axis=0)
#             merge_batch_data.append(merge)
#
#         return merge_batch_data
#
#     def find_trajectory_fragment(self, trajectory,startframe,seq_length,skip):
#         '''
#         Query the trajectory fragment based on the index. Replace where data isn't exsist with 0.
#         '''
#         return_trajec = np.zeros((seq_length, self.args.input_size+1))
#         endframe=startframe+(self.args.pred_length-1+self.args.min_obs)*skip
#         start_n = np.where(trajectory[:, 0] == startframe)
#         end_n=np.where(trajectory[:,0]==endframe)
#         ifexsitobs = False
#         real_startframe = startframe
#         offset_start = self.args.obs_length - self.args.min_obs
#         if start_n[0].shape[0] != 0 and end_n[0].shape[0] != 0:
#             end_n = end_n[0][0]
#             for i in range(0, self.args.obs_length- self.args.min_obs + 1):
#                 if np.where(trajectory[:, 0] == startframe-(self.args.obs_length - self.args.min_obs-i)*skip)[0].shape[0] != 0:
#                     real_startframe = startframe-(self.args.obs_length - self.args.min_obs-i)*skip
#                     start_n = np.where(trajectory[:, 0] == real_startframe)[0][0]
#                     offset_start = i
#                     break
#         else:
#             return return_trajec, ifexsitobs
#
#         candidate_seq=trajectory[start_n:end_n+1]
#         try:
#             return_trajec[offset_start:,:] = candidate_seq
#             if offset_start > 0:
#                 return_trajec[:offset_start,:] = candidate_seq[0, :]
#         except:
#             self.miss+=1
#             return return_trajec, ifexsitobs
#
#         if return_trajec[self.args.obs_length - 1, 1] != 0:
#             ifexsitobs = True
#
#         return return_trajec,  ifexsitobs
#
#
#     def massup_batch(self,batch_data):
#         '''
#         Massed up data fragements in different time window together to a batch
#         '''
#         num_Peds=0
#         for batch in batch_data:
#             num_Peds+=batch.shape[1]
#         seq_list_b=np.zeros((self.args.seq_length,0))
#         nodes_batch_b=np.zeros((self.args.seq_length,0,self.args.input_size))
#         nei_list_b=np.zeros((self.args.seq_length,num_Peds,num_Peds))
#         nei_num_b=np.zeros((self.args.seq_length,num_Peds))
#         num_Ped_h=0
#         batch_pednum=[]
#         for batch in batch_data:
#             num_Ped=batch.shape[1]
#             seq_list, nei_list,nei_num = self.get_social_inputs_numpy(batch)
#             nodes_batch_b=np.append(nodes_batch_b,batch,1)
#             seq_list_b=np.append(seq_list_b,seq_list,1)
#             nei_list_b[:,num_Ped_h:num_Ped_h+num_Ped,num_Ped_h:num_Ped_h+num_Ped]=nei_list
#             nei_num_b[:,num_Ped_h:num_Ped_h+num_Ped]=nei_num
#             batch_pednum.append(num_Ped)
#             num_Ped_h +=num_Ped
#             batch_data = (nodes_batch_b, seq_list_b, nei_list_b,nei_num_b,batch_pednum)
#         return self.get_dm_offset(batch_data)
#
#     def get_dm_offset(self, inputs):
#         """   batch_abs: the (orientated) batch [H, N, inputsize] inputsize: x,y,z,yaw,h,w,l,label
#         batch_norm: the batch shifted by substracted the last position.
#         shift_value: the last observed position
#         seq_list: [seq_length, num_peds], mask for position with actual values at each frame for each ped
#         nei_list: [seq_length, num_peds, num_peds], mask for neigbors at each frame
#         nei_num: [seq_length, num_peds], neighbors at each frame for each ped
#         batch_pednum: list, number of peds in each batch"""
#         nodes_abs, seq_list, nei_list, nei_num, batch_pednum = inputs
#         cur_ori = nodes_abs.copy()
#         cur_ori, seq_list =  cur_ori.transpose(1, 0, 2), seq_list.transpose(1, 0)
#         nei_num = nei_num.transpose(1, 0)
#         return [cur_ori, seq_list, nei_num], nei_list  #[N, H], [N, H], [N, H], [H, N, N
#
#     def get_social_inputs_numpy(self, inputnodes):
#         '''
#         Get the sequence list (denoting where data exsist) and neighboring list (denoting where neighbors exsist).
#         '''
#         num_Peds = inputnodes.shape[1]
#
#         seq_list = np.zeros((inputnodes.shape[0], num_Peds))
#         # denote where data not missing
#         for pedi in range(num_Peds):
#             seq = inputnodes[:, pedi]
#             seq_list[seq[:, 0] != 0, pedi] = 1
#         # get relative cords, neighbor id list
#         nei_list = np.zeros((inputnodes.shape[0], num_Peds, num_Peds))
#         nei_num = np.zeros((inputnodes.shape[0], num_Peds))
#         # nei_list[f,i,j] denote if j is i's neighbors in frame f
#         for pedi in range(num_Peds):
#             nei_list[:, pedi, :] = seq_list
#             nei_list[:, pedi, pedi] = 0  # person i is not the neighbor of itself
#             nei_num[:, pedi] = np.sum(nei_list[:, pedi, :], 1)
#             seqi = inputnodes[:, pedi]
#             for pedj in range(num_Peds):
#                 seqj = inputnodes[:, pedj]
#                 select = (seq_list[:, pedi] > 0) & (seq_list[:, pedj] > 0)
#                 relative_cord = seqi[select, :2] - seqj[select, :2]
#                 # invalid data index
#                 select_dist = (abs(relative_cord[:, 0]) > self.args.neighbor_thred) | (abs(relative_cord[:, 1]) > self.args.neighbor_thred)
#                 nei_num[select, pedi] -= select_dist
#                 select[select == True] = select_dist
#                 nei_list[select, pedi, pedj] = 0
#         return seq_list, nei_list, nei_num

    # def rotate_shift_batch(self,batch_data,epoch,idx,ifrotate=True):
    #     '''
    #     Random ration and zero shifting.
    #     Random rotation is also helpful for reducing overfitting.
    #     For one mini-batch, random rotation is employed for data augmentation.
    #     #[N, H, 2] [N, H], [N, G, G, 4] , (B, H, W) #[position, angle, framenum, ego or nei]
    #     '''
    #     nodes_abs, seq_list, nei_num = batch_data
    #     nodes_abs = nodes_abs.transpose(1, 0, 2) #[H, N, 2]
    #     #rotate batch
    #     if ifrotate:
    #         th = np.random.random() * np.pi
    #         cur_ori = nodes_abs.copy()
    #         nodes_abs[:, :, 0] = cur_ori[:, :, 0] * np.cos(th) - cur_ori[:,:, 1] * np.sin(th)
    #         nodes_abs[:, :, 1] = cur_ori[:, :, 0] * np.sin(th) + cur_ori[:,:, 1] * np.cos(th)
    #     s = nodes_abs[self.args.obs_length - 1,:,:2]
    #     #，shift the origin to the latest observed time step
    #     shift_value = np.repeat(s.reshape((1, -1, 2)), self.args.seq_length, 0)
    #     batch_data=nodes_abs, nodes_abs[:,:,:2]-shift_value, shift_value, seq_list, nei_num
    #     return batch_data


    # def get_train_batch(self,idx,epoch):
    #     batch_data,batch_split,nei_lists = self.trainbatch[idx]
    #     batch_data = self.rotate_shift_batch(batch_data,epoch,idx,ifrotate=self.args.randomRotate)
    #
    #     return batch_data,batch_split,nei_lists
    # def get_val_batch(self,idx,epoch):
    #     batch_data,batch_split,nei_lists  = self.valbatch[idx]
    #     batch_data = self.rotate_shift_batch(batch_data,epoch,idx,ifrotate=False)
    #     return batch_data,batch_split,nei_lists

    # def get_test_batch(self,idx,epoch):
    #     batch_data, batch_split, nei_lists  = self.testbatch[idx]
    #     # batch_data = self.rotate_shift_batch(batch_data,epoch,idx,ifrotate=False)
    #     return batch_data, batch_split, nei_lists
