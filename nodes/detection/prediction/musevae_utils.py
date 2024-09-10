# Based on https://github.com/ml1323/musevae

import torch.nn.functional as nnf

import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from torch.distributions.normal import Normal
from torch.utils import data
from scipy import ndimage
import torch
from torchvision import transforms
import os
import argparse
import cv2
import os.path as osp
import pathlib
sys.path.append(osp.join(pathlib.Path(__file__).parent.resolve(), 'MUSE_VAE'))
from model import *
from unet.unet import *

def parse_args():
    '''
    Create a parser for command-line arguments
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_id', default=0, type=int,
                        help='run id')
    parser.add_argument('--model_name', default='micro', type=str,
                        help='model name: one of [lg_ae, lg_cvae, sg_net, micro]')
    parser.add_argument('--device', default='cuda', type=str,
                        help='cpu/cuda')

    # training hyperparameters
    parser.add_argument('--batch_size', default=8, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')

    # saving directories and checkpoint/sample iterations
    parser.add_argument('--ckpt_load_iter', default=0, type=int,
                        help='iter# to load the previously saved model ' +
                             '(default=0 to start from the scratch)')
    parser.add_argument('--max_iter', default=10000, type=float,
                        help='maximum number of batch iterations')
    parser.add_argument('--ckpt_dir', default='ckpts', type=str)

    # Dataset options
    parser.add_argument('--delim', default=',', type=str)
    parser.add_argument('--loader_num_workers', default=0, type=int)
    parser.add_argument('--dt', default=0.4, type=float)
    parser.add_argument('--obs_len', default=8, type=int)
    parser.add_argument('--pred_len', default=12, type=int)
    parser.add_argument('--skip', default=1, type=int)
    parser.add_argument('--dataset_dir', default='./datasets/sdd', type=str, help='dataset directory')
    parser.add_argument('--dataset_name', default='sdd', type=str,
                        help='dataset name')
    parser.add_argument('--scale', default=100.0, type=float)
    parser.add_argument('--heatmap_size', default=256, type=int)
    parser.add_argument('--anneal_epoch', default=10, type=int)

    # Macro
    parser.add_argument('--pretrained_lg_path', default='ckpts/pretrained_models_pfsd/lg_cvae.pt', type=str)
    parser.add_argument('--w_dim', default=10, type=int)
    parser.add_argument('--fcomb', default=2, type=int)
    parser.add_argument('--fb', default=1, type=float)
    parser.add_argument('--num_goal', default=3, type=int)


    # Micro
    parser.add_argument('--kl_weight', default=50.0, type=float,
                        help='kl weight')
    parser.add_argument('--ll_prior_w', default=1.0, type=float)
    parser.add_argument('--z_dim', default=20, type=int,
                        help='dimension of the shared latent representation')
    parser.add_argument('--encoder_h_dim', default=64, type=int)
    parser.add_argument('--decoder_h_dim', default=128, type=int)
    parser.add_argument('--map_feat_dim', default=32, type=int)
    parser.add_argument('--dropout_mlp', default=0.3, type=float)
    parser.add_argument('--dropout_rnn', default=0.25, type=float)
    parser.add_argument('--mlp_dim', default=256, type=int)
    parser.add_argument('--map_mlp_dim', default=256, type=int)

    # Evaluation
    parser.add_argument('--n_w', default=5, type=int)
    parser.add_argument('--n_z', default=1, type=int)

    args, _ = parser.parse_known_args()

    return args

def load_checkpoint(ckpt_dir, device):
    sg_unet_path = os.path.join(
        ckpt_dir,
        'sg_net.pt'
    )
    encoderMx_path = os.path.join(
        ckpt_dir,
        'encoderMx.pt'
    )
    encoderMy_path = os.path.join(
        ckpt_dir,
        'encoderMy.pt'
    )
    decoderMy_path = os.path.join(
        ckpt_dir,
        'decoderMy.pt'
    )
    lg_cvae_path = os.path.join(
        ckpt_dir,
        'lg_cvae.pt'
    )

    encoderMx = torch.load(encoderMx_path, map_location=device)
    encoderMy = torch.load(encoderMy_path, map_location=device)
    decoderMy = torch.load(decoderMy_path, map_location=device)
    lg_cvae = torch.load(lg_cvae_path, map_location=device)
    sg_unet = torch.load(sg_unet_path, map_location=device)

    return encoderMx, encoderMy, decoderMy, lg_cvae, sg_unet


def eval_mode(models):
    encoderMx, encoderMy, decoderMy, lg_cvae, sg_unet = models

    encoderMx.eval()
    encoderMy.eval()
    decoderMy.eval()
    lg_cvae.eval()
    sg_unet.eval()

    return encoderMx, encoderMy, decoderMy, lg_cvae, sg_unet

class heatmap_generation(object):
    def __init__(self, dataset, obs_len, heatmap_size, sg_idx=None, device='cpu'):
        self.obs_len = obs_len
        self.device = device
        self.sg_idx = sg_idx
        self.heatmap_size = heatmap_size
        if dataset == 'pfsd':
            self.make_heatmap = self.create_psfd_heatmap
        elif dataset == 'sdd':
            self.make_heatmap = self.create_sdd_heatmap
        else:
            self.make_heatmap = self.create_nu_heatmap


    def make_one_heatmap(self, local_map, local_ic):
        map_size = local_map.shape[0]
        half = self.heatmap_size // 2
        if map_size < self.heatmap_size:
            heatmap = np.zeros_like(local_map)
            heatmap[local_ic[0], local_ic[1]] = 1
            heatmap = ndimage.filters.gaussian_filter(heatmap, sigma=2)
            extended_map = np.zeros((self.heatmap_size, self.heatmap_size))
            extended_map[half - map_size // 2:half + map_size // 2,half - map_size // 2:half + map_size // 2] = heatmap
            heatmap = extended_map
        else:
            heatmap = np.zeros_like(local_map)
            heatmap[local_ic[0], local_ic[1]] = 1000
            if map_size > 1000:
                heatmap = cv2.resize(ndimage.filters.gaussian_filter(heatmap, sigma=2),
                                           dsize=((map_size + self.heatmap_size) // 2, (map_size + self.heatmap_size) // 2))
            heatmap = cv2.resize(ndimage.filters.gaussian_filter(heatmap, sigma=2),
                                       dsize=(self.heatmap_size, self.heatmap_size))
            heatmap = heatmap / heatmap.sum()
            heatmap = ndimage.filters.gaussian_filter(heatmap, sigma=2)
        return heatmap


    def create_sdd_heatmap(self, local_ic, local_map, aug=False):
        heatmaps=[]
        half = self.heatmap_size//2
        for i in range(len(local_ic)):
            map_size = local_map[i].shape[0]
            # past
            if map_size < self.heatmap_size:
                env = np.full((self.heatmap_size,self.heatmap_size),3)
                env[half-map_size//2:half+map_size//2, half-map_size//2:half+map_size//2] = local_map[i]
                all_heatmap = [env/5]
                heatmap = np.zeros_like(local_map[i])
                heatmap[local_ic[i, :self.obs_len, 0], local_ic[i, :self.obs_len, 1]] = 1
                heatmap= ndimage.filters.gaussian_filter(heatmap, sigma=2)
                heatmap = heatmap / heatmap.sum()
                extended_map = np.zeros((self.heatmap_size, self.heatmap_size))
                extended_map[half-map_size//2:half+map_size//2, half-map_size//2:half+map_size//2] = heatmap
                all_heatmap.append(extended_map)
            else:
                env = cv2.resize(local_map[i], dsize=(self.heatmap_size, self.heatmap_size))
                all_heatmap = [env/5]
                heatmap = np.zeros_like(local_map[i])
                heatmap[local_ic[i, :self.obs_len, 0], local_ic[i, :self.obs_len, 1]] = 100
                if map_size > 1000:
                    heatmap = cv2.resize(ndimage.filters.gaussian_filter(heatmap, sigma=2),
                                               dsize=((map_size+self.heatmap_size)//2, (map_size+self.heatmap_size)//2))
                    heatmap = heatmap / heatmap.sum()
                heatmap = cv2.resize(ndimage.filters.gaussian_filter(heatmap, sigma=2), dsize=(self.heatmap_size, self.heatmap_size))
                if map_size > 3500:
                    heatmap[np.where(heatmap > 0)] = 1
                else:
                    heatmap = heatmap / heatmap.sum()
                heatmap = ndimage.filters.gaussian_filter(heatmap, sigma=2)
                all_heatmap.append(heatmap / heatmap.sum())

            # future
            if self.sg_idx is None:
                heatmap = self.make_one_heatmap(local_map[i], local_ic[i, -1])
                all_heatmap.append(heatmap)
            else:
                for j in (self.sg_idx + self.obs_len):
                    heatmap = self.make_one_heatmap(local_map[i], local_ic[i, j])
                    all_heatmap.append(heatmap)
            heatmaps.append(np.stack(all_heatmap))

        heatmaps = torch.tensor(np.stack(heatmaps)).float().to(self.device)

        if aug:
            degree = np.random.choice([0,90,180, -90])
            heatmaps = transforms.Compose([
                transforms.RandomRotation(degrees=(degree, degree))
            ])(heatmaps)
        if self.sg_idx is None:
            return heatmaps[:, :2], heatmaps[:, 2:]
        else:
            return heatmaps[:, :2], heatmaps[:, 2:], heatmaps[:, -1].unsqueeze(1)

def musevae_iter(dataset, models, device, args, n=5):

    dataloader = data.DataLoader(
        dataset, batch_size=len(dataset), shuffle=False, num_workers=0, collate_fn=seq_collate)

    encoderMx, encoderMy, decoderMy, lg_cvae, sg_unet = models

    dt = args.dt
    obs_len = args.obs_len
    pred_len = args.pred_len
    heatmap_size = args.heatmap_size
    dataset_name = args.dataset_name
    scale = args.scale
    n_w = n
    n_z = args.n_z

    eps = 1e-9
    sg_idx = np.array(range(pred_len))
    sg_idx = np.flip(pred_len - 1 - sg_idx[::(pred_len // args.num_goal)])

    hg = heatmap_generation(args.dataset_name, obs_len, args.heatmap_size, sg_idx=None, device=device)
    make_heatmap = hg.make_heatmap
    make_one_heatmap = hg.make_one_heatmap

    batch_by_batch_guesses = []
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            batch_by_batch_guesses.append([])
            (obs_traj, _, obs_traj_st, _, seq_start_end,
             _, _,
             local_map, local_ic, local_homo) = batch
            batch_size = obs_traj.size(1)

            obs_heat_map, _ = make_heatmap(local_ic, local_map)

            lg_cvae.forward(obs_heat_map, None, training=False)
            fut_rel_pos_dists = []
            pred_lg_wcs = []
            pred_sg_wcs = []

            ####### long term goals and the corresponding (deterministic) short term goals ########
            w_priors = []
            for _ in range(n_w):
                w_priors.append(lg_cvae.prior_latent_space.sample())

            for w_prior in w_priors:
                # -------- long term goal --------
                pred_lg_heat = F.sigmoid(lg_cvae.sample(lg_cvae.unet_enc_feat, w_prior))

                pred_lg_wc = []
                pred_lg_ics = []
                for i in range(batch_size):
                    map_size = local_map[i].shape
                    pred_lg_ic = []
                    for heat_map in pred_lg_heat[i]:
                        heat_map = nnf.interpolate(heat_map.unsqueeze(0).unsqueeze(0),
                                                   size=map_size, mode='bicubic',
                                                   align_corners=False).squeeze(0).squeeze(0)
                        argmax_idx = heat_map.argmax()
                        argmax_idx = [argmax_idx // map_size[0], argmax_idx % map_size[0]]
                        pred_lg_ic.append(argmax_idx)

                    pred_lg_ic = torch.tensor(pred_lg_ic).float().to(device)
                    pred_lg_ics.append(pred_lg_ic)
                    back_wc = torch.matmul(
                        torch.cat([pred_lg_ic, torch.ones((len(pred_lg_ic), 1)).to(device)], dim=1),
                        torch.transpose(local_homo[i].float().to(device), 1, 0))
                    pred_lg_wc.append(back_wc[0, :2] / back_wc[0, 2])
                pred_lg_wc = torch.stack(pred_lg_wc)
                pred_lg_wcs.append(pred_lg_wc)

                # -------- short term goal --------
                pred_lg_heat_from_ic = []
                for i in range(len(pred_lg_ics)):
                    pred_lg_heat_from_ic.append(make_one_heatmap(local_map[i], pred_lg_ics[i][
                        0].detach().cpu().numpy().astype(int)))
                pred_lg_heat_from_ic = torch.tensor(np.stack(pred_lg_heat_from_ic)).unsqueeze(1).float().to(
                    device)
                pred_sg_heat = F.sigmoid(
                    sg_unet.forward(torch.cat([obs_heat_map, pred_lg_heat_from_ic], dim=1)))

                pred_sg_wc = []
                for i in range(batch_size):
                    pred_sg_ic = []
                    map_size = local_map[i].shape
                    for heat_map in pred_sg_heat[i]:
                        heat_map = nnf.interpolate(heat_map.unsqueeze(0).unsqueeze(0),
                                                   size=map_size, mode='bicubic',
                                                   align_corners=False).squeeze(0).squeeze(0)
                        argmax_idx = heat_map.argmax()
                        argmax_idx = [argmax_idx // map_size[0], argmax_idx % map_size[0]]
                        pred_sg_ic.append(argmax_idx)

                    pred_sg_ic = torch.tensor(pred_sg_ic).float().to(device)

                    back_wc = torch.matmul(
                        torch.cat([pred_sg_ic, torch.ones((len(pred_sg_ic), 1)).to(device)], dim=1),
                        torch.transpose(local_homo[i].float().to(device), 1, 0))
                    back_wc /= back_wc[:, 2].unsqueeze(1)
                    pred_sg_wc.append(back_wc[:, :2])
                pred_sg_wc = torch.stack(pred_sg_wc)
                pred_sg_wcs.append(pred_sg_wc)

            ##### trajectories per long&short goal ####
            # -------- Micro --------
            (hx, mux, log_varx) \
                = encoderMx(obs_traj_st, seq_start_end, lg_cvae.unet_enc_feat, local_homo)

            p_dist = Normal(mux, torch.sqrt(torch.exp(log_varx)))
            z_priors = []
            for _ in range(n_z):
                z_priors.append(p_dist.sample())

            for pred_sg_wc in pred_sg_wcs:
                for z_prior in z_priors:
                    fut_rel_pos_dist_prior = decoderMy(
                        obs_traj_st[-1],
                        obs_traj[-1, :, :2],
                        hx,
                        z_prior,
                        pred_sg_wc,  # goal prediction
                        sg_idx
                    )
                    fut_rel_pos_dists.append(fut_rel_pos_dist_prior)

            ade, fde = [], []
            for dist in fut_rel_pos_dists:
                pred_fut_traj = integrate_samples(dist.rsample() * scale, obs_traj[-1, :, :2], dt=dt)
                pred_fut_traj = pred_fut_traj.cpu().numpy().swapaxes(1, 0)
                batch_by_batch_guesses[len(batch_by_batch_guesses) - 1].append(pred_fut_traj)

    true_guesses = [[] for _ in range(n)]
    for batch_guess in batch_by_batch_guesses:
        for i in range(n):
            true_guesses[i].extend(batch_guess[i])

    return true_guesses


def seq_collate(data):
    (obs_seq_list, _,
     _, inv_h_t,
     local_map, local_ic, local_homo, scale) = zip(*data)
    scale = scale[0]

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    obs_traj = torch.stack(obs_seq_list, dim=0).permute(1, 0, 2)
    # fut_traj = torch.stack(pred_seq_list, dim=0).permute(2, 0, 1)
    seq_start_end = torch.LongTensor(seq_start_end)

    inv_h_t = np.stack(inv_h_t)
    local_ic = np.stack(local_ic)
    local_homo = torch.tensor(np.stack(local_homo)).float().to(obs_traj.device)

    obs_traj_st = obs_traj.clone()
    obs_traj_st[:, :, :2] = (obs_traj_st[:,:,:2] - obs_traj_st[-1, :, :2]) / scale
    obs_traj_st[:, :, 2:] /= scale
    out = [
        obs_traj, [], obs_traj_st, [], seq_start_end,
        [], inv_h_t,
        local_map, local_ic, local_homo
    ]

    return tuple(out)


class MuseVAEDatasetInit(data.Dataset):
    def __init__(self, detected_object_trajs, velocity_objects, acceleration_objects, global_map, end_points, pad_past=8, pad_future=0, inference_timer_duration=0.5, scale=100, device='cuda'):
        self.obs_len = pad_past
        # self.pred_len = 12
        self.skip = 1
        self.scale = scale
        self.global_map = global_map

        self.traj_flat = detected_object_trajs
        traj = np.copy(self.traj_flat)
        self.velocitytraj = velocity_objects
        self.acceltraj = acceleration_objects

        self.traj = np.concatenate((traj, self.velocitytraj,
                                    self.acceltraj), axis=2)

        self.stats = {}
        self.maps = {}
        self.device = device

    def __len__(self):
        return len(self.traj)

    def __getitem__(self, idx):
        inv_h_t = np.expand_dims(np.eye(3), axis=0)

        local_map, local_ic, local_homo = self.get_local_map_ic(self.global_map, self.traj[idx][:, :2], zoom=1,
                                                                radius=80,
                                                                compute_local_homo=True)

        out = [
            torch.FloatTensor(self.traj[idx]).to(self.device), [],
            [], inv_h_t,
            local_map, local_ic, local_homo, self.scale
        ]

        return out

    def get_local_map_ic(self, global_map, all_traj, zoom=10, radius=8, compute_local_homo=False):
            radius = radius * zoom
            context_size = radius * 2
            expanded_obs_img = np.full((global_map.shape[0] + context_size, global_map.shape[1] + context_size),
                                       3, dtype=np.float32)
            expanded_obs_img[radius:-radius, radius:-radius] = global_map.astype(np.float32)  # 99~-99

            all_pixel = all_traj[:,[1,0]]
            all_pixel = radius + np.round(all_pixel).astype(int)

            local_map = expanded_obs_img[all_pixel[7, 0] - radius: all_pixel[7, 0] + radius,
                        all_pixel[7, 1] - radius: all_pixel[7, 1] + radius]

            all_pixel_local = None
            h = None
            if compute_local_homo:
                fake_pt = [all_traj[7]]
                per_pixel_dist = radius // 10

                for i in range(per_pixel_dist, radius // 2 - per_pixel_dist, per_pixel_dist):
                    fake_pt.append(all_traj[7] + [i, i] + np.random.rand(2) * (per_pixel_dist//2))
                    fake_pt.append(all_traj[7] + [-i, -i] + np.random.rand(2) * (per_pixel_dist//2))
                    fake_pt.append(all_traj[7] + [i, -i] + np.random.rand(2) * (per_pixel_dist//2))
                    fake_pt.append(all_traj[7] + [-i, i] + np.random.rand(2) * (per_pixel_dist//2))
                fake_pt = np.array(fake_pt)


                fake_pixel = fake_pt[:,[1,0]]
                fake_pixel = radius + np.round(fake_pixel).astype(int)

                temp_map_val = []
                for i in range(len(fake_pixel)):
                    temp_map_val.append(expanded_obs_img[fake_pixel[i, 0], fake_pixel[i, 1]])
                    expanded_obs_img[fake_pixel[i, 0], fake_pixel[i, 1]] = i + 10

                fake_local_pixel = []
                for i in range(len(fake_pixel)):
                    fake_local_pixel.append([np.where(local_map == i + 10)[0][0], np.where(local_map == i + 10)[1][0]])
                    expanded_obs_img[fake_pixel[i, 0], fake_pixel[i, 1]] = temp_map_val[i]

                h, _ = cv2.findHomography(np.array([fake_local_pixel]), np.array(fake_pt))

                all_pixel_local = np.matmul(np.concatenate([all_traj, np.ones((len(all_traj), 1))], axis=1),
                                            np.linalg.pinv(np.transpose(h)))
                all_pixel_local /= np.expand_dims(all_pixel_local[:, 2], 1)
                all_pixel_local = np.round(all_pixel_local).astype(int)[:, :2]
            return local_map, all_pixel_local, h



def integrate_samples(v, p_0, dt=1):
    """
    Integrates deterministic samples of velocity.

    :param v: Velocity samples
    :return: Position samples
    """
    v=v.permute(1, 0, 2)
    abs_traj = torch.cumsum(v, dim=1) * dt + p_0.unsqueeze(1)
    return  abs_traj.permute((1, 0, 2))


