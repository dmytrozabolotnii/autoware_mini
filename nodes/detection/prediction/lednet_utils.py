# Based on https://github.com/MediaBrain-SJTU/LED

import argparse
import torch
import numpy as np
import random
import yaml

import os.path as osp
from torch.utils import data

from LED.models.model_led_initializer import LEDInitializer as InitializationModel
from LED.models.model_diffusion import TransformerDenoisingModel as CoreDenoisingModel


def make_beta_schedule(schedule: str = 'linear',
                       n_timesteps: int = 1000,
                       start: float = 1e-5, end: float = 1e-2) -> torch.Tensor:
    '''
    Make beta schedule.

    Parameters
    ----
    schedule: str, in ['linear', 'quad', 'sigmoid'],
    n_timesteps: int, diffusion steps,
    start: float, beta start, `start<end`,
    end: float, beta end,

    Returns
    ----
    betas: Tensor with the shape of (n_timesteps)

    '''
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start

    return betas


def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)

    return out.reshape(*reshape)


def p_sample_accelerate(x, mask, cur_y, t, betas, alphas, one_minus_alphas_bar_sqrt, model):
    if t == 0:
        z = torch.zeros_like(cur_y).to(x.device)
    else:
        z = torch.randn_like(cur_y).to(x.device)
    t = torch.tensor([t]).cuda()
    # Factor to the model output
    eps_factor = (
                (1 - extract(alphas, t, cur_y)) / extract(one_minus_alphas_bar_sqrt, t, cur_y))
    # Model output
    beta = extract(betas, t.repeat(x.shape[0]), cur_y)
    eps_theta = model.generate_accelerate(cur_y, beta, x, mask)
    mean = (1 / extract(alphas, t, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))
    # Generate z
    z = torch.randn_like(cur_y).to(x.device)
    # Fixed sigma
    sigma_t = extract(betas, t, cur_y).sqrt()
    sample = mean + sigma_t * z * 0.00001

    return (sample)


def p_sample_loop_accelerate(x, mask, loc, betas, alphas, one_minus_alphas_bar_sqrt, model, NUM_Tau=5):
    '''
    Batch operation to accelerate the denoising process.

    x: [11, 10, 6]
    mask: [11, 11]
    cur_y: [11, 10, 20, 2]
    '''
    prediction_total = torch.Tensor().cuda()
    cur_y = loc[:, :10]
    for i in reversed(range(NUM_Tau)):
        cur_y = p_sample_accelerate(x, mask, cur_y, i, betas, alphas, one_minus_alphas_bar_sqrt, model)
    cur_y_ = loc[:, 10:]
    for i in reversed(range(NUM_Tau)):
        cur_y_ = p_sample_accelerate(x, mask, cur_y_, i, betas, alphas, one_minus_alphas_bar_sqrt, model)
    # shape: B=b*n, K=10, T, 2
    prediction_total = torch.cat((cur_y_, cur_y), dim=1)

    return prediction_total


def lednet_iter(dataset, model_initializer, model, betas, alphas, one_minus_alphas_bar_sqrt, n):

    dataloader = data.DataLoader(
        dataset, batch_size=len(dataset), shuffle=False, num_workers=0)

    model.eval()
    model_initializer.eval()

    with torch.no_grad():
        batch_by_batch_guesses = []

        for past_traj, traj_mask, in dataloader:
            batch_by_batch_guesses.append([])

            sample_prediction, mean_estimation, variance_estimation = model_initializer(past_traj, traj_mask)
            sample_prediction = torch.exp(variance_estimation / 2)[
                                    ..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(
                dim=(1, 2))[:, None, None, None]
            loc = sample_prediction + mean_estimation[:, None]

            pred_traj = p_sample_loop_accelerate(past_traj, traj_mask, loc, betas, alphas, one_minus_alphas_bar_sqrt, model)
            print(pred_traj.shape)

            pred_traj = pred_traj * dataset.traj_scale + dataset.traj_mean
            pred_traj = pred_traj.cpu().numpy()
            print(pred_traj.shape)
            for j in range(n):
                batch_by_batch_guesses[len(batch_by_batch_guesses) - 1].append(pred_traj[:, j, :, :])

            # b*n, K, T, 2
            # distances = torch.norm(fut_traj - pred_traj, dim=-1) * self.traj_scale
            # for time_i in range(1, 5):
            #     ade = (distances[:, :, :5 * time_i]).mean(dim=-1).min(dim=-1)[0].sum()
            #     fde = (distances[:, :, 5 * time_i - 1]).min(dim=-1)[0].sum()
            #     performance['ADE'][time_i - 1] += ade.item()
            #     performance['FDE'][time_i - 1] += fde.item()
            # samples += distances.shape[0]
            # count += 1
        # if count==2:
        # 	break
    # for time_i in range(4):
    #     print_log('--ADE({}s): {:.4f}\t--FDE({}s): {:.4f}'.format(time_i + 1, performance['ADE'][time_i] / samples, \
    #                                                               time_i + 1, performance['FDE'][time_i] / samples),
    #               log=self.log)

    true_guesses = [[] for _ in range(n)]
    for batch_guess in batch_by_batch_guesses:
        for i in range(n):
            true_guesses[i].extend(batch_guess[i])

    return true_guesses


class LEDNetDatasetInit(data.Dataset):
    def __init__(
            self, detected_object_trajs, end_points, pad_past=9, pad_future=0, obs_len=10, pred_len=20, inference_timer_duration=0.5
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        self.trajs = np.array([np.pad(np.array(traj), ((pad_past, pad_future), (0, 0)),
                                     mode='edge')[end_points[i]:(end_points[i] + pad_past + pad_future + 1)] for i, traj in enumerate(detected_object_trajs)])
        traj_mean = np.mean(self.trajs.reshape((-1, 2)), axis=0)
        print(traj_mean)
        traj_scale = 10

        # super(NBADataset, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        # self.norm_lap_matr = norm_lap_matr

        # if training:
        #     data_root = './data/files/nba_train.npy'
        # else:
        #     data_root = './data/files/nba_test.npy'

        # self.trajs = np.load(data_root)  # (N,15,11,2)
        # self.trajs /= (94 / 28)
        # if training:
        #     self.trajs = self.trajs[:32500]
        # else:
        #     self.trajs = self.trajs[:12500]
            # self.trajs = self.trajs[12500:25000]

        # self.batch_len = len(self.trajs)
        # print(self.batch_len)

        self.traj_abs = torch.from_numpy(self.trajs).type(torch.float)
        self.traj_norm = torch.from_numpy(self.trajs - self.trajs[:, self.obs_len - 1:self.obs_len]).type(torch.float)

        # self.traj_abs = self.traj_abs.permute(0, 2, 1, 3)
        # self.traj_norm = self.traj_norm.permute(0, 2, 1, 3)
        # self.actor_num = self.traj_abs.shape[1]

        self.traj_abs = self.traj_abs.unsqueeze(0)
        self.traj_norm = self.traj_norm.unsqueeze(0)
        self.actor_num = self.traj_abs.shape[1]
        print(self.traj_abs.shape, self.traj_abs.get_device(), self.traj_abs.cuda().get_device())
        self.traj_mean = torch.FloatTensor(traj_mean).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.traj_scale = traj_scale

        # batch_size = data['pre_motion_3D'].shape[0]

        batch_size = 1

        self.traj_mask = torch.zeros(batch_size * self.actor_num, batch_size * self.actor_num).cuda()
        for i in range(batch_size):
            self.traj_mask[i * self.actor_num:(i + 1) * self.actor_num, i * self.actor_num:(i + 1) * self.actor_num] = 1.

        initial_pos = self.traj_abs.cuda()[:, :, -1:]
        # augment input: absolute position, relative position, velocity
        past_traj_abs = self.traj_abs.cuda() - self.traj_mean
        past_traj_abs = ((self.traj_abs.cuda() - self.traj_mean) / self.traj_scale).contiguous().view(-1, 10, 2)
        past_traj_rel = ((self.traj_abs.cuda() - initial_pos) / self.traj_scale).contiguous().view(-1, 10, 2)
        past_traj_vel = torch.cat(
            (past_traj_rel[:, 1:] - past_traj_rel[:, :-1], torch.zeros_like(past_traj_rel[:, -1:])), dim=1) / inference_timer_duration
        self.past_traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1)

        fut_traj = ((self.traj_abs.cuda() - initial_pos) / traj_scale).contiguous().view(-1, 20, 2)

        # return batch_size, traj_mask, past_traj, fut_traj

        # print(self.traj_abs.shape)
    def __len__(self):
        return len(self.past_traj)

    def __getitem__(self, index):
        # print(self.traj_abs.shape)
        return self.past_traj[index], self.traj_mask[index]