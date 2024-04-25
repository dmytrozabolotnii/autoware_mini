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

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--learning_rate", type=int, default=0.002)
    parser.add_argument("--max_epochs", type=int, default=128)

    parser.add_argument('--cfg', default='led_augment')
    parser.add_argument('--gpu', type=int, default=0, help='Specify which GPU to use.')
    parser.add_argument('--train', type=int, default=1, help='Whether train or evaluate.')

    parser.add_argument("--info", type=str, default='', help='Name of the experiment. '
                                                             'It will be used in file creation.')
    return parser.parse_args()

def seq_collate(data):
    # batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs

    (pre_motion_3D, fut_motion_3D,pre_motion_mask,fut_motion_mask) = zip(*data)

    pre_motion_3D = torch.stack(pre_motion_3D,dim=0)
    fut_motion_3D = torch.stack(fut_motion_3D,dim=0)
    fut_motion_mask = torch.stack(fut_motion_mask,dim=0)
    pre_motion_mask = torch.stack(pre_motion_mask,dim=0)

    data = {
        'pre_motion_3D': pre_motion_3D,
        'fut_motion_3D': fut_motion_3D,
        'fut_motion_mask': fut_motion_mask,
        'pre_motion_mask': pre_motion_mask,
        'traj_scale': 1,
        'pred_mask': None,
        'seq': 'nba',
    }
    # out = [
    #     batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum
    # ]

    return data

class Trainer:
    def __init__(self, config):
        if torch.cuda.is_available(): torch.cuda.set_device(config.gpu)
        self.device = torch.device('cuda') if config.cuda else torch.device('cpu')
        # 	self.cfg = Config(config.cfg, config.info)
        self.cfg = yaml.safe_load(open(osp.join('LED', 'led_augment.yaml'), 'r'))


        # ------------------------- prepare train/test data loader -------------------------
        # train_dset = NBADataset(
        #     obs_len=self.cfg.past_frames,
        #     pred_len=self.cfg.future_frames,
        #     training=True)
        #
        # self.train_loader = data.DataLoader(
        #     train_dset,
        #     batch_size=self.cfg.train_batch_size,
        #     shuffle=True,
        #     num_workers=4,
        #     collate_fn=seq_collate,
        #     pin_memory=True)

        test_dset = LEDNetDatasetInit(
            obs_len=self.cfg.past_frames,
            pred_len=self.cfg.future_frames,
            training=False)

        self.test_loader = data.DataLoader(
            test_dset,
            batch_size=self.cfg.test_batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=seq_collate,
            pin_memory=True)

        # data normalization parameters
        self.traj_mean = torch.FloatTensor(self.cfg.traj_mean).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.traj_scale = self.cfg.traj_scale

        # ------------------------- define diffusion parameters -------------------------
        self.n_steps = self.cfg.diffusion.steps  # define total diffusion steps

        # make beta schedule and calculate the parameters used in denoising process.
        self.betas = self.make_beta_schedule(
            schedule=self.cfg.diffusion.beta_schedule, n_timesteps=self.n_steps,
            start=self.cfg.diffusion.beta_start, end=self.cfg.diffusion.beta_end).cuda()

        self.alphas = 1 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)

        # ------------------------- define models -------------------------
        self.model = CoreDenoisingModel().cuda()
        # load pretrained models
        model_cp = torch.load(osp.join('LED', 'base_diffusion_model.p'), map_location='cpu')
        self.model.load_state_dict(model_cp['model_dict'])

        self.model_initializer = InitializationModel(t_h=10, d_h=6, t_f=20, d_f=2, k_pred=20).cuda()

        # self.opt = torch.optim.AdamW(self.model_initializer.parameters(), lr=config.learning_rate)
        # self.scheduler_model = torch.optim.lr_scheduler.StepLR(self.opt, step_size=self.cfg.decay_step,
        #                                                        gamma=self.cfg.decay_gamma)

        # # ------------------------- prepare logs -------------------------
        # self.log = open(os.path.join(self.cfg.log_dir, 'log.txt'), 'a+')
        # self.print_model_param(self.model, name='Core Denoising Model')
        # self.print_model_param(self.model_initializer, name='Initialization Model')

        # temporal reweight in the loss, it is not necessary.
        # self.temporal_reweight = torch.FloatTensor([21 - i for i in range(1, 21)]).cuda().unsqueeze(0).unsqueeze(0) / 10

    def make_beta_schedule(self, schedule: str = 'linear',
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

    def extract(self, input, t, x):
        shape = x.shape
        out = torch.gather(input, 0, t.to(input.device))
        reshape = [t.shape[0]] + [1] * (len(shape) - 1)
        return out.reshape(*reshape)

    def p_sample_accelerate(self, x, mask, cur_y, t):
        if t == 0:
            z = torch.zeros_like(cur_y).to(x.device)
        else:
            z = torch.randn_like(cur_y).to(x.device)
        t = torch.tensor([t]).cuda()
        # Factor to the model output
        eps_factor = (
                    (1 - self.extract(self.alphas, t, cur_y)) / self.extract(self.one_minus_alphas_bar_sqrt, t, cur_y))
        # Model output
        beta = self.extract(self.betas, t.repeat(x.shape[0]), cur_y)
        eps_theta = self.model.generate_accelerate(cur_y, beta, x, mask)
        mean = (1 / self.extract(self.alphas, t, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))
        # Generate z
        z = torch.randn_like(cur_y).to(x.device)
        # Fixed sigma
        sigma_t = self.extract(self.betas, t, cur_y).sqrt()
        sample = mean + sigma_t * z * 0.00001
        return (sample)

    def test_single_model(self):
        model_path = './results/checkpoints/led_new.p'
        model_dict = torch.load(model_path, map_location=torch.device('cpu'))['model_initializer_dict']
        self.model_initializer.load_state_dict(model_dict)
        performance = {'FDE': [0, 0, 0, 0],
                       'ADE': [0, 0, 0, 0]}
        samples = 0
        # print_log(model_path, log=self.log)

        def prepare_seed(rand_seed):
            np.random.seed(rand_seed)
            random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed_all(rand_seed)

        prepare_seed(0)
        count = 0
        with torch.no_grad():
            for data in self.test_loader:
                batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data)

                sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
                sample_prediction = torch.exp(variance_estimation / 2)[
                                        ..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(
                    dim=(1, 2))[:, None, None, None]
                loc = sample_prediction + mean_estimation[:, None]

                pred_traj = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)

                fut_traj = fut_traj.unsqueeze(1).repeat(1, 20, 1, 1)
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

class LEDNetDatasetInit(data.Dataset):
    def __init__(
            self, obs_len=5, pred_len=10, training=True
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

        # super(NBADataset, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        # self.norm_lap_matr = norm_lap_matr

        if training:
            data_root = './data/files/nba_train.npy'
        else:
            data_root = './data/files/nba_test.npy'

        self.trajs = np.load(data_root)  # (N,15,11,2)
        self.trajs /= (94 / 28)
        if training:
            self.trajs = self.trajs[:32500]
        else:
            self.trajs = self.trajs[:12500]
            # self.trajs = self.trajs[12500:25000]

        self.batch_len = len(self.trajs)
        print(self.batch_len)

        self.traj_abs = torch.from_numpy(self.trajs).type(torch.float)
        self.traj_norm = torch.from_numpy(self.trajs - self.trajs[:, self.obs_len - 1:self.obs_len]).type(torch.float)

        self.traj_abs = self.traj_abs.permute(0, 2, 1, 3)
        self.traj_norm = self.traj_norm.permute(0, 2, 1, 3)
        self.actor_num = self.traj_abs.shape[1]
        # print(self.traj_abs.shape)
    def __len__(self):
        return self.batch_len

    def __getitem__(self, index):
        # print(self.traj_abs.shape)
        pre_motion_3D = self.traj_abs[index, :, :self.obs_len, :]
        fut_motion_3D = self.traj_abs[index, :, self.obs_len:, :]
        pre_motion_mask = torch.ones(11,self.obs_len)
        fut_motion_mask = torch.ones(11,self.pred_len)
        out = [
            pre_motion_3D, fut_motion_3D,
            pre_motion_mask, fut_motion_mask
        ]
        return out