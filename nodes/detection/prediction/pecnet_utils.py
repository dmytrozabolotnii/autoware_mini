# Based on https://github.com/HarshayuGirase/Human-Path-Prediction/tree/master/PECNet

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers)-1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x


class PECNet(nn.Module):
    def __init__(self, enc_past_size, enc_dest_size, enc_latent_size, dec_size, predictor_size, non_local_theta_size, non_local_phi_size, non_local_g_size, fdim, zdim, nonlocal_pools, non_local_dim, sigma, past_length, future_length, verbose):
        '''
        Args:
            size parameters: Dimension sizes
            nonlocal_pools: Number of nonlocal pooling operations to be performed
            sigma: Standard deviation used for sampling N(0, sigma)
            past_length: Length of past history (number of timesteps)
            future_length: Length of future trajectory to be predicted
        '''
        super(PECNet, self).__init__()

        self.zdim = zdim
        self.nonlocal_pools = nonlocal_pools
        self.sigma = sigma

        # takes in the past
        self.encoder_past = MLP(input_dim = past_length*2, output_dim = fdim, hidden_size=enc_past_size)

        self.encoder_dest = MLP(input_dim = 2, output_dim = fdim, hidden_size=enc_dest_size)

        self.encoder_latent = MLP(input_dim = 2*fdim, output_dim = 2*zdim, hidden_size=enc_latent_size)

        self.decoder = MLP(input_dim = fdim + zdim, output_dim = 2, hidden_size=dec_size)

        self.non_local_theta = MLP(input_dim = 2*fdim + 2, output_dim = non_local_dim, hidden_size=non_local_theta_size)
        self.non_local_phi = MLP(input_dim = 2*fdim + 2, output_dim = non_local_dim, hidden_size=non_local_phi_size)
        self.non_local_g = MLP(input_dim = 2*fdim + 2, output_dim = 2*fdim + 2, hidden_size=non_local_g_size)

        self.predictor = MLP(input_dim = 2*fdim + 2, output_dim = 2*(future_length-1), hidden_size=predictor_size)

        architecture = lambda net: [l.in_features for l in net.layers] + [net.layers[-1].out_features]

        # if verbose:
        #     print("Past Encoder architecture : {}".format(architecture(self.encoder_past)))
        #     print("Dest Encoder architecture : {}".format(architecture(self.encoder_dest)))
        #     print("Latent Encoder architecture : {}".format(architecture(self.encoder_latent)))
        #     print("Decoder architecture : {}".format(architecture(self.decoder)))
        #     print("Predictor architecture : {}".format(architecture(self.predictor)))
        #
        #     print("Non Local Theta architecture : {}".format(architecture(self.non_local_theta)))
        #     print("Non Local Phi architecture : {}".format(architecture(self.non_local_phi)))
        #     print("Non Local g architecture : {}".format(architecture(self.non_local_g)))

    def non_local_social_pooling(self, feat, mask):

        # N,C
        theta_x = self.non_local_theta(feat)

        # C,N
        phi_x = self.non_local_phi(feat).transpose(1,0)

        # f_ij = (theta_i)^T(phi_j), (N,N)
        f = torch.matmul(theta_x, phi_x)

        # f_weights_i =  exp(f_ij)/(\sum_{j=1}^N exp(f_ij))
        f_weights = F.softmax(f, dim = -1)

        # setting weights of non neighbours to zero
        f_weights = f_weights * mask

        # rescaling row weights to 1
        f_weights = F.normalize(f_weights, p=1, dim=1)

        # ith row of all_pooled_f = \sum_{j=1}^N f_weights_i_j * g_row_j
        pooled_f = torch.matmul(f_weights, self.non_local_g(feat))

        return pooled_f + feat

    def forward(self, x, initial_pos, dest = None, mask = None, device=torch.device('cpu')):

        # provide destination iff training
        # assert model.training
        assert self.training ^ (dest is None)
        assert self.training ^ (mask is None)

        # encode
        ftraj = self.encoder_past(x)

        if not self.training:
            z = torch.Tensor(x.size(0), self.zdim)
            z.normal_(0, self.sigma)

        else:
            # during training, use the destination to produce generated_dest and use it again to predict final future points

            # CVAE code
            dest_features = self.encoder_dest(dest)
            features = torch.cat((ftraj, dest_features), dim = 1)
            latent =  self.encoder_latent(features)

            mu = latent[:, 0:self.zdim] # 2-d array
            logvar = latent[:, self.zdim:] # 2-d array

            var = logvar.mul(0.5).exp_()
            eps = torch.DoubleTensor(var.size()).normal_()
            eps = eps.to(device)
            z = eps.mul(var).add_(mu)

        z = z.double().to(device)
        decoder_input = torch.cat((ftraj, z), dim = 1)
        generated_dest = self.decoder(decoder_input)

        if self.training:
            # prediction in training, no best selection
            generated_dest_features = self.encoder_dest(generated_dest)

            prediction_features = torch.cat((ftraj, generated_dest_features, initial_pos), dim = 1)

            for i in range(self.nonlocal_pools):
                # non local social pooling
                prediction_features = self.non_local_social_pooling(prediction_features, mask)

            pred_future = self.predictor(prediction_features)
            return generated_dest, mu, logvar, pred_future

        return generated_dest

    # separated for forward to let choose the best destination
    def predict(self, past, generated_dest, mask, initial_pos):
        ftraj = self.encoder_past(past)
        generated_dest_features = self.encoder_dest(generated_dest)
        prediction_features = torch.cat((ftraj, generated_dest_features, initial_pos), dim = 1)

        for i in range(self.nonlocal_pools):
            # non local social pooling
            prediction_features = self.non_local_social_pooling(prediction_features, mask)

        interpolated_future = self.predictor(prediction_features)
        return interpolated_future


def pecnet_iter(dataset, model, device, hyper_params, n=5):

    dataloader = data.DataLoader(
        dataset, batch_size=len(dataset), shuffle=False, num_workers=0)

    model.eval()
    all_dest_recons = []

    with torch.no_grad():
        for i, trajx in enumerate(dataloader):
            shift = trajx[:, 0, :].cpu().numpy()
            traj = trajx - trajx[:, :1, :]
            # TODO: resolve this hardcoded value
            traj *= hyper_params["data_scale"] * 1000
            trajx = torch.DoubleTensor(trajx).to(device)

            x = traj[:, :hyper_params["past_length"], :]
            y = traj[:, hyper_params["past_length"]:, :]
            y = y.cpu().numpy()

            # reshape the data
            x = x.view(-1, x.shape[1] * x.shape[2])
            x = x.to(device)

            future = y[:, :-1, :]
            dest = y[:, -1, :]
            all_l2_errors_dest = []
            all_guesses = []
            for j in range(n):
                dest_recon = model.forward(x, trajx[:, hyper_params["past_length"] - 1, :] / 1000, device=device)
                dest_recon = dest_recon.cpu().numpy()
                dest_recon = dest_recon / hyper_params["data_scale"] / 1000 + shift
                all_guesses.append(dest_recon)

    return all_guesses


class PECNetDatasetInit(data.Dataset):
    def __init__(self, detected_object_trajs, end_points, pad_past=8, pad_future=12, set_name="train", id=False, verbose=True):
        self.traj = np.array([np.pad(np.array(traj), ((pad_past, pad_future), (0, 0)),
                                     mode='edge')[end_points[i]:end_points[i] + pad_past + pad_future] for i, traj in enumerate(detected_object_trajs)])
        self.traj_flat = np.copy(self.traj)

    def __len__(self):
        return len(self.traj)

    def __getitem__(self, idx):
        return self.traj[idx]