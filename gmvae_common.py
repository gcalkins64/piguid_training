import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from pytorch_lightning import LightningDataModule
import random
from tqdm.notebook import tqdm_notebook
import json
from torch.utils.data import DataLoader

def seabornSettings():
    sns.set_theme('notebook', style='whitegrid', palette='Paired', rc={"lines.linewidth": 2.5, "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 12,'xtick.labelsize': 9.0, 'ytick.labelsize': 9.0, "font.family": "serif"})
    return

def plot_latent_space_with_clusters(samples, labels, num_clusters, cluster_means, cluster_logvars, savepath,
                                    text_labels, label_colors, data_colors, epoch_num=None, x_min=None, x_max=None, y_min=None, y_max=None, dpi=100):
    latent_dim = samples.shape[1]

    if latent_dim == 2:
        samples_ = samples
        cluster_means_ = cluster_means
        cluster_stds_ = torch.exp(0.5 * cluster_logvars)
        cluster_angles_ = torch.zeros(num_clusters)

    elif latent_dim > 2:
        pca = PCA(n_components=2)
        samples_ = pca.fit_transform(samples)
        cluster_means_ = pca.transform(cluster_means)
        A = pca.components_  # projection matrix
        C = torch.diag_embed(torch.exp(cluster_logvars)) # covariance matrix [num_clusters, latent_dim, latent_dim]
        C_proj = np.matmul(np.matmul(A, C), A.T) # [num_clusters, 2, 2]
        u, s, vh = np.linalg.svd(C_proj, full_matrices=True)
        cluster_stds_ = np.sqrt(s)
        cluster_angles_ = np.arctan(u[:, 0, 1] / u[:, 0, 0])

    fig, ax = plt.subplots(figsize=(6.5,4))
    markers = ['o', '^', "s", "d", "+", "*", "v", "x", "H", "p", "D", "P", "X"]
    # assert(len(markers) >= len(text_labels))

    unique_labels = torch.unique(labels)
    print(len(unique_labels))
    for i in range(len(unique_labels)):
        print(i)
        this_label = unique_labels[i]
        samples_i = samples_[labels == this_label]
        if samples_i.shape[0] > 0:
          ax.scatter(samples_i[:, 0], samples_i[:, 1], marker=markers[i], s=50, label=text_labels[i], color=data_colors[i])

    for i in range(num_clusters):
        ax.plot(cluster_means_[i, 0], cluster_means_[i, 1], 'x', markersize=12, label=text_labels[i]+r' $\mu$', color=label_colors[i])
        ellipse2 = mpatches.Ellipse(xy=cluster_means_[i], width=4.0 * cluster_stds_[i, 0],
                                    height=4.0 * cluster_stds_[i, 1],  angle=cluster_angles_[i] * 180 / np.pi,
                                    label=text_labels[i]+r' $2\sigma$', color=label_colors[i], alpha=0.5)
        ax.add_patch(ellipse2)

    if latent_dim == 2:
        ax.set_xlabel('$z_1$')
        ax.set_ylabel('$z_2$')
    elif latent_dim > 2:
        ax.set_xlabel('PC$(z)_1$')
        ax.set_ylabel('PC$(z)_2$')

    if x_min is not None:
        ax.set_xlim([x_min, x_max])
    if y_min is not None:
        ax.set_ylim([y_min, y_max])
    # ax.set_xlim([-90, 80])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # ax.legend(loc='best')
    if epoch_num is not None:
        plt.title("Latent Space Epoch {epoch_num}".format(epoch_num=epoch_num))
    else:
        plt.title("Latent Space")
    fig.tight_layout()
    fig.savefig(savepath + '.png', dpi=dpi)
    plt.close()

# Define encoder architecture
class Encoder(nn.Module):
    """ Neural network defining q(z | x). """

    def __init__(self, data_dim, latent_dim, hidden_dims=[32, 16]):
        super().__init__()
        self.latent_dim = latent_dim
        # self.fc = nn.Sequential(
        #     nn.Linear(in_features=data_dim, out_features=hidden_dims[0]),
        #     nn.GELU(),
        #     nn.Linear(in_features=hidden_dims[0], out_features=hidden_dims[1]),
        #     nn.GELU(),
        #     nn.Linear(in_features=hidden_dims[1], out_features=2 * latent_dim),
        # )

        modules = []
        in_channels = data_dim
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.GELU())
            )
            in_channels = h_dim
        modules.append(nn.Linear(in_channels, 2 * latent_dim))

        self.fc = nn.Sequential(*modules)


    def forward(self, x):
        """ Returns Normal conditional distribution for q(z | x), with mean and
        log-variance output by a neural network.

        Args:
            x: (N, data_dim) torch.tensor
        Returns:
            Normal distribution with a batch of (N, latent_dim) means and standard deviations
        """

        out = self.fc(x)
        mu = out[:, 0:self.latent_dim]
        logsigmasq = out[:, self.latent_dim:]

        return mu, logsigmasq

# Define decoder architecture
class Decoder(nn.Module):
    """ Neural network defining p(x | z) """

    def __init__(self, data_dim, latent_dim, decoder_var, hidden_dims=[16, 32]):
        super().__init__()
        self.data_dim = data_dim
        self.decoder_var = decoder_var

        # self.fc = nn.Sequential(
        #     nn.Linear(in_features=latent_dim, out_features=hidden_dims[0]),
        #     nn.Linear(in_features=hidden_dims[0], out_features=hidden_dims[1]),
        #     nn.GELU(),
        #     nn.Linear(in_features=hidden_dims[1], out_features=data_dim),
        # )

        modules = []
        modules.append(nn.Linear(latent_dim, hidden_dims[0]))
        for i in range(len(hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.GELU())
                # nn.ReLU())
            )
        modules.append(nn.Linear(hidden_dims[-1], data_dim))
        self.fc = nn.Sequential(*modules)


    def forward(self, z):
        """ Returns Bernoulli conditional distribution of p(x | z), parametrized
        by logits.
        Args:
            z: (N, latent_dim) torch.tensor
        Returns:
            Normal distribution with a batch of (N, data_dim)
        """

        out = self.fc(z)
        mu = out
        logsigmasq = torch.ones_like(mu) * np.log(self.decoder_var)

        return mu, logsigmasq
    

def encoder_step(x_list, encoder_list, decoder_list):
    """
    Maps D-modality data to distributions of latent embeddings.
    :param x_list: length-D list of (N, data_dim) torch.tensor
    :param encoder_list: length-D list of Encoder
    :param decoder_list: length-D list of Decoder
    :param params: dictionary of non-DNN parameters
    :return:
        mu: (N, latent_dim) torch.tensor containing the mean of embeddings
        sigma: (N, latent_dim) torch.tensor containing the std dev of embeddings
    """

    assert(len(encoder_list) == len(decoder_list))

    if len(encoder_list) == 1:
        mu, logsigmasq = encoder_list[0].forward(x_list[0])

    else:
        # compute distribution of qz as product of experts
        qz_inv_var = 0
        qz_mean_inv_var = 0

        for d, encoder in enumerate(encoder_list):
            mu_, logsigmasq_ = encoder.forward(x_list[d])
            qz_inv_var += torch.exp(-logsigmasq_)
            qz_mean_inv_var += mu_ * torch.exp(-logsigmasq_)

        mu = qz_mean_inv_var / qz_inv_var  # mu = qz_mean
        logsigmasq = - torch.log(qz_inv_var)  # sigma = qz_stddev

    return mu, logsigmasq

def em_step(z, mu, logsigmasq, params, em_reg, update_by_batch=False):
    # compute gamma_c ~ p(c|z) for each x
    pi_c = params['pi_c']
    mu_c = params['mu_c']  # (K, Z)
    logsigmasq_c = params['logsigmasq_c']  # (K, Z)
    sigma_c = torch.exp(0.5 * logsigmasq_c)

    log_prob_zc = Normal(mu_c, sigma_c).log_prob(z.unsqueeze(dim=1)).sum(dim=2) + torch.log(pi_c)  #[N, K]
    log_prob_zc -= log_prob_zc.logsumexp(dim=1, keepdims=True)
    gamma_c = torch.exp(log_prob_zc) + em_reg
    gamma_c /= gamma_c.sum(dim=1, keepdims=True)

    denominator = torch.sum(gamma_c, dim=0).unsqueeze(1)
    mu_c = torch.einsum('nc,nz->cz', gamma_c, mu) / denominator
    logsigmasq_c = torch.log(torch.einsum('nc,ncz->cz', gamma_c, torch.square(mu.unsqueeze(dim=1) - mu_c) + torch.exp(logsigmasq).unsqueeze(dim=1))) - torch.log(denominator)

    if not update_by_batch:
        return gamma_c, mu_c, logsigmasq_c

    else:
        hist_weights = params['hist_weights']
        hist_mu_c = params['hist_mu_c']
        hist_logsigmasq_c = params['hist_logsigmasq_c']

        curr_weights = denominator
        new_weights = hist_weights + curr_weights
        new_mu_c = (hist_weights * hist_mu_c + curr_weights * mu_c) / new_weights
        new_logsigmasq_c = torch.log(hist_weights * torch.exp(hist_logsigmasq_c) +
                                      curr_weights * torch.exp(logsigmasq_c)) - torch.log(new_weights)

        params['hist_weights'] = new_weights
        params['hist_mu_c'] = new_mu_c
        params['hist_logsigmasq_c'] = new_logsigmasq_c
        return gamma_c, new_mu_c, new_logsigmasq_c



def decoder_step(x_list, z, encoder_list, decoder_list, params, mu, logsigmasq, gamma_c):
    """
    Computes a stochastic estimate of the ELBO.
    :param x_list: length-D list of (N, data_dim) torch.tensor
    :param z: MC samples of the encoded distributions
    :param encoder_list: length-D list of Encoder
    :param decoder_list: length-D list of Decoder
    :param params: dictionary of non-DNN parameters
    :return:
        elbo: (,) tensor containing the elbo estimation
    """
    assert(len(encoder_list) == len(decoder_list))
    mu_c = params['mu_c']
    logsigmasq_c = params['logsigmasq_c']
    pi_c = params['pi_c']

    sse = 0
    elbo = 0
    elbo_terms = np.zeros(4)
    for d, decoder in enumerate(decoder_list):
        mu_, logsigmasq_ = decoder.forward(z)
        elbo += Normal(mu_, torch.exp(0.5 * logsigmasq_)).log_prob(x_list[d]).sum()
        elbo_terms[0] = Normal(mu_, torch.exp(0.5 * logsigmasq_)).log_prob(x_list[d]).sum()
        sse += torch.sum((x_list[d] - mu_) ** 2)

    elbo += - 0.5 * torch.sum(gamma_c * (logsigmasq_c + (torch.exp(logsigmasq).unsqueeze(1) + (mu.unsqueeze(1) - mu_c) ** 2) / torch.exp(logsigmasq_c)).sum(dim=2))
    elbo_terms[1] = - 0.5 * torch.sum(gamma_c * (logsigmasq_c + (torch.exp(logsigmasq).unsqueeze(1) + (mu.unsqueeze(1) - mu_c) ** 2) / torch.exp(logsigmasq_c)).sum(dim=2))
    elbo += torch.sum(gamma_c * (torch.log(pi_c) - torch.log(gamma_c))) + 0.5 * torch.sum(1 + logsigmasq)
    elbo_terms[2] = torch.sum(gamma_c * (torch.log(pi_c) - torch.log(gamma_c)))
    elbo_terms[3] = 0.5 * torch.sum(1 + logsigmasq)

    return elbo, sse, elbo_terms

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class AerocaptureDataModuleCUDA(LightningDataModule):
    def __init__(self, data_dir: str = "./", n_train: int = 5000, n_val: int = 100, n_test: int = 100,
                 train_batch: int = 1, val_batch: int = 1, test_batch: int = 1, num_workers=8, downsampleNum=64, SEED=42):
        super().__init__()
        self.data_dir = data_dir
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.n_samples = n_train + n_val + n_test
        self.train_batch = train_batch
        self.val_batch = val_batch
        self.test_batch = test_batch
        self.num_workers = num_workers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.downsampleNum = downsampleNum
        self.generator = torch.Generator(device="cuda").manual_seed(SEED)
        self.SEED = SEED

    def setup(self, stage=None, sample_list=None):
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)

        f = open(self.data_dir)
        print('Loading File...')
        data_dict = json.load(f)
        # data_dict = pickle.load(f)
        print('...File Loaded')

        assert (self.n_samples <= len(data_dict))

        data_tr = []
        label_tr = []
        data_val = []
        label_val = []
        data_test = []
        label_test = []

        # Randomize samples
        total_samples = len(data_dict)
        if sample_list is None:
            print('Generating sample_list')
            sample_list = random.sample(range(total_samples), self.n_samples)
        print('Sample List:', sample_list)

        # ASSUMES DATA IS ALREADY DOWNSAMPLED AND SCALED
        for i in tqdm_notebook(range(self.n_samples)):
            j = sample_list[i]
            this_data = np.array(data_dict[f'sample{j}']['energy'])[:]
            this_label = data_dict[f'sample{j}']['label']

            if i >= 0 and i < self.n_train:
                data_tr.append(torch.tensor(this_data, dtype=torch.float).to(self.device))
                label_tr.append(torch.tensor(this_label, dtype=torch.uint8).to(self.device))
            elif i >= self.n_train and i < self.n_train + self.n_val:
                data_val.append(torch.tensor(this_data, dtype=torch.float).to(self.device))
                label_val.append(torch.tensor(this_label, dtype=torch.uint8).to(self.device))
            else:
                data_test.append(torch.tensor(this_data, dtype=torch.float).to(self.device))
                label_test.append(torch.tensor(this_label, dtype=torch.uint8).to(self.device))

        self.train_dataset = tuple(zip(data_tr, label_tr))
        self.val_dataset = tuple(zip(data_val, label_val))
        self.test_dataset = tuple(zip(data_test, label_test))

        self.input_dim = len(self.train_dataset[0][0])

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_stage_dataset = self.train_dataset
            self.val_stage_dataset = self.val_dataset

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_stage_dataset = self.test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch, 
            shuffle=True,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=self.generator)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch, 
            shuffle=True,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=self.generator)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch, 
            shuffle=True,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=self.generator)
