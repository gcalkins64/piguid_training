import numpy as np
import os
import json
import sys
from gmvae_common import *  # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from torch.distributions import Normal
import torch
# from plotting import seabornSettings # type: ignore
import warnings
import glob
import re
warnings.simplefilter("ignore", FutureWarning)


def seabornSettings():
    sns.set_style('whitegrid')
    sns.color_palette("husl", 8)
    sns.set_context("notebook", rc={"lines.linewidth": 2.5, "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 12,
                                    'xtick.labelsize': 9.0, 'ytick.labelsize': 9.0, "font.family": "serif"})
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
def main():
    seabornSettings()
    sns.set_palette("Paired")
    # Set your directory path
    basePath = "/Users/gracecalkins/Local_Documents/local_code/piguid_training/data/"

    # Structure - run once for each input data, but can run multiple hyperparameters within each input file
    # INPUTS
    inputDataPath = "/Users/gracecalkins/Local_Documents/local_code/piguid_training/data/orion_mars_return_2000_data_alt_range_final_phase_scaled_downsampled.json"
    # prefix = "gmvae_orion_mars_return_(20260111|20260112)"  # Original GMVAE for PredGuid training
    prefix = "gmvae_orion_mars_return_(20260115)"

    LDs = [3,4,5] #
    NCs = [3,4,5]  #git s

    # load in json
    with open(inputDataPath, 'r') as f:
        inputData = json.load(f)

    Nsamples = len(inputData)
    # Load in samples, compute number with energy above zero at the end
    samples = np.array([inputData[f'sample{i}']['alt_range'] for i in range(Nsamples)])

    mean = np.mean(samples, axis=0)
    variance = np.var(samples, axis=0)


    for LD in LDs:
        for NC in NCs:
            print(f"Generating samples for LD: {LD}, NC: {NC}")
            if len(LDs) > 1:
                pattern = rf"^{prefix}_\d{{6}}_{LD}_{NC}$"
                folder_path = [
                    f for f in os.listdir(basePath)
                    if os.path.isdir(os.path.join(basePath, f)) and re.fullmatch(pattern, f)
                ]
                if folder_path:
                    print(f"LD {LD}, NC {NC} → {folder_path}")
                    folder_path = os.path.join(basePath, folder_path[0])
                else:
                    print(f"LD {LD}, NC {NC} → no match")

            # Get the file string after "encoder"
            suffix = [file for file in os.listdir(folder_path) if file.startswith("encoder")][0][8:-3]

            # Load in decoder and params
            decoder, params = loadDecoderAndParams(folder_path, suffix, data_dim=64, latent_dim=LD, hidden_dims=[16, 32])  # type: ignore

            # Randomly sample Nsamples from a GMM with means / variances from params
            gen_samples = []
            for j in range(Nsamples):
                c = np.random.choice(NC, p=params['pi_c'].detach().numpy())
                mu_c = params['mu_c'][c].clone().detach()
                sigma_c = torch.exp(0.5 * params['logsigmasq_c'][c]).clone().detach()
                z = Normal(0, 1).sample(mu_c.shape) * sigma_c + mu_c
                mu_x = decoder.forward(z)[0].detach().numpy()
                gen_samples.append(mu_x)

            # Get mean and variance of generate samples
            gen_samples = np.array(gen_samples)
            gen_mean = np.mean(gen_samples, axis=0)
            gen_variance = np.var(gen_samples, axis=0)

            # Plot hair plot of generated samples with 3 sigma shading of input mean and variance and generated mean and variance
            plt.figure()
            for i in range(Nsamples):
                plt.plot(gen_samples[i], color='grey', alpha=0.1)
            plt.plot([], label='GMVAE Generated Samples', color='grey', alpha=0.1)
            plt.plot(mean, label='Input Mean', color='C0', zorder=10000)
            plt.fill_between(range(len(mean)), mean - 3 * np.sqrt(variance), mean + 3 * np.sqrt(variance), alpha=0.5, label='Input 3 Sigma', color='C1', zorder=10001)
            plt.plot(gen_mean, label='Generated Mean', color='C2', zorder=10002)
            plt.fill_between(range(len(gen_mean)), gen_mean - 3 * np.sqrt(gen_variance), gen_mean + 3 * np.sqrt(gen_variance), alpha=0.5, label='Generated 3 Sigma', color='C3', zorder=10003)
            ll = plt.legend(loc = 'lower left')
            ll.set_zorder(10004)
            plt.xlabel('Downsample Index')
            plt.ylabel('Scaled Energy')
            plt.title(f"Latent Dim: {LD}, Clusters: {NC}")
            plt.tight_layout()
            plt.savefig(os.path.join(folder_path, f"generated_samples_LD{LD}_NC{NC}.png"), dpi=300)
            plt.close()


if __name__ == "__main__":
    main()