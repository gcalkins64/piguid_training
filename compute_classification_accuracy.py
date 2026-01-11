import numpy as np
import warnings
import json
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append("/Users/gracecalkins/Local_Documents/local_code/piguid")
import torch
from gmvae_encoder import *  # type: ignore
from plotting import plot_latent_space_with_clusters, seabornSettings  # type: ignore
import glob
import re
import joblib  # safer and more compact than pickle for sklearn models
warnings.simplefilter("ignore", FutureWarning)

def main():
    seabornSettings()
    sns.set_palette("Paired")
    # Set your directory path
    basePath = "/Users/gracecalkins/Local_Documents/local_code/piguid_training/data/"

    inputDataPath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/orion_mars_return_2500_data_alt_range_scaled_downsampled.json"

    saveTag = 'orion_mars_return'

    prefix = "gmvae_orion_mars_return_20260111"

    LDs = [3]
    NCs = [3]

    # load in json
    with open(inputDataPath, 'r') as f:
        inputData = json.load(f)

    Nsamples = len(inputData)
    # Load in samples, compute number with energy above zero at the end
    samples = np.array([inputData[f'sample{i}']['alt_range'] for i in range(Nsamples)])
    # Get all samples labels
    labels = np.array([inputData[f'sample{i}']['label'] for i in range(Nsamples)])
    # Get probabilities
    capture_prob, escape_prob, impact_prob, miss_prob = 0, 0, 0, 0
    for ii in range(Nsamples):  # 0 capture, 1 escape, 2 impact
        if labels[ii] == "capture":
            capture_prob += 1
        elif labels[ii] == "impact":
            impact_prob += 1
        elif labels[ii] == "escape":
            escape_prob += 1
        elif labels[ii] == "miss":
            miss_prob += 1

    # Normalize capture and escape probabilities
    capture_prob /= Nsamples
    escape_prob /= Nsamples
    impact_prob /= Nsamples
    miss_prob /= Nsamples
    print("True Capture Prob: ", capture_prob)
    print("True Escape Prob: ", escape_prob)
    print("True Impact Prob: ", impact_prob)
    print("True Miss Prob: ", miss_prob)

    pred_capture_probs = np.zeros((len(LDs), len(NCs)))
    pred_escape_probs = np.zeros((len(LDs), len(NCs)))
    pred_crash_probs = np.zeros((len(LDs), len(NCs)))
    pred_miss_probs = np.zeros((len(LDs), len(NCs)))

    false_capture_percent = np.zeros((len(LDs), len(NCs)))
    false_escape_percent = np.zeros((len(LDs), len(NCs)))
    false_crash_percent = np.zeros((len(LDs), len(NCs)))
    false_miss_percent = np.zeros((len(LDs), len(NCs)))

    # Loop over all LDs and NCs
    for ll, LD in enumerate(LDs):
        for nn, NC in enumerate(NCs):
            # Load in encoder
            print(f"LD: {LD}, NC: {NC}")
            if len(NCs) > 1 or len(LDs) > 1:
                pattern = rf"^{prefix}_\d{{6}}_L{LD}_C{NC}$"
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
            encoder, params, em_reg = loadEncoderAndParams(folder_path, suffix, data_dim=36, latent_dim=LD, hidden_dims=[36,18,9], oldFlag=False)  # type: ignore

            # For each GMVAE, figure out which mixands describe which clusters as which data cluster has the smallest mahalanobis distance from each cluster mean / variance in latent space
            # run all samples through encoder
            assigned_cluster_inds = []
            for ii in range(NC):  # for each GMVAE cluster
                mahalanobis_distances = [[], [], []]  # Two lists - one for each mode
                encoded_samples = []
                for kk in range(Nsamples): # Loop over all samples
                    # Compute mahalanobis distance from sample to cluster mean / variance
                    z, logsigmasq = encoder.forward(torch.tensor(samples[kk][np.newaxis, :]).float())
                    encoded_samples.append(z)
                    mu_c = params['mu_c'][ii].clone().detach()
                    logsigmasq_c = params['logsigmasq_c'][ii].clone().detach()
                    mahalanobis_distance = torch.sqrt((z - mu_c) @ torch.inverse(torch.diag(torch.exp(logsigmasq_c))) @ (z - mu_c).T)
                    assigned_label = labels[kk]
                    mahalanobis_distances[assigned_label].append(mahalanobis_distance.detach().numpy())
                # Get mean mahanalobis distance for this mixand to each true cluster
                mean_mahalanobis_distances = [np.mean(mahalanobis_distances[0]), np.mean(mahalanobis_distances[1]), np.mean(mahalanobis_distances[2])]
                for qq in range(3):
                    print(f"Mean Mahalanobis distance for mixand {ii} to true cluster {qq}: {mean_mahalanobis_distances[qq]}")
                # Assign mixand to cluster with smallest mean mahalanobis distance
                assigned_cluster = np.nanargmin(mean_mahalanobis_distances)
                assigned_cluster_inds.append(assigned_cluster)

            # Print out assigned cluster inds
            print(assigned_cluster_inds)

            # Plot latent space with samples and color all mixands by their assigned cluster to check
            cluster_labels, cluster_colors = [], []
            bad_num, bad_num1, good_num = 0, 0, 0
            for aa, assigned_ind in enumerate(assigned_cluster_inds):
                if assigned_ind == 1:
                    cluster_labels.append(f'Escape {bad_num}')
                    cluster_colors.append('C2')
                    bad_num += 1
                elif assigned_ind == 2:
                    cluster_labels.append(f'Miss {bad_num1}')
                    cluster_colors.append('C4')
                    bad_num1 += 1
                else:  # capture
                    cluster_labels.append(f'Capture {good_num}')
                    cluster_colors.append('C0')
                    good_num += 1

            encoded_samples = np.squeeze(np.array([t.detach().numpy() for t in encoded_samples]))
            names = ['Capture', 'Escape', 'Miss']
            plot_latent_space_with_clusters(encoded_samples, labels, NC, params['mu_c'], params['logsigmasq_c'],
                                            os.path.join(folder_path,
                                                f'predicted_latent_clusters_{saveTag}_LD{LD}_NC{NC}'), names,
                                            ['C1', 'C3', 'C5'], cluster_labels, cluster_colors, dpi=300,
                                            titleTag=f" LD: {LD}, NC: {NC}")
            # plt.show()

            # compute true cluster probability by summing prob
            # ability for all mixands in that cluster
            pred_capture_prob, pred_escape_prob, pred_miss_prob = 0, 0, 0
            for aa, assigned_ind in enumerate(assigned_cluster_inds):
                if assigned_ind == 0:
                    pred_capture_prob += params['pi_c'][aa].detach().numpy()
                elif assigned_ind == 1:
                    pred_escape_prob += params['pi_c'][aa].detach().numpy()
                else:
                    pred_miss_prob += params['pi_c'][aa].detach().numpy()
            pred_capture_probs[ll, nn] = pred_capture_prob
            pred_escape_probs[ll, nn] = pred_escape_prob
            pred_crash_probs[ll, nn] = pred_miss_prob

            # pass all samples through the encoder and perform em step to see which cluster they are assigned to
            pred_labels = []
            for ii in range(Nsamples):
                sample = torch.tensor(samples[ii][:, np.newaxis].T).float()
                z, logsigmasq = encoder.forward(sample)
                gamma_c, _, _ = em_step(z, z, logsigmasq, params, em_reg)  # type: ignore
                cluster_ind = np.argmax(gamma_c.detach().numpy())
                pred_labels.append(assigned_cluster_inds[cluster_ind])
            pred_labels = np.array(pred_labels)

            # Compute number of false assignments in each cluster
            # Find indices where labels and pred_labels are different
            false_assignments = np.where(labels != pred_labels)[0]
            print(f"Number of false assignments: {len(false_assignments)}")
            print(f"False assignment %: {len(false_assignments) / Nsamples * 100}")

            # Compute number of false assignments in each cluster
            false_assignments_capture = np.where((labels == 0) & (pred_labels != 0))[0]
            false_assignments_escape = np.where((labels == 1) & (pred_labels != 1))[0]
            false_assignments_miss = np.where((labels == 2) & (pred_labels != 2))[0]
            print(f"Number of false assignments to capture cluster: {len(false_assignments_capture)}")
            print(f"Number of false assignments to escape cluster: {len(false_assignments_escape)}")
            print(f"Number of false assignments to miss cluster: {len(false_assignments_miss)}")
            # Get false assignment percentage
            false_assignment_capture_percentage = len(false_assignments_capture) / len(np.where(labels == 0)[0])
            print(
                f"False assignment percentage of member of capture cluster: {false_assignment_capture_percentage * 100}")
            if len(np.where(labels == 1)[0]) > 0:
                false_assignment_escape_percentage = len(false_assignments_escape) / len(np.where(labels == 1)[0])
                print(
                    f"False assignment percentage of member of  escape cluster: {false_assignment_escape_percentage * 100}")
            else:
                false_assignment_escape_percentage = np.nan
                print("No members in escape cluster, setting false assignment percentage to 0")
            if len(np.where(labels == 2)[0]) > 0:
                false_assignment_miss_percentage = len(false_assignments_miss) / len(np.where(labels == 2)[0])
                print(
                    f"False assignment percentage of member of miss cluster: {false_assignment_miss_percentage * 100}")
            else:
                false_assignment_miss_percentage = np.nan
                print("No members in miss cluster, setting false assignment percentage to 0")

            false_capture_percent[ll, nn] = false_assignment_capture_percentage
            false_escape_percent[ll, nn] = false_assignment_escape_percentage
            false_crash_percent[ll, nn] = false_assignment_miss_percentage

            # Plot input data energy colored by assigned cluster (pred_label)
            fig, ax = plt.subplots()
            for ii in range(Nsamples):
                if labels[ii] == 0:  # Capture
                    if pred_labels[ii] != labels[ii]:
                        ax.plot(samples[ii], color='C1', alpha=0.5)
                    else:
                        ax.plot(samples[ii], color='C0', alpha=0.5)
                elif labels[ii] == 1:  # Escape
                    if pred_labels[ii] != labels[ii]:
                        ax.plot(samples[ii], color='C3', alpha=0.5)
                    else:
                        ax.plot(samples[ii], color='C2', alpha=0.5)
                else:  # Crash
                    if labels[ii] != labels[ii]:
                        ax.plot(samples[ii], color='C5', alpha=0.5)
                    else:
                        ax.plot(samples[ii], color='C4', alpha=0.5)
            ax.set_ylabel('Scaled Energy')
            ax.set_xlabel('Downsample Index')
            ax.plot([], color='C0', label='Correctly Predicted Capture')
            ax.plot([], color='C2', label='Correctly Predicted Escape')
            ax.plot([], color='C4', label='Correctly Predicted Miss')
            ax.plot([], color='C1', label='Incorrectly Predicted Capture')
            ax.plot([], color='C3', label='Incorrectly Predicted Escape')
            ax.plot([], color='C5', label='Incorrectly Predicted Miss')
            ax.axhline(0, color='black', linestyle='--')
            ax.legend(loc='lower left')
            plt.title(f"LD: {LD}, NC: {NC}")
            plt.tight_layout()
            plt.savefig(os.path.join(folder_path, f"predicted_clusters_{saveTag}_LD{LD}_NC{NC}.png"), dpi=300)
            # plt.show()

            fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
            for ii in range(Nsamples):
                if labels[ii] == 0:  # Capture
                    if pred_labels[ii] == 1:  # Falsely assigned to escape
                        axs[0].plot(samples[ii], color='C6', alpha=0.5)
                    elif pred_labels[ii] == 2:  # Falsely assigned to impact
                        axs[0].plot(samples[ii], color='C7', alpha=0.5)
                    else:
                        axs[0].plot(samples[ii], color='C0', alpha=0.5)
                elif labels[ii] == 1:  # Escape
                    if pred_labels[ii] == 0:  # Falsely assigned to capture
                        axs[1].plot(samples[ii], color='C8', alpha=0.5)
                    elif pred_labels[ii] == 2:  # Falsely assigned to impact
                        axs[1].plot(samples[ii], color='C9', alpha=0.5)
                    else:
                        axs[1].plot(samples[ii], color='C2', alpha=0.5)
                else:  # Crash
                    if labels[ii] == 0:  # Falsely assigned to capture
                        axs[2].plot(samples[ii], color='C10', alpha=0.5)
                    elif labels[ii] == 1:  # Falsely assigned to escape
                        axs[2].plot(samples[ii], color='C11', alpha=0.5)
                    else:
                        axs[2].plot(samples[ii], color='C4', alpha=0.5)

            axs[0].plot([], color='C0', label='Correctly Predicted Capture')
            axs[0].plot([], color='C6', label='Assigned to Escape')
            axs[0].plot([], color='C7', label='Assigned to Miss')
            axs[1].plot([], color='C2', label='Correctly Predicted Escape')
            axs[1].plot([], color='C8', label='Assigned to Capture')
            axs[1].plot([], color='C9', label='Assigned to Miss')
            axs[2].plot([], color='C4', label='Correctly Predicted Miss')
            axs[2].plot([], color='C10', label='Assigned to Capture')
            axs[2].plot([], color='C11', label='Assigned to Escape')
            for ax in axs:
                ax.set_ylabel('Scaled Energy')
                ax.set_xlabel('Downsample Index')
                ax.axhline(0, color='black', linestyle='--')
                ax.legend(loc='lower left')
            plt.suptitle(f"LD: {LD}, NC: {NC}")
            plt.tight_layout()
            plt.savefig(os.path.join(folder_path, f"breakout_predicted_clusters_{saveTag}_LD{LD}_NC{NC}.png"), dpi=300)
            # plt.show()

        if len(LDs) == 1 or len(NCs) == 1:
            print(f'"encoderPathX": "{folder_path}",')
            print(f'"encoderSuffixX": "{suffix}", ')
            print(f'"captureIndsX": {[i for i, val in enumerate(assigned_cluster_inds) if val == 0]}, ')
            print(f'"escapeIndsX": {[i for i, val in enumerate(assigned_cluster_inds) if val == 1]}, ')
            print(f'"crashIndsX": {[i for i, val in enumerate(assigned_cluster_inds) if val == 2]}, ')

        if len(LDs) > 1 or len(NCs) > 1:
            # Print a booktabs latex table of the predicted capture probability for number of cluster and latent dimension
            print("Predicted Capture Probabilities")
            print("\\begin{tabular}{l" + "c" * len(NCs) + "}")
            print("\\toprule")
            print("Latent Dim & " + " & ".join([str(NC) for NC in NCs]) + " \\\\")
            print("\\midrule")
            for ll, LD in enumerate(LDs):
                print(f"{LD} & " + " & ".join(
                    [f"{pred_capture_probs[ll, nn]:.4f}" for nn in range(len(NCs))]) + " \\\\")
            print("\\bottomrule")
            print("\\end{tabular}")

            # Print a booktabs latex table of the predicted escape probability for number of cluster and latent dimension
            print("Predicted Escape Probabilities")
            print("\\begin{tabular}{l" + "c" * len(NCs) + "}")
            print("\\toprule")
            print("Latent Dim & " + " & ".join([str(NC) for NC in NCs]) + " \\\\")
            print("\\midrule")
            for ll, LD in enumerate(LDs):
                print(f"{LD} & " + " & ".join(
                    [f"{pred_escape_probs[ll, nn]:.4f}" for nn in range(len(NCs))]) + " \\\\")
            print("\\bottomrule")

            # Print a booktabs latex table of the predicted crash probability for number of cluster and latent dimension
            print("Predicted Miss Probabilities")
            print("\\begin{tabular}{l" + "c" * len(NCs) + "}")
            print("\\toprule")
            print("Latent Dim & " + " & ".join([str(NC) for NC in NCs]) + " \\\\")
            print("\\midrule")
            for ll, LD in enumerate(LDs):
                print(f"{LD} & " + " & ".join(
                    [f"{pred_crash_probs[ll, nn]:.4f}" for nn in range(len(NCs))]) + " \\\\")
            print("\\bottomrule")

            # Print capture percent misassignment
            print("\\begin{table}{H}")
            print("\\centering")
            print("\\caption{Predicted Capture Misassignments}")
            print("\\begin{tabular}{l" + "c" * len(NCs) + "}")
            print("\\toprule")
            print("Latent Dim & " + " & ".join([str(NC) for NC in NCs]) + " \\\\")
            print("\\midrule")
            for ll, LD in enumerate(LDs):
                print(f"{LD} & " + " & ".join(
                    [f"{false_capture_percent[ll, nn] * 100:.4f}" for nn in range(len(NCs))]) + " \\\\")
            print("\\bottomrule")
            print("\\end{tabular}")
            print("\\end{table}")

            # Print escape percent misassignment
            print("\\begin{table}{H}")
            print("\\centering")
            print("\\caption{Predicted Escape Misassignments}")
            print("\\begin{tabular}{l" + "c" * len(NCs) + "}")
            print("\\toprule")
            print("Latent Dim & " + " & ".join([str(NC) for NC in NCs]) + " \\\\")
            print("\\midrule")
            for ll, LD in enumerate(LDs):
                print(f"{LD} & " + " & ".join(
                    [f"{false_escape_percent[ll, nn] * 100:.4f}" for nn in range(len(NCs))]) + " \\\\")
            print("\\bottomrule")
            print("\\end{tabular}")
            print("\\end{table}")

            # Print crash percent misassignment
            print("\\begin{table}{H}")
            print("\\centering")
            print("\\caption{Predicted Miss Misassignments}")
            print("\\begin{tabular}{l" + "c" * len(NCs) + "}")
            print("\\toprule")
            print("Latent Dim & " + " & ".join([str(NC) for NC in NCs]) + " \\\\")
            print("\\midrule")
            for ll, LD in enumerate(LDs):
                print(f"{LD} & " + " & ".join(
                    [f"{false_crash_percent[ll, nn] * 100:.4f}" for nn in range(len(NCs))]) + " \\\\")
            print("\\bottomrule")
            print("\\end{tabular}")
            print("\\end{table}")

            # Print average percent misassignment
            print("\\begin{table}{H}")
            print("\\centering")
            print("\\caption{Average Misassignments}")
            print("\\begin{tabular}{l" + "c" * len(NCs) + "}")
            print("\\toprule")
            print("Latent Dim & " + " & ".join([str(NC) for NC in NCs]) + " \\\\")
            print("\\midrule")
            for ll, LD in enumerate(LDs):
                print(f"{LD} & " + " & ".join(
                    [f"{np.nanmean([false_crash_percent[ll, nn], false_escape_percent[ll, nn], false_capture_percent[ll, nn]]) * 100:.4f}"
                        for nn in range(len(NCs))]) + " \\\\")
            print("\\bottomrule")
            print("\\end{tabular}")
            print("\\end{table}")

    if __name__ == "__main__":
        main()