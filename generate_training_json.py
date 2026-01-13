import numpy as np
import json
from scipy.io import loadmat, savemat
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Step 2: Determine proportional splits for escape and crash
def split_proportional(indices, proportions):
    np.random.shuffle(indices)
    total = len(indices)
    n_train = int(proportions[0] * total)
    n_val = int(proportions[1] * total)
    n_test = total - n_train - n_val
    return (
        indices[:n_train],
        indices[n_train:n_train + n_val],
        indices[n_train + n_val:]
    )


def main():
    np.random.seed(1)
    # Constants
    savePath = '/Users/gracecalkins/Local_Documents/local_code/piguid_training/data'
    R_eq = 6371000 # m, Earth
    mu = 3.986e14,  # m^3/s^2
    Nruns = 2500

    # Near Escape
    removeFlags = []  # "capture", "escape", "impact"
    tag = 'orion_mars_return'
    dataPaths = ['/Users/gracecalkins/Local_Documents/local_code/piguid/data/20260110104244_dispersed_orion_mars_return_old_thetanm_R42_C2500_Ra2740']
    # norm = 1  # alt range norm
    # norm = 46665617.798219256  # Energy norm
    offset_x = 0
    offset_y = 0
    norm_x = 1
    norm_y = 1
    distributeFailureFlag = True  # if true, distribute the failures evenly across the runs, if false, randomly distribute cases
    dataType = 'alt_vel'  # energy, alt_range, alt_vel
    timeMode = 'huntest'  # huntest if only to beginning of huntest, final_phase if to beginning of final phase

    flagDownsample = True
    flagScale = True # if true, scale the data, if false, don't scale
    dataName = dataType

    downsampleNum = 36
    tau_common = np.linspace(0.0, 1.0, downsampleNum)

    # Load in runs 0 to 5000 from folder that are in mat files
    trajectories = []
    stats = []
    for dataPath in dataPaths:
        for run in trange(Nruns, desc=f"Loading from {dataPath}"):
            trajectory = pd.read_csv(f'{dataPath}/trajectory_{run}.csv')
            loaded = np.load(f'{dataPath}/stats_{run}.npz', allow_pickle=True)
            stat = dict(loaded)
            stat["phase_starts"] = dict(stat["phase_starts"].item())
            trajectories.append(trajectory)
            stats.append(stat)

    data_dict = {}
    data_mat = np.zeros((Nruns*len(dataPaths), downsampleNum))

    goodInds = []
    save_ind = 0
    labels = []
    for run in range(Nruns*len(dataPaths)):
        trajectory = trajectories[run]
        stat = stats[run]
        ts = trajectory['time'].to_numpy()
        downrange = stat['GUID_RANGE_TO_GO']

        # We only want the trajectories to final phase if that exists (end of Upcontrol)
        if timeMode == 'final_phase':
            if not np.isnan(stat['phase_starts']['FINAL_PHASE_START']):
                tF = stat['phase_starts']['FINAL_PHASE_START']
                cutoff_ind = np.where(ts <= tF)[0][-1] + 1
            else:
                tF = ts[-1]
                cutoff_ind = len(downrange)  # TODO because downrange is shorter than ts
        else:  # Huntest, Huntest start time is defined for all trajectories
            tF = stat['phase_starts']['HUNTEST_START']
            cutoff_ind = np.where(ts <= tF)[0][-1] + 1


        # ---- classification ----
        downrange = stat['GUID_RANGE_TO_GO']
        vel = trajectory['v'].to_numpy()  # m/s
        r = trajectory['r'].to_numpy()  # m
        h = r - R_eq  # m

        h_init = r[0] - R_eq
        h_final = r[-1] - R_eq
        v_final = vel[-1]
        range_final = stat['GUID_RANGE_TO_GO'][-1] # s

        if h_final > h_init:
            label = 'escape'
        elif v_final > 1000.0:
            label = 'impact'
        elif h_final < h_init and range_final > 500e3:  # add a miss but capture condition
            label = 'miss'
        else:
            label = 'capture'

        # Cutoff all data
        downrange = downrange[:cutoff_ind]
        vel = vel[:cutoff_ind]
        r = r[:cutoff_ind]
        h = h[:cutoff_ind]

        # Remove selected indicators
        if label in removeFlags:
            continue
        else:
            goodInds.append(run)

        # ---- compute data ----
        if dataType == 'energy':
            data_x =  ts[:cutoff_ind] / tF  # ∈ [0, 1]
            data_y = vel ** 2 / 2 - mu / r
        elif dataType == 'alt_range':
            data_y = h / h[0]
            data_x = (downrange.max() - downrange) / (downrange.max() - downrange.min())  # ∈ [0, 1]
            # Reverse both arrays so interp works correctly
            data_x = data_x[::-1]
            data_y = data_y[::-1]
        elif dataType == 'alt_vel':
            data_y = h / h[0]
            data_x = vel / vel[0]
            # Reverse both arrays so interp works correctly
            data_x = data_x[::-1]
            data_y = data_y[::-1]
        else:
            raise ValueError(f'Unknown dataType {dataType}')

        if flagScale:
            normed_data_x = (data_x - offset_x) / norm_x
            normed_data_y = (data_y - offset_y) / norm_y
        else:
            normed_data_x = data_x
            normed_data_y = data_y

        # ---- interpolate to fixed nondimensional grid ----
        # print(f"{len(normed_data_x)}, {len(normed_data_y)}")
        data_ds = np.interp(tau_common, normed_data_x, normed_data_y)

        # TODO
        # plt.figure()
        # plt.plot(normed_data_x, normed_data_y, marker='o')
        # plt.plot(tau_common, data_ds, marker='x')
        # plt.title(f'Sample {run} before interpolation')
        # plt.show()


        data_dict[f'sample{save_ind}'] = {dataName: data_ds.tolist(), 'label': label}
        data_mat[save_ind, :] = data_ds
        save_ind += 1
        labels.append(label)

    suffix = f'{dataName}_{timeMode}_{"scaled_" if flagScale else ""}{"downsampled" if flagDownsample else ""}'
    with open(f'{savePath}/{tag}_{Nruns}_data_{suffix}.json', 'w') as f:
        json.dump(data_dict, f)
    Nruns = len(data_dict)

    # Save datamat as a mat file
    save_path = f'{savePath}/{tag}_{Nruns}_data_{suffix}.mat'
    savemat(save_path, {'data': data_mat})

    n_train, n_test, n_val = 1024, 128, 128
    n_total = n_train + n_test + n_val
    if distributeFailureFlag:
        # Get 1024 training, 128 validation, and 128 test sample indices including ALL crashes and escapes proportionally distributed between the three sets and filling the rest with the captures
        labels = np.array(labels)

        # Step 1: Separate indices by label
        capture_idx = np.where(labels == "capture")[0]
        escape_idx = np.where(labels == "escape")[0]
        crash_idx = np.where(labels == "impact")[0]
        miss_idx = np.where(labels == "miss")[0]

        # Print the number of captures, escapes, and crashs
        print(f"Number of captures: {len(capture_idx)}")
        print(f"Number of escapes: {len(escape_idx)}")
        print(f"Number of crashes: {len(crash_idx)}")
        print(f"Number of misses: {len(miss_idx)}")

        # 1024 + 128 + 128 = 1280 total samples
        proportions = [n_train / n_total, n_test / n_total, n_val / n_total]  # [train, val, test]

        if "escape" not in removeFlags:
            escape_train, escape_val, escape_test = split_proportional(escape_idx, proportions)
        else:
            escape_train, escape_val, escape_test = np.array([]), np.array([]), np.array([])
        if "crash" not in removeFlags:
            crash_train, crash_val, crash_test = split_proportional(crash_idx, proportions)
        else:
            crash_train, crash_val, crash_test = np.array([]), np.array([]), np.array([])
        if "miss" not in removeFlags:
            miss_train, miss_val, miss_test = split_proportional(miss_idx, proportions)
        else:
            miss_train, miss_val, miss_test = np.array([]), np.array([]), np.array([])

        # Step 3: Compute how many capture samples are needed to fill each set
        train_needed = n_train - len(escape_train) - len(crash_train) - len(miss_train)
        val_needed = n_test - len(escape_val) - len(crash_val) - len(miss_val)
        test_needed = n_val - len(escape_test) - len(crash_test) - len(miss_test)

        np.random.shuffle(capture_idx)
        capture_train = capture_idx[:train_needed]
        capture_val = capture_idx[train_needed:train_needed + val_needed]
        capture_test = capture_idx[train_needed + val_needed:train_needed + val_needed + test_needed]

        # Step 4: Combine and shuffle
        train_indices = np.concatenate([escape_train, crash_train, miss_train, capture_train])
        val_indices = np.concatenate([escape_val, crash_val, miss_val, capture_val])
        test_indices = np.concatenate([escape_test, crash_test, miss_test, capture_test])

        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)
    else:
        # Randomly draw 1024 training, 128 validation, and 128 test sample
        inds = np.random.choice(Nruns, size=n_total, replace=False)
        train_indices = inds[:n_train]
        val_indices = inds[n_train:n_train+n_val]
        test_indices = inds[n_train+n_val:]

    if dataType == 'energy':
        xlabel = "Time"
        ylabel = r"Energy"
    elif dataType == 'alt_range':
        xlabel = "Downrange"
        ylabel = "Altitude"
    elif dataType == 'alt_vel':
        ylabel = "Altitude"
        xlabel = "Velocity"
    else:
        raise ValueError(f'Unknown dataType {dataType}')

    # Results
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    # Print number of captures, escapes, and misses in each list
    def count_labels(indices):
        labels_subset = [data_dict[f'sample{idx}']['label'] for idx in indices]
        unique, counts = np.unique(labels_subset, return_counts=True)
        label_counts = dict(zip(unique, counts))
        return label_counts

    train_counts = count_labels(train_indices)
    val_counts = count_labels(val_indices)
    test_counts = count_labels(test_indices)
    print(f"Training set label counts: {train_counts}")
    print(f"Validation set label counts: {val_counts}")
    print(f"Testing set label counts: {test_counts}")

    # Put the train, val, and test indices into a single sequential list and save it as a json
    all_inds = np.concatenate([train_indices, val_indices, test_indices])
    print(all_inds)
    output = {'sample_list': all_inds.tolist()}

    with open(f'{savePath}/{tag}_{Nruns}_inds_{suffix}.json', "w") as f:
        json.dump(output, f, indent=2)

    # Plot the data
    sns.set_theme('notebook', style='whitegrid', palette='Paired', rc={"lines.linewidth": 2.5, "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 12,'xtick.labelsize': 9.0, 'ytick.labelsize': 9.0, "font.family": "serif"})
    fig, ax = plt.subplots()
    for ii, run in enumerate(goodInds):
        label = data_dict[f'sample{ii}']['label']
        color = "C0" if label == "capture" else ("C1" if label == "escape" else "C2")
        ax.plot(data_mat[ii, :], color=color, marker='o', alpha=0.3)
    line1 = plt.Line2D([0], [0], color='C0', label='Capture', marker='o')
    line2 = plt.Line2D([0], [0], color='C1', label='Escape', marker='o')
    line3 = plt.Line2D([0], [0], color='C2', label='Miss', marker='o')
    ax.legend(handles=[line1, line2, line3])
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if dataType == 'energy':
        plt.hlines(0, 0, downsampleNum, colors='r', linestyles='dashed')
    plt.savefig(f'/Users/gracecalkins/Local_Documents/local_code/piguid_training/figs/{tag}_{Nruns}_data_{suffix}_{dataType}.png', dpi=300)


    # Plot the training, validation, and testing data in three subplots
    fig, axs = plt.subplots(1,3, figsize=(8,5), sharey=True)

    for ii, run in enumerate(train_indices):
        label = data_dict[f'sample{ii}']['label']
        if label == "capture":
            color = "C0"
        elif label == "escape":
            color = "C1"
        else:  # Miss
            color = "C2"
        axs[0].plot(data_mat[ii, :], color=color, marker='o', alpha=0.3)
    axs[0].set_title('Training Data')

    for ii, run in enumerate(val_indices):
        label = data_dict[f'sample{ii}']['label']
        if label == "capture":
            color = "C0"
        elif label == "escape":
            color = "C1"
        else:  # Miss
            color = "C2"
        axs[1].plot(data_mat[ii, :], color=color, marker='o', alpha=0.3)
    axs[1].set_title('Validation Data')
    axs[1].set_yticklabels([])

    for ii, run in enumerate(test_indices):
        label = data_dict[f'sample{ii}']['label']
        if label == "capture":
            color = "C0"
        elif label == "escape":
            color = "C1"
        else:  # Miss
            color = "C2"
        axs[2].plot(data_mat[ii, :], color=color, marker='o', alpha=0.3)
    axs[2].set_title('Testing Data')
    axs[2].set_yticklabels([])
    axs[0].set_ylabel(ylabel)
    axs[0].set_xlabel(xlabel)
    axs[1].set_xlabel(xlabel)
    axs[2].set_xlabel(xlabel)
    if dataType == 'energy':
        axs[0].hlines(0, 0, downsampleNum, colors='r', linestyles='dashed')
        axs[1].hlines(0, 0, downsampleNum, colors='r', linestyles='dashed')
        axs[2].hlines(0, 0, downsampleNum, colors='r', linestyles='dashed')
    axs[0].legend(handles=[line1, line2, line3])
    plt.tight_layout()
    plt.savefig(f'/Users/gracecalkins/Local_Documents/local_code/piguid_training/figs/{tag}_{Nruns}_data_{suffix}_train_val_test_{dataType}.png', dpi=300)

    # Get maximum initial energy
    max_energy = 0
    mean_energy_0 = 0
    for run in range(Nruns):
        energy = data_mat[run, 0]
        if energy > max_energy:
            max_energy = energy
        mean_energy_0 += energy
    mean_energy_0 /= Nruns
    print(f'Maximum initial value: {max_energy}')
    print(f'Mean initial value: {mean_energy_0}')



if __name__ == '__main__':
    main()
    plt.show()
