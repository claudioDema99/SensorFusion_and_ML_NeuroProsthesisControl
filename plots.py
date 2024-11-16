import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

base_folder = "C:/Users/claud/Desktop/CBPR_Recording_Folders/"
participant_folders = ["1_30_08", "2_30_08", "3_30_08", "1_31_08", "2_31_08", "3_31_08", "1_02_09", "2_02_09", "1_04_09", "2_04_09", "3_04_09", "1_05_09", "2_05_09", "3_05_09", "1_06_09"]
subfolder_types = ["DATASET_ANGLE_ESTIMATION", "DATASET_EMG", "DATASET_RAW_IMU"]
recording_numbers = ["0", "1"]

def load_data(file_path):
    data = np.load(file_path, allow_pickle=True)
    return data['emg'], data['imu'], data['label']

def plot_participant_data(participant_data, participant_folder):
    fig, axes = plt.subplots(3, 1, figsize=(15, 20))
    fig.suptitle(f"Data for Participant {participant_folder}", fontsize=16)

    for i, data_type in enumerate(subfolder_types):
        emg_data = np.concatenate([d['emg'] for d in participant_data[data_type]])
        imu_data = np.concatenate([d['imu'] for d in participant_data[data_type]])
        label_data = np.concatenate([d['label'] for d in participant_data[data_type]])

        # Plot EMG data
        axes[i].plot(emg_data[:, 0], label='EMG Channel 1')
        axes[i].set_title(f"{data_type} - EMG and IMU Data")
        axes[i].set_xlabel("Sample")
        axes[i].set_ylabel("Amplitude")
        axes[i].legend(loc='upper left')

        # Plot IMU data on twin axis
        ax2 = axes[i].twinx()
        ax2.plot(imu_data[:, 0], 'r-', label='IMU Channel 1')
        ax2.set_ylabel("IMU Value")
        ax2.legend(loc='upper right')

        # Add label information
        unique_labels, counts = np.unique(label_data, return_counts=True)
        label_info = ", ".join([f"Label {l}: {c}" for l, c in zip(unique_labels, counts)])
        axes[i].text(0.05, 0.95, label_info, transform=axes[i].transAxes, verticalalignment='top')

    plt.tight_layout()
    plt.savefig(f"participant_{participant_folder}_data_plot.png")
    plt.close()

# Main execution
for participant_folder in participant_folders:
    participant_data = {subfolder_type: [] for subfolder_type in subfolder_types}
    
    for subfolder_type in subfolder_types:
        for recording_number in recording_numbers:
            subfolder = f"{subfolder_type}_{recording_number}"
            if subfolder_type == "DATASET_ANGLE_ESTIMATION":
                file_path = os.path.join(base_folder, participant_folder, subfolder, "angles_estimation_dataset.npz")
            elif subfolder_type == "DATASET_EMG":
                file_path = os.path.join(base_folder, participant_folder, subfolder, "emg_dataset.npz")
            elif subfolder_type == "DATASET_RAW_IMU":
                file_path = os.path.join(base_folder, participant_folder, subfolder, "raw_imu_dataset.npz")
            else:
                file_path = os.path.join(base_folder, participant_folder, subfolder, f"{subfolder_type.lower()}_dataset.npz")
            
            if os.path.exists(file_path):
                emg, imu, label = load_data(file_path)
                participant_data[subfolder_type].append({'emg': emg, 'imu': imu, 'label': label})
            else:
                print(f"File not found: {file_path}")
    
    plot_participant_data(participant_data, participant_folder)

print("Plotting complete. Check the output directory for the generated plots.")