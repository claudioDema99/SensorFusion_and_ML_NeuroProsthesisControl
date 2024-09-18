#%% IMPORT LIBRARIES
import os
import random
import numpy as np
from scipy import signal
from vqf import offlineVQF
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# IMPORT CUSTOM MODULES
from cbpr_master_thesis.preprocessing_and_normalization import notch_filter, bandpass_filter, highpass_filter, lowpass_filter, normalize_angles_for_channels, normalize_EMG_all_channels, normalize_angles
from cbpr_master_thesis.feature_extraction import calculate_quaternions, calculate_quaternions_imu, joint_angle_determination, forward_kinematics, create_windows, get_feature_vector, extract_EMG_features, extract_features, rotation_matrix_x, rotation_matrix_y, rotation_matrix_z
from cbpr_master_thesis.model import MyMultimodalNetwork, train_binary, test_binary, test_multi_and_log, MyEMGNetwork, train_EMG, test_EMG, get_tensor_dataset, train_multiclass, test_multiclass

DEFAULT_RANDOM_SEED = 0

if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
else:
    print("No GPU available")

# gpu info
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def seed_basic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_everything(seed=DEFAULT_RANDOM_SEED):
    seed_basic(seed)

#%% FUNCTIONS

# LOAD AND LOG
'''
It loads the data and the labels from the locally saved numpy files.
If wandb_enabled is True, the function will log the raw data to W&B. 
Otherwise, it will simply return the data.
'''

def load_and_log(name_dataset, name_labels, wandb_enabled=False, no_weight_classification=True, wandb_project=None):
    if wandb_enabled:
        if wandb_project is None:
            project_name = "SL_pipeline"
        else:
            project_name = wandb_project
        with wandb.init(project=project_name, job_type="load-data") as run:
            # LOAD THE DATASET
            data, label = load(no_weight_classification, name_dataset, name_labels)
            raw_data = wandb.Artifact(
                "raw-data", type="dataset",
                description="Raw EMG and IMU data directly from the sensors",
                metadata={"source": "Delsys Trigno System",
                        "size": [np.array(data).shape]})
            # SAVE THE DATASET AND THE LABELS
            with raw_data.new_file("raw_data.pt", mode="wb") as file:
                torch.save((data, label), file)
            run.log_artifact(raw_data)
        return data, label
    else:
        return load(no_weight_classification, name_dataset, name_labels)

def load(no_weight_classification, name_dataset, name_labels):
    save_path = 'C:/Users/claud/Desktop/LocoD/SavedData/Dataset/' + name_dataset
    save_path_label = 'C:/Users/claud/Desktop/LocoD/SavedData/Dataset/' + name_labels
    data = np.load(save_path)
    label = np.load(save_path_label)
    if no_weight_classification:
        for i in range(len(label)):
            if label[i] == 1 or label[i] == 2:
                label[i] = 0
            elif label[i] == 3 or label[i] == 4:
                label[i] = 1
            else:
                print("Error")
    for i in range(len(label)):
        label[i] = label[i] - 1
    return data, label

def load_and_channel_max(name_dataset):
    data = np.load('C:/Users/claud/Desktop/LocoD/SavedData/Dataset/' + name_dataset)
    return np.max(data[:11,:], axis=1)

#%% ADDED FUNCTIONS 

# Lowpass filter design
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def high_pass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data, axis=0)

def apply_lowpass_filter(dataset, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order)
    sample_size = dataset.shape[2]
    for data in dataset:
        data = np.array(data).reshape(sample_size,18)
        filtered_data = filtfilt(b, a, data, axis=0)
        data = filtered_data
    return np.array(dataset)

def apply_highpass_filter(dataset, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order)
    sample_size = len(dataset)
    dataset = np.array(dataset).reshape(sample_size,1)
    filtered_data = filtfilt(b, a, dataset, axis=0)
    dataset = filtered_data
    return dataset

def integrate_gyroscope_imu_and_extract(data):
    gyro = []
    for d in data:
        g = []
        g.append(np.cumsum(d[3,:]))
        g.append(np.cumsum(d[4,:]))
        g.append(np.cumsum(d[5,:]))
        g.append(np.cumsum(d[9,:]))
        g.append(np.cumsum(d[10,:]))
        g.append(np.cumsum(d[11,:]))
        g.append(np.cumsum(d[15,:]))
        g.append(np.cumsum(d[16,:]))
        g.append(np.cumsum(d[17,:]))
        gyro.append(g)
    return np.array(gyro)

def extract_angles(data, mov = 0): # noise constant and degree map [5]
    gyro = integrate_gyroscope_imu_and_extract(data)
    angles = []

    shoulder_add = gyro[mov,8,:]
    for index, value in enumerate(shoulder_add):
        shoulder_add[index] = value #+ (1.422*index)
        #shoulder_add[index] = shoulder_add[index] * (-5.5770e-4)

    shoulder_flex = gyro[mov,6,:]
    for index, value in enumerate(shoulder_flex):
        shoulder_flex[index] = value #+ (13.8979*index)
        #shoulder_flex[index] = shoulder_flex[index] * (-5.4185e-4)

    shoulder_rot = gyro[mov,7,:]
    for index, value in enumerate(shoulder_rot):
        shoulder_rot[index] = value #- (0.974537*index)
        #shoulder_rot[index] = shoulder_rot[index] * (-8.4622e-4)

    sensor2_x = data[mov,9,:]
    sensor3_x = data[mov,15,:]
    elbow_gyr = sensor2_x - sensor3_x
    elbow = np.cumsum(elbow_gyr)
    for index, value in enumerate(elbow):
        elbow[index] = value #- (11.7011*index) #1.7
        #elbow[index] = elbow[index] * (6.1026e-4) + 90

    wrist = gyro[mov,1,:]
    for index, value in enumerate(wrist):
        wrist[index] = value #- (4.8824*index)
        #wrist[index] = wrist[index] * (5.1667e-4)

    angles.append(shoulder_add)
    angles.append(shoulder_flex)
    angles.append(shoulder_rot)
    angles.append(elbow)
    angles.append(wrist)
    return np.array(angles)

def extract_angles_shoulder(data, mov = 0): # noise constant and degree map [5]
    gyro = integrate_gyroscope_imu_and_extract(data)
    shoulder_add = gyro[mov,8,:]
    for index, value in enumerate(shoulder_add):
        shoulder_add[index] = value #+ (1.422*index)
        #shoulder_add[index] = shoulder_add[index] * (-5.5770e-4)
    #mov +=1
    shoulder_flex = gyro[mov,6,:]
    for index, value in enumerate(shoulder_flex):
        shoulder_flex[index] = value #+ (13.8979*index)
        #shoulder_flex[index] = shoulder_flex[index] * (-5.4185e-4)
    #mov +=1
    shoulder_rot = gyro[mov,7,:]
    for index, value in enumerate(shoulder_rot):
        shoulder_rot[index] = value #- (0.974537*index)
        #shoulder_rot[index] = shoulder_rot[index] * (-8.4622e-4)
    return shoulder_add, shoulder_flex, shoulder_rot


#%% LOAD DATA AND LOW PASS

data, label = load_and_log('single_positions_dataset.npy', 'single_positions_labels.npy', no_weight_classification=False, wandb_enabled=False, wandb_project="SL_multiclass")
#data, label = load_and_log('multi3_dataset.npy', 'multi3_labels.npy', no_weight_classification=False, wandb_enabled=False, wandb_project="SL_multiclass")
#data = apply_lowpass_filter(data, cutoff=50, fs=2000.0, order=5)
print(data.shape)

#%% EXTRACT ANGLES

#shoulder_add, shoulder_flex, shoulder_rot = extract_angles_shoulder(data, mov = 1)
#angles = np.array([shoulder_add, shoulder_flex, shoulder_rot])

angles = extract_angles(data, mov=0)

print(angles.shape)

#%% CALCULATE HIGH PASS AND TREND (LINEAR REGRESSION)

bias = np.zeros(angles.shape)
trend = np.zeros(angles.shape)
desired_trend = np.zeros(angles.shape)
t = np.arange(len(angles[0]))
for i in range(trend.shape[0]):
    bias[i,:] = high_pass_filter(angles[i,:], cutoff=0.5, fs=2000.0, order=4)
    p = np.polyfit(t, angles[i,:], 1)
    trend[i,:] = np.polyval(p, t)
    p = np.polyfit(t, bias[i,:], 1)
    desired_trend[i,:] = np.polyval(p, t)

#%% ADJUST VALUES

original_angles = angles
angles = angles - (trend - desired_trend)
for ang in angles:
    ang -= (ang[0]+ang[-1])/2
    #ang -= ang[-1]

#%% PLOT

num_plots = 5
fig, axs = plt.subplots(num_plots, 1, figsize=(12, 18), sharex=True, sharey=True)
time = [range(len(angles[0])),range(len(angles[1])),range(len(angles[2])),range(len(angles[3])),range(len(angles[4]))]
titles = ['Shoulder Adduction', 'Shoulder Flexion', 'Shoulder Rotation', 'Elbow', 'Wrist']
for i in range(num_plots):
    axs[i].plot(time[i], original_angles[i])
    axs[i].plot(time[i], angles[i])
    #axs[i].plot(time[i], bias[i])
    #axs[i].plot(time[i], trend[i])
    #axs[i].plot(time[i], desired_trend[i])
    axs[i].set_ylabel(f'Angle {i+1} values')
    axs[i].grid(True)
    axs[i].set_title(titles[i])
fig.suptitle('Subplots of Data', y=0.92)
plt.show()


# %%
