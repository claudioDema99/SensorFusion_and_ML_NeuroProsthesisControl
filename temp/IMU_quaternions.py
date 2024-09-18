#%% IMPORT LIBRARIES
import os
import random
import numpy as np
from scipy import signal
from vqf import offlineVQF, VQF
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
    sample_size = dataset.shape[2]
    for data in dataset:
        data = np.array(data).reshape(sample_size,18)
        filtered_data = filtfilt(b, a, data, axis=0)
        data = filtered_data
    return dataset

def high_pass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data, axis=0)

#%% ADDED FUNCTIONS 

# vqf needs gyr in rad/s and acc in m/s^2 but the data is in deg/s and g
def convert_to_SI(dataset):
    for data in dataset:
        data[:3,:] = data[:3,:] * 9.81
        data[6:9,:] = data[6:9,:] * 9.81
        data[12:15,:] = data[12:15,:] * 9.81
        data[3:6,:] = np.deg2rad(data[3:6,:])
        data[9:12,:] = np.deg2rad(data[9:12,:])
        data[15:18,:] = np.deg2rad(data[15:18,:])
    return dataset

def extract_quaternions(dataset):
    num_samples = dataset.shape[2]
    vqf_wrist = VQF(gyrTs = 0.0005, accTs = 0.0005)
    vqf_upper = VQF(gyrTs = 0.0005, accTs = 0.0005)
    vqf_arm = VQF(gyrTs = 0.0005, accTs = 0.0005)
    quat = []
    for data in dataset:
        quats_wrist = []
        quats_upper = []
        quats_arm = []
        for i in range(num_samples):
            vqf_wrist.update(gyr=np.ascontiguousarray(data[3:6,i]), acc=np.ascontiguousarray(data[:3,i]))
            vqf_upper.update(gyr=np.ascontiguousarray(data[9:12,i]), acc=np.ascontiguousarray(data[6:9,i]))
            vqf_arm.update(gyr=np.ascontiguousarray(data[15:,i]), acc=np.ascontiguousarray(data[12:15,i]))
            quats_wrist.append(vqf_wrist.getQuat6D())
            quats_upper.append(vqf_upper.getQuat6D())
            quats_arm.append(vqf_arm.getQuat6D())
        quat.append(np.array([quats_wrist, quats_upper, quats_arm]))
    return np.array(quat)

def integrate_speeds(speeds):
    num_movements, num_sensors, _, num_angles = speeds.shape
    angles = np.zeros((num_movements, num_sensors, _, num_angles))
    for m in range(num_movements):
        for s in range(num_sensors):
            for i in range(num_angles):
                angles[m, s, :, i] = np.cumsum(speeds[m, s, :, i])
    return angles

def quaternion_to_rotation_matrix(quat):
    R = []
    for sample in quat:
        w, x, y, z = sample
        r = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]])
        R.append(r)
    return np.array(R)

def rotation_matrix_to_euler_angles(rotation_matrix):
    r = R.from_matrix(rotation_matrix)
    return r.as_euler('xyz', degrees=True)
def calculate_euler_angle_displacements(rotation_matrices):
    num_samples = rotation_matrices.shape[0]
    euler_angle_displacements = []
    for i in range(num_samples):
        if i == 0:
            R_rel = rotation_matrices[i] * 0.01
        else:
            R_rel = np.dot(rotation_matrices[i], np.linalg.inv(rotation_matrices[i-1]))
        euler_angles = rotation_matrix_to_euler_angles(R_rel)
        euler_angle_displacements.append(euler_angles)
    #euler_angle_displacements.append(rotation_matrix_to_euler_angles(rotation_matrices[-1]))# not sure
    return np.array(euler_angle_displacements)

# ABOVE THE ONES USED - BELOW THE ONES TO BE ASSESSED

def adjust_quaternions_with_bias(quat6D, bias):
    adjusted_quat6D = []
    for i in range(len(quat6D)):
        rotation = R.from_quat(quat6D[i])
        bias_rotation = R.from_rotvec(bias[i])
        adjusted_rotation = rotation * bias_rotation
        adjusted_quat6D.append(adjusted_rotation.as_quat())
    return np.array(adjusted_quat6D)

from pyquaternion import Quaternion

def quaternion_to_euler(q):
    """
    Convert a quaternion to Euler angles (in radians).
    """
    q = Quaternion(q)
    roll = np.arctan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x ** 2 + q.y ** 2))
    pitch = np.arcsin(2 * (q.w * q.y - q.z * q.x))
    yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
    return np.array([roll, pitch, yaw])

def extract_angles_from_quaternions(quaternion_data):
    # quaternion_data: array containing quaternion data for each sensor and frame [3, 20000, 4]
    angles = []
    # Processing shoulder adduction/abduction angles (sensor 1, example)
    shoulder_add_quat = quaternion_data[2,:,:]  # Quaternion data for shoulder adduction/abduction
    shoulder_add_euler = np.apply_along_axis(quaternion_to_euler, 1, shoulder_add_quat)  # Convert to Euler angles
    shoulder_add = shoulder_add_euler[:, 0]  # Extracting shoulder adduction angle (roll)
    # Processing shoulder flexion/extension angles (sensor 2, example)
    shoulder_flex_quat = quaternion_data[2,:,:]  # Quaternion data for shoulder flexion/extension
    shoulder_flex_euler = np.apply_along_axis(quaternion_to_euler, 1, shoulder_flex_quat)  # Convert to Euler angles
    shoulder_flex = shoulder_flex_euler[:, 1]  # Extracting shoulder flexion angle (pitch)
    # Processing shoulder internal/external rotation angles (sensor 3, example)
    shoulder_rot_quat = quaternion_data[2,:,:]  # Quaternion data for shoulder internal/external rotation
    shoulder_rot_euler = np.apply_along_axis(quaternion_to_euler, 1, shoulder_rot_quat)  # Convert to Euler angles
    shoulder_rot = shoulder_rot_euler[:, 2]  # Extracting shoulder rotation angle (yaw)
    # Processing elbow angles (combining sensors 2 and 3)
    sensor2_quat = quaternion_data[1,:,:]  # Quaternion data from sensor 2
    sensor3_quat = quaternion_data[2,:,:]  # Quaternion data from sensor 3
    elbow_quat = np.array([Quaternion(q1) * Quaternion(q2).inverse for q1, q2 in zip(sensor2_quat, sensor3_quat)])  # Quaternion for elbow rotation
    #elbow_euler = np.apply_along_axis(quaternion_to_euler, 1, elbow_quat)  # Convert to Euler angles
    #elbow = elbow_euler[:, 0]  # Extracting elbow angle (roll)
    # Processing wrist angles (sensor 1, example)
    wrist_quat = quaternion_data[0,:,:]  # Quaternion data for wrist
    wrist_euler = np.apply_along_axis(quaternion_to_euler, 1, wrist_quat)  # Convert to Euler angles
    wrist = wrist_euler[:, 0]  # Extracting wrist angle (roll)
    # Collect all angles into a list and convert to a numpy array
    angles.append(shoulder_add)
    angles.append(shoulder_flex)
    angles.append(shoulder_rot)
    #angles.append(elbow)
    angles.append(wrist)
    return np.array(angles)

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

def integrate_gyroscope_and_accelerometer_imu_and_extract(data):
    final_data = []
    for d in data:
        f = []
        f.append(np.cumsum(d[0,:]))
        f.append(np.cumsum(d[1,:]))
        f.append(np.cumsum(d[2,:]))
        f.append(np.cumsum(d[3,:]))
        f.append(np.cumsum(d[4,:]))
        f.append(np.cumsum(d[5,:]))
        f.append(np.cumsum(d[6,:]))
        f.append(np.cumsum(d[7,:]))
        f.append(np.cumsum(d[8,:]))
        f.append(np.cumsum(d[9,:]))
        f.append(np.cumsum(d[10,:]))
        f.append(np.cumsum(d[11,:]))
        f.append(np.cumsum(d[12,:]))
        f.append(np.cumsum(d[13,:]))
        f.append(np.cumsum(d[14,:]))
        f.append(np.cumsum(d[15,:]))
        f.append(np.cumsum(d[16,:]))
        f.append(np.cumsum(d[17,:]))
        final_data.append(f)
    return np.array(final_data)

def extract_angles(data, noise_constants, degree_map, mov = 0): # noise constant and degree map [5]
    gyro = integrate_gyroscope_imu_and_extract(data)
    angles = []

    shoulder_add = gyro[mov,8,:]
    for index, value in enumerate(shoulder_add):
        shoulder_add[index] = value + (1.422*index)
        shoulder_add[index] = shoulder_add[index] * (-5.5770e-4)

    shoulder_flex = gyro[mov,6,:]
    for index, value in enumerate(shoulder_flex):
        shoulder_flex[index] = value + (13.8979*index)
        shoulder_flex[index] = shoulder_flex[index] * (-5.4185e-4)

    shoulder_rot = gyro[mov,7,:]
    for index, value in enumerate(shoulder_rot):
        shoulder_rot[index] = value - (0.974537*index)
        shoulder_rot[index] = shoulder_rot[index] * (-8.4622e-4)

    sensor2_x = data[mov,9,:]
    sensor3_x = data[mov,15,:]
    elbow_gyr = sensor2_x - sensor3_x
    elbow = np.cumsum(elbow_gyr)
    for index, value in enumerate(elbow):
        elbow[index] = value - (11.7011*index) #1.7
        elbow[index] = elbow[index] * (6.1026e-4) + 90

    wrist = gyro[mov,1,:]
    for index, value in enumerate(wrist):
        wrist[index] = value - (4.8824*index)
        wrist[index] = wrist[index] * (5.1667e-4)

    angles.append(shoulder_add)
    angles.append(shoulder_flex)
    angles.append(shoulder_rot)
    angles.append(elbow)
    angles.append(wrist)
    return np.array(angles)

def extract_angles_elbow(data, noise_constants, degree_map, mov = 3): # noise constant and degree map [5]
    sensor2_x = data[mov,9,:]
    sensor3_x = data[mov,15,:]
    elbow_gyr = sensor2_x - sensor3_x
    elbow = np.cumsum(elbow_gyr)
    for index, value in enumerate(elbow):
        elbow[index] = value - (11.7011*index)
        elbow[index] = elbow[index] * (6.1026e-4) + 90
    return elbow

def extract_angles_wrist(data, noise_constants, degree_map, mov = 4): # noise constant and degree map [5]
    gyro = integrate_gyroscope_imu_and_extract(data)
    wrist = gyro[mov,1,:]
    for index, value in enumerate(wrist):
        wrist[index] = value - (4.8824*index)
        wrist[index] = wrist[index] * (5.1667e-4)
    return wrist

#%%
###############################################################################################################################
                                                                                                                              #
                                                                                                                              #
                                                                                                                              #
# Function to convert gyroscope data to orientation quaternions
def integrate_gyroscope_to_quaternions(gyro_data, dt=0.01):
    quaternions = [R.from_quat([1, 0, 0, 0])]  # Initial orientation (identity quaternion)
    
    for omega in gyro_data:
        omega_norm = np.linalg.norm(omega)
        if omega_norm > 0:
            axis = omega / omega_norm
            theta = omega_norm * dt
            delta_q = R.from_rotvec(axis * theta)
            new_orientation = quaternions[-1] * delta_q
            quaternions.append(new_orientation)
        else:
            quaternions.append(quaternions[-1])
    
    return quaternions

# Function to remove y-axis rotation effect
def remove_y_axis_rotation_effect(accel_data, gyro_data, quaternions):
    corrected_accel = []
    corrected_gyro = []
    
    for i, (accel, gyro, quat) in enumerate(zip(accel_data, gyro_data, quaternions)):
        # Convert quaternion to rotation matrix
        rotation_matrix = quat.as_matrix()
        
        # Extract the y-axis rotation component
        y_axis_rotation = R.from_euler('y', quat.as_euler('zyx')[1])
        inv_y_axis_rotation_matrix = y_axis_rotation.inv().as_matrix()
        
        # Remove y-axis rotation from accelerometer data
        corrected_accel.append(inv_y_axis_rotation_matrix @ accel)
        
        # Remove y-axis rotation from gyroscope data
        corrected_gyro.append(inv_y_axis_rotation_matrix @ gyro)
    
    return np.array(corrected_accel), np.array(corrected_gyro)

# Example usage with sample data
num_samples = 30000
dt = 0.0005  # Time interval between samples (adjust as per your data)

# Assuming `gyro_data` and `accel_data` are numpy arrays of shape (num_samples, 3)
gyro_data = data[15:18,:]  # Replace with actual gyroscope data
accel_data = data[12:15,:]  # Replace with actual accelerometer data

# Integrate gyroscope data to get orientation quaternions
quaternions = integrate_gyroscope_to_quaternions(gyro_data.T, dt)

# Remove y-axis rotation effect
corrected_accel, corrected_gyro = remove_y_axis_rotation_effect(accel_data.T, gyro_data.T, quaternions)

data, label = load_and_log('test_angles_dataset.npy', 'test_angles_labels.npy', no_weight_classification=False, wandb_enabled=False, wandb_project="SL_multiclass")
data[0,12:15,:] = corrected_accel.T
data[0,15:18,:] = corrected_gyro.T

print("Corrected Accelerometer Data:", corrected_accel)
print("Corrected Gyroscope Data:", corrected_gyro)
                                                                                                                              #
                                                                                                                              #
                                                                                                                              #
###############################################################################################################################

# %% QUATERNIONS ATTEMPT

# this is the other approach that I want to apply but I'm not able because of the quaternion library
# try with other library
data, label = load_and_log('test_angles_dataset.npy', 'test_angles_labels.npy', no_weight_classification=False, wandb_enabled=False, wandb_project="SL_multiclass")
data = apply_lowpass_filter(data, cutoff=50, fs=2000.0, order=5)
#integrated_data = integrate_gyroscope_and_accelerometer_imu_and_extract(data)
quat = calculate_quaternions_imu_no_avg(data)

# Quaternion multiplication function
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

# Normalization function
def normalize(quaternion):
    norm = np.linalg.norm(quaternion)
    return quaternion / norm

# Integrate quaternions for each sensor
def integrate_quaternions(quaternions):
    num_movements, num_sensors, num_samples, _ = quaternions.shape
    integrated_quaternions = np.zeros((num_movements, num_sensors, num_samples, 4))
    
    for move in range(num_movements):
        # Initialize with identity quaternion
        for sensor in range(num_sensors):
            integrated_quaternions[move, sensor, 0] = np.array([1, 0, 0, 0])
            
            for i in range(1, num_samples):
                q_current = integrated_quaternions[move, sensor, i-1]
                q_new = quaternions[move, sensor, i]
                q_integrated = quaternion_multiply(q_current, q_new)
                q_integrated = normalize(q_integrated)
                integrated_quaternions[move, sensor, i] = q_integrated
            
    return integrated_quaternions

# Example usage
# Assume `sensor_data` is the [3, 30000, 4] array of quaternions
sensor_data = np.random.rand(3, 30000, 4)  # Replace with actual data

# Normalize initial sensor data (assuming each quaternion might not be normalized)
sensor_data = np.array([[normalize(q) for q in sensor] for sensor in sensor_data])

integrated_data = integrate_quaternions(quat)

print(integrated_data.shape)  # Should be (3, 30000, 4)

# for the calculation of the elbow angle
def remove_y_rotation(R_true, gyro_data): # R_init = 3x3
    # Calculate the angular velocity around the y-axis
    gyro_y = gyro_data[1, :]  # Gyroscope data for the y-axis
    # Integrate angular velocity to get the rotation angle over time
    #dt = 1 / 2000  # Assuming a sampling rate of 2000 Hz
    #theta_y = np.cumsum(gyro_y) * dt
    # Create the inverse y-axis rotation matrices
    cos_theta_y = np.cos(-gyro_y)
    sin_theta_y = np.sin(-gyro_y)
    R_adjusted = np.zeros((gyro_data.shape[1], 3, 3))
    for i in range(gyro_data.shape[1]):
        R_y_inv = np.array([
            [cos_theta_y[i], 0, sin_theta_y[i]],
            [0, 1, 0],
            [-sin_theta_y[i], 0, cos_theta_y[i]]
        ])
        #R = np.eye(3)  # Replace with your actual initial rotation matrix for each step
        R_adjusted[i, :, :] = R_y_inv @ R_true[i, :, :]
    return R_adjusted

# R1 = wrist
# R2 = forearm
# R3 = upper arm
R1 = []
R2 = []
R3 = []
for i in range(quat.shape[2]):
    r1 = R.from_quat(integrated_data[3,0,i,:]).as_matrix()#quat
    r2 = R.from_quat(integrated_data[3,1,i,:]).as_matrix()
    r3 = R.from_quat(integrated_data[3,2,i,:]).as_matrix()
    R1.append(np.array(r1))
    R2.append(np.array(r2))
    R3.append(np.array(r3))
R1 = np.array(R1)
R2 = np.array(R2)
R3 = np.array(R3)

#R2 = remove_y_rotation(R2, data[0,9:11,:]) 

angle_elbow = []
angle_wrist = []
# Loop through each pair of consecutive rotation matrices
for i in range(data.shape[2]):
    r1 = R1[i]
    r2 = R2[i]
    r3 = R3[i]
    R_rel = np.dot(r2.T, r3)
    # Extract the angle around the x-axis from the relative rotation matrix
    # Assuming the x-axis is the first column in the rotation matrix
    angle_x = np.arccos(R_rel[0, 0])
    # Convert the angle from radians to degrees
    angle_elbow.append(np.degrees(angle_x))
    #angle_elbow.append(angle_x)
    # Calculate the relative rotation matrix
    R_rel2 = np.dot(r1.T, r2)
    # Extract the angle around the y-axis from the relative rotation matrix
    # Since the y-axis is the second column in the rotation matrix
    # We need to calculate the rotation around y-axis which involves elements R_rel[0, 2] and R_rel[2, 0]
    angle_y = np.arctan2(R_rel2[0, 2], R_rel2[2, 2])
    # Convert the angle from radians to degrees
    angle_wrist.append(np.degrees(angle_y))
    #angle_wrist.append(angle_y)

def moving_average_filter(signal, window_size=100):
    # Create an array to store the filtered signal
    filtered_signal = np.convolve(signal, np.ones(window_size) / window_size, mode='valid')
    return filtered_signal

angle_elbow = moving_average_filter(angle_elbow, window_size=1000)
angle_wrist = moving_average_filter(angle_wrist, window_size=1000)

fig, axs = plt.subplots(figsize=(12, 6), sharex=True)
time = range(len(angle_elbow))
axs.plot(time, angle_elbow, label='Elbow Angle')
axs.grid(True)
axs.set_title('Elbow Angle')
axs.set_xlabel('Time')
axs.set_ylabel('Angle')
axs.legend()
fig.suptitle('Elbow Angle', y=0.95)
plt.show()
# %% ADJUST ROTATION-Y PHILOSOPHY

data, label = load_and_log('test_angles2_dataset.npy', 'test_angles2_labels.npy', no_weight_classification=False, wandb_enabled=False, wandb_project="SL_multiclass")
data = apply_lowpass_filter(data, cutoff=50, fs=2000.0, order=5)

def update_rotation_y(gyro_y, dt, rotation_y):
    # Update the rotation around the y-axis
    rotation_y += gyro_y * dt
    return rotation_y

def get_adjusted_gyro_data(gyro_x, gyro_z, rotation_y):
    # Calculate the adjusted gyroscope data considering the rotation around the y-axis
    cos_y = np.cos(rotation_y)
    sin_y = np.sin(rotation_y)
    
    # Adjusted gyroscope data
    adjusted_gyro_z = cos_y * gyro_z - sin_y * gyro_x
    adjusted_gyro_x = sin_y * gyro_z + cos_y * gyro_x
    
    return adjusted_gyro_x, adjusted_gyro_z

def calculate_abduction_angle(adjusted_gyro_x, adjusted_gyro_z, dt):
    # Calculate the abduction angle by integrating the adjusted gyroscope data
    abduction_angle = np.cumsum(adjusted_gyro_x) * dt
    return abduction_angle

# Example usage
rotation_y = 0  # Initial rotation around the y-axis
dt = 0.0005  # Time step (example value, should match your data's sampling rate)

# Lists to store adjusted gyro data and abduction angles
adjusted_gyro_x_data = []
adjusted_gyro_z_data = []
abduction_angles = []

gyro = integrate_gyroscope_imu_and_extract(data)
shoulder_add = gyro[0,8,:]
for index, value in enumerate(shoulder_add):
    shoulder_add[index] = value + (1.422*index)
    shoulder_add[index] = shoulder_add[index] * (-5.5770e-4)
shoulder_flex = gyro[0,6,:]
for index, value in enumerate(shoulder_flex):
    shoulder_flex[index] = value + (13.8979*index)
    shoulder_flex[index] = shoulder_flex[index] * (-5.4185e-4)

for i in range(data.shape[2]):
    gyro_x, gyro_y, gyro_z = data[0,15:18,i]
    
    # Update rotation around the y-axis
    rotation_y = update_rotation_y(gyro_y, dt, rotation_y)

    # THE IDEA IS TO USE THEìIS ROTATION_Y TO UNDERSTAND HOW MUCH WE'LL TAKE FROM Z-AXIS AND Y-AXIS ROTATION

    abduction_angles.append(shoulder_add[i]*(1-rotation_y) + shoulder_flex[i]*rotation_y)

abduction_angles = np.array(abduction_angles)
fig, axs = plt.subplots(figsize=(12, 6), sharex=True)
time = range(data.shape[2])
axs.plot(time, abduction_angles)
axs.set_ylabel(f'Angles values')
axs.grid(True)
axs.set_title('abduction')
fig.suptitle('Subplots of Data', y=0.92)
plt.show()


# %% LOAD DATA AND LOWPASS FILTER

'''
For now I skip the 'adjust y-rotation' philosophy
'''
data, label = load_and_log('single_positions_dataset.npy', 'single_positions_labels.npy', no_weight_classification=False, wandb_enabled=False, wandb_project="SL_multiclass")
data = convert_to_SI(data)
data = apply_lowpass_filter(data, cutoff=50, fs=2000.0, order=5)
print(data.shape)

# %% EXTRACT QUATERNIONS FROM IMU DATA

quat = extract_quaternions(data)
#quat, bias, bias_sigma = calculate_quaternions_imu_no_avg(data)
print(quat.shape)
#print(bias.shape)
#print(bias_sigma.shape)

# %% CALCULATE ROTATION MATRICES FROM QUATERNIONS

#quat = quat[:,2,:,:]

#adjusted_quat = np.array([adjust_quaternions_with_bias(q, b) for q, b in zip(quat, bias)])
#print(adjusted_quat.shape)
rotation_matrices = np.empty((quat.shape[0], quat.shape[1], quat.shape[2], 3, 3))
for i in range(quat.shape[0]):
    for j in range(quat.shape[1]):
        rotation_matrices[i, j] = quaternion_to_rotation_matrix(quat[i, j])

#rotation_matrices = np.array([quaternion_to_rotation_matrix(q) for q in quat])

print(rotation_matrices.shape)
# HUGE rotation matrices array: [movement, sensor, sample, 3x3]

def extract_angles_from_rot_matrix(rotation_matrices):
    num_movements, num_sensors, num_samples, _, _ = rotation_matrices.shape
    speeds = np.zeros((num_movements, num_sensors, num_samples, _))
    for m in range(num_movements):
        for s in range(num_sensors):
            # to optimize -> i don't need to calculate the speed for all of them
            speeds[m, s] = calculate_euler_angle_displacements(rotation_matrices[m, s])
    positions = integrate_speeds(speeds)
    angles = np.zeros((num_movements, num_samples, 5))
    for m in range(num_movements):
        for i in range(num_samples):
            angles[m, i, 0] = positions[m, 2, i, 2] # shoulder abduction 
            angles[m, i, 1] = positions[m, 2, i, 0] # shoulder flexion
            angles[m, i, 2] = positions[m, 2, i, 1] # shoulder rotation
            #R_rel = np.dot(rotation_matrices[m, 1, i, :, :].T, rotation_matrices[m, 2, i, :, :])
            #angles[m, i, 3] = np.cumsum(np.arctan2(R_rel[2, 1], R_rel[1, 1])) # elbow
            #angles[m, i, 3] = np.cumsum(speeds[m, 1, i, 0] - speeds[m, 2, i, 0]) # elbow
            angles[m, i, 3] = positions[m, 1, i, 0] - positions[m, 2, i, 0]# elbow
            angles[m, i, 4] = positions[m, 0, i, 1] # wrist
    return np.array(angles)

# %% CANATA

#euler_speeds = np.array([calculate_euler_angle_displacements(rot_matrices) for rot_matrices in rotation_matrices])
#euler_angles = integrate_angles(euler_speeds)

# PROBLEM: FIRST ANGLES ALL 0
euler_angles = extract_angles_from_rot_matrix(rotation_matrices)

# %% ADJUST BIAS 

# FOR DOING THE SAME I NEED TO STORE IN A LIST EVERY ANGLES AND THATS IT
# OTHER MOVEMENT, OTHER LIST

# for the one without sensors, delete second dimension
#angles = np.array([euler_angles[5, 2, :, 0], euler_angles[5, 2, :, 1], euler_angles[5, 2, :, 2]])
def bias_correction(euler_angles):
    bias = np.zeros(euler_angles.shape)
    trend = np.zeros(euler_angles.shape)
    desired_trend = np.zeros(euler_angles.shape)
    t = np.arange(len(euler_angles[0]))
    for m in range(euler_angles.shape[0]):
        for i in range(trend.shape[2]):
            bias[m,:,i] = high_pass_filter(euler_angles[m,:,i], cutoff=0.5, fs=2000.0, order=4)
            p = np.polyfit(t, euler_angles[m,:,i], 1)
            trend[m,:,i] = np.polyval(p, t)
            p = np.polyfit(t, bias[m,:,i], 1)
            desired_trend[m,:,i] = np.polyval(p, t)
        angles_corrected = euler_angles - (trend - desired_trend)
        for ang in angles_corrected:
            ang -= (ang[0]+ang[-1])/2
            ang[:,3] += 90
    return angles_corrected
    #ang -= ang[-1]

angles = bias_correction(euler_angles)

# %% PLOT

movement = 5
num_plots = 5
time = np.arange(1, len(euler_angles[0]) + 1)
fig, axs = plt.subplots(num_plots, 1, figsize=(10, 25), sharex=True, sharey=True)
label = ['Shoulder Abduction', 'Shoulder Flexion', 'Shoulder Rotation', 'Elbow', 'Wrist']
for i in range(num_plots):
    axs[i].plot(time, euler_angles[movement, :, i])
    axs[i].plot(time, angles[movement, :, i], label=label[i])
    #axs[i].plot(time, original_angles[i, :], label='Original')
    #axs[i].plot(time, euler_speeds[0,2,:,i], label='speed')
    #axs[i].plot(time, euler_second_way[movement, :, i], label='Second Way')
    #axs[i].plot(time, bias_sigma[1,:-1], label='Bias')
    #axs[i].plot(time, bias[1,:-1,i], label='Bias')
    #axs[i].plot(time, bias[1,:-1,i] + bias_sigma[1,:-1], label='Bias')
    axs[i].set_title(f'Angular Displacement around Axis {i}')
    axs[i].set_xlabel('Sample')
    axs[i].set_ylabel('Angle (degrees)')
    axs[i].grid(True)
    axs[i].legend()
plt.show()

#%% FLIP QUATERNIONS..

def unwrap_quaternions(quaternions, threshold=0.3, debug=True):
    unwrapped = np.array(quaternions)
    flips = 0
    for i in range(1, len(unwrapped)):
        q1 = unwrapped[i-1]
        q2 = unwrapped[i]
        
        # Check if any component has changed more than the threshold
        if np.any(np.abs(q1 - q2) > threshold):
            # Flip the sign of the current quaternion
            unwrapped[i] = -q2
            flips += 1
            if debug and flips <= 10:  # Print first 10 flips
                print(f"Flip at index {i}:")
                print(f"  Previous: {q1}")
                print(f"  Before flip: {q2}")
                print(f"  After flip: {-q2}")
    
    if debug:
        print(f"Total flips: {flips}")
    
    return unwrapped

# Usage example:
# Assuming your quaternions are stored in a numpy array called 'quaternions'
unwrapped_quat = unwrap_quaternions(quat[0, 2, :, :])

#%% QUATERNION DEBUGGING

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# MOVEMENT(0 -> 5) AND SENSOR(0 -> 2)
movement = 0
sensor = 0
quaternions = quat[movement, sensor]
quaternions = quaternions / np.linalg.norm(quaternions, axis=1)[:, np.newaxis]  # Normalize quaternions

def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
    ])

def plot_frame(ax, R, origin):
    for i, color in enumerate(['r', 'g', 'b']):
        ax.quiver(origin[0], origin[1], origin[2], 
                  R[0, i], R[1, i], R[2, i], 
                  color=color, length=0.1)

# Assuming you have your quaternions in a list called 'quaternions'
# and you want to plot every 1000th quaternion
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(0, 30000, 1000):
    R = quat_to_rotmat(quaternions[i])
    origin = np.array([i/1000, 0, 0])  # Spread frames along x-axis
    plot_frame(ax, R, origin)

ax.set_xlim(0, 30)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
plt.show()

num_plots = 4
time = np.arange(1, len(data[0,0]) + 1)
fig, axs = plt.subplots(num_plots, 1, figsize=(10, 15), sharex=True, sharey=True)
for i in range(num_plots):
    axs[i].plot(time, quat[movement, sensor, :, i])
    #axs[i].plot(time, bias_sigma[1,:-1], label='Bias')
    #axs[i].plot(time, bias[1,:-1,i], label='Bias')
    #axs[i].plot(time, bias[1,:-1,i] + bias_sigma[1,:-1], label='Bias')
    axs[i].set_title(f'Angular Displacement around Axis {i}')
    axs[i].set_xlabel('Sample')
    axs[i].set_ylabel('Angle (degrees)')
    axs[i].grid(True)
    axs[i].legend()
plt.show()

num_plots = 6
time = np.arange(1, len(data[0,0]) + 1)
fig, axs = plt.subplots(num_plots, 1, figsize=(10, 15), sharex=True, sharey=True)
for i in range(num_plots):
    axs[i].plot(time, data[movement, (sensor * 6) + i, :])
    #axs[i].plot(time, bias_sigma[1,:-1], label='Bias')
    #axs[i].plot(time, bias[1,:-1,i], label='Bias')
    #axs[i].plot(time, bias[1,:-1,i] + bias_sigma[1,:-1], label='Bias')
    axs[i].set_title(f'Angular Displacement around Axis {i}')
    axs[i].set_xlabel('Sample')
    axs[i].set_ylabel('Angle (degrees)')
    axs[i].grid(True)
    axs[i].legend()
plt.show()
