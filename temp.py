#%% FEATURE EXTRACTION NOT USED

def rotation_matrix_to_euler_new(R, order='xyz'):
    """
    Convert a rotation matrix to Euler angles (in radians).
    
    :param R: 3x3 rotation matrix
    :param order: Rotation order of euler angles, default is 'xyz'
    :return: Euler angles [x, y, z] in radians
    """
    def is_gimbal_lock(R, tolerance=1e-7):
        return abs(abs(R[2, 0]) - 1) < tolerance
    if order == 'xyz':
        if is_gimbal_lock(R):
            # Gimbal lock case
            x = 0
            y = np.arcsin(R[2, 0])
            z = np.arctan2(-R[0, 1], R[1, 1])
        else:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arcsin(-R[2, 0])
            z = np.arctan2(R[1, 0], R[0, 0])
    else:
        raise ValueError("Only 'xyz' order is implemented in this example.")
    # Ensure angles are in the range [-pi, pi]
    return np.array([np.mod(angle + np.pi, 2 * np.pi) - np.pi for angle in [x, y, z]])


def get_feature_vector(windows):
    emg_vector = normalize_EMG_all_channels(extract_EMG_features(windows))
    quat = calculate_quaternions(windows)
    joint_angles = []
    for i in range(quat.shape[1]):
        R1 = R.from_quat(quat[0,i,:]).as_matrix()
        R2 = R.from_quat(quat[1,i,:]).as_matrix()
        R3 = R.from_quat(quat[2,i,:]).as_matrix()
        # to try
        #joint_angles.append(forward_kinematics(joint_angle_determination(R1, R2, R3),1,1))
        joint_angles.append(joint_angle_determination(R1, R2, R3))
    imu_vector = normalize_angles(joint_angles)
    return emg_vector, imu_vector

def calculate_quaternions(dataset):
    Ts = 0.0005  # Sampling period in seconds
    avg_gyr_upper_all = []
    avg_acc_upper_all = []
    avg_gyr_wrist_all = []
    avg_acc_wrist_all = []
    avg_gyr_arm_all = []
    avg_acc_arm_all = []

    # Loop over the dataset to calculate and store averages
    for data in dataset:
        avg_gyr_upper_all.append(np.mean(data[14:17, :], axis=1))
        avg_acc_upper_all.append(np.mean(data[11:14, :], axis=1))
        avg_gyr_wrist_all.append(np.mean(data[20:23, :], axis=1))
        avg_acc_wrist_all.append(np.mean(data[17:20, :], axis=1))
        avg_gyr_arm_all.append(np.mean(data[26:29, :], axis=1))
        avg_acc_arm_all.append(np.mean(data[23:26, :], axis=1))

    # Reshape the data to match the expected input format for offlineVQF
    gyr_upper = np.ascontiguousarray(np.array(avg_gyr_upper_all))
    acc_upper = np.ascontiguousarray(np.array(avg_acc_upper_all))
    gyr_wrist = np.ascontiguousarray(np.array(avg_gyr_wrist_all))
    acc_wrist = np.ascontiguousarray(np.array(avg_acc_wrist_all))
    gyr_arm = np.ascontiguousarray(np.array(avg_gyr_arm_all))
    acc_arm = np.ascontiguousarray(np.array(avg_acc_arm_all))

    # Compute the quaternions using the averaged data for each accelerometer
    out_upper = offlineVQF(acc_upper, gyr_upper, mag=None, Ts=Ts, params=None)
    out_wrist = offlineVQF(acc_wrist, gyr_wrist, mag=None, Ts=Ts, params=None)
    out_arm = offlineVQF(acc_arm, gyr_arm, mag=None, Ts=Ts, params=None)

    # Extract the quaternions and store them
    quaternion = np.array([out_upper['quat6D'], out_wrist['quat6D'], out_arm['quat6D']])
    return quaternion

def calculate_quaternions_imu(dataset):
    Ts = 0.0005  # Sampling period in seconds
    avg_gyr_upper_all = []
    avg_acc_upper_all = []
    avg_gyr_wrist_all = []
    avg_acc_wrist_all = []
    avg_gyr_arm_all = []
    avg_acc_arm_all = []

    # Loop over the dataset to calculate and store averages
    for data in dataset:
        avg_gyr_upper_all.append(np.mean(data[3:6, :], axis=1))
        avg_acc_upper_all.append(np.mean(data[:3, :], axis=1))
        avg_gyr_wrist_all.append(np.mean(data[9:12, :], axis=1))
        avg_acc_wrist_all.append(np.mean(data[6:9, :], axis=1))
        avg_gyr_arm_all.append(np.mean(data[15:18, :], axis=1))
        avg_acc_arm_all.append(np.mean(data[12:15, :], axis=1))

    # Reshape the data to match the expected input format for offlineVQF
    gyr_upper = np.ascontiguousarray(np.array(avg_gyr_upper_all))
    acc_upper = np.ascontiguousarray(np.array(avg_acc_upper_all))
    gyr_wrist = np.ascontiguousarray(np.array(avg_gyr_wrist_all))
    acc_wrist = np.ascontiguousarray(np.array(avg_acc_wrist_all))
    gyr_arm = np.ascontiguousarray(np.array(avg_gyr_arm_all))
    acc_arm = np.ascontiguousarray(np.array(avg_acc_arm_all))

    # Compute the quaternions using the averaged data for each accelerometer
    out_upper = offlineVQF(acc_upper, gyr_upper, mag=None, Ts=Ts, params=None)
    out_wrist = offlineVQF(acc_wrist, gyr_wrist, mag=None, Ts=Ts, params=None)
    out_arm = offlineVQF(acc_arm, gyr_arm, mag=None, Ts=Ts, params=None)

    # Extract the quaternions and store them
    quaternion = np.array([out_upper['quat6D'], out_wrist['quat6D'], out_arm['quat6D']])
    return quaternion

def joint_angle_determination(R1, R2, R3):
    # Initialize dictionary to store joint angles
    joint_angles = []

    # 1. Horizontal shoulder adduction/abduction
    x1 = R1[:3,0] #R1[:3, 0, i]  in realta era i,:3,0 ... i in prima pos per tutti
    y2 = -R2[:3,1] #-R2[:3, 1, i]
    z1 = -R3[:3,2] #-R1[:3, 2, i]
    
    # Projection to plane
    n1 = np.cross(z1, x1)
    n1_norm = n1 / np.linalg.norm(n1)
    y2_pro = y2 - np.dot(y2, n1_norm) * n1_norm
    
    # Preparation for angle calculation
    p1 = np.dot(y2_pro, x1)
    p2 = np.dot(y2_pro, z1)
    norm1 = np.linalg.norm(y2_pro)
    norm2 = np.linalg.norm(x1)
    norm3 = np.linalg.norm(z1)
    
    # Absolute adduction/abduction angle
    shoulder1_pos = np.degrees(np.arccos(p1 / (norm1 * norm2)))
    # Definition of positive and negative angles
    a_comp = np.degrees(np.arccos(p2 / (norm1 * norm3)))
    if a_comp < 90:
        shoulder1 = shoulder1_pos  # adduction
    else:
        shoulder1 = -shoulder1_pos  # abduction
    
    # 2. Shoulder flexion/extension
    y1 = -R1[:3,1] #-R1[:3, 1, i]
    y2 = -R2[:3,1] #-R2[:3, 1, i]
    p3 = np.dot(y2, y1)
    norm4 = np.linalg.norm(y2)
    norm5 = np.linalg.norm(y1)
    # Angle between y1 and y2
    shoulder2 = np.degrees(np.arccos(p3 / (norm4 * norm5))) - 90
    
    ## 3. Shoulder Rotation (internal and external)
    y1 = R1[:3,1] #R1[:3, 1, i]
    x2 = R2[:3,0] #R2[:3, 0, i]
    z2 = R2[:3,2] #R2[:3, 2, i]
    # Projection of y1 to the xz-plane of sensor 2
    n3 = np.cross(x2, z2)
    n3_norm = n3 / np.linalg.norm(n3)
    y1_pro = y1 - np.dot(y1, n3_norm) * n3_norm
    # Preperation for angle calculation
    p4 = np.dot(y1_pro, z2)
    p5 = np.dot(y1_pro, x2)
    norm6 = np.linalg.norm(y1_pro)
    norm7 = np.linalg.norm(z2)
    norm8 = np.linalg.norm(x2)
    # Absolute shoulder rotation angle
    shoulder3_pos = np.degrees(np.arccos(p4 / (norm6 * norm7)))
    # Definition of positive and negative angles
    a_comp = np.degrees(np.arccos(p5 / (norm6 * norm8)))
    if a_comp < 90:
        shoulder3 = -shoulder3_pos  # external rotation
    else:
        shoulder3 = shoulder3_pos  # internal rotation

    ## 4. Elbow flexion angle
    y2 = -R2[:3,1] #-R2[:3, 1, i]
    y3 = -R3[:3,1] #-R3[:3, 1, i]
    p6 = np.dot(y2, y3)
    norm9 = np.linalg.norm(y2)
    norm10 = np.linalg.norm(y3)
    # Absolute elbow flexion angle
    elbow_korr = np.degrees(np.arccos(p6 / (norm9 * norm10)))
    elbow = elbow_korr * (-1)

    ## 5. Pronation/Supination
    z2 = -R2[:3,2] #-R2[:3, 2, i]
    z3 = -R3[:3,2] #-R3[:3, 2, i]
    x3 = R3[:3,0] #R3[:3, 0, i]
    p7 = np.dot(z2, z3)
    p8 = np.dot(z2, x3)
    norm11 = np.linalg.norm(z2)
    norm12 = np.linalg.norm(z3)
    norm13 = np.linalg.norm(x3)
    # Absolute Pronation/Supination angle
    pronation_pos = np.degrees(np.arccos(p7 / (norm11 * norm12)))
    # Definition of positive and negative angles
    a_comp = np.degrees(np.arccos(p8 / (norm12 * norm13)))
    if a_comp < 90:
        pronation = pronation_pos  # supination
    else:
        pronation = pronation_pos  # pronation
    joint_angles = np.array([shoulder1, shoulder2, shoulder3, elbow, pronation])

    return joint_angles.T

def forward_kinematics(joint_angles, L1, L2):
    shoulder1 = joint_angles[0]
    shoulder2 = joint_angles[1]
    shoulder3 = joint_angles[2]
    elbow = joint_angles[3]
    pronation = joint_angles[4]

    # Calculate the shoulder rotation matrix
    R_shoulder = np.dot(np.dot(rotation_matrix_z(shoulder3), rotation_matrix_y(shoulder2)), rotation_matrix_x(shoulder1))
    #print('R_shoulder: ', R_shoulder)
    # Calculate the elbow position relative to the shoulder
    P_elbow_wrt_shoulder = np.dot(R_shoulder, np.array([L1, 0, 0]))
    #print('P_elbow_wrt_shoulder: ', P_elbow_wrt_shoulder)

    # Calculate the forearm rotation matrix
    R_forearm = np.dot(rotation_matrix_x(pronation), rotation_matrix_y(elbow))
    #print('R_forearm: ', R_forearm)
    # Calculate the wrist position relative to the elbow
    P_wrist_wrt_elbow = np.dot(R_forearm, np.array([L2, 0, 0]))
    #print('P_wrist_wrt_elbow: ', P_wrist_wrt_elbow)
    # Calculate the wrist position relative to the shoulder
    P_wrist_wrt_shoulder = P_elbow_wrt_shoulder + P_wrist_wrt_elbow
    #print('P_wrist_wrt_shoulder: ', P_wrist_wrt_shoulder)
    #input('Press Enter to continue...')
    #print()
    #print('-----------------------------------')
    #print()
    return np.array([shoulder1, shoulder2, shoulder3, elbow, pronation]), np.array([P_wrist_wrt_shoulder, P_elbow_wrt_shoulder])

# Define the elementary rotation matrices
def rotation_matrix_x(theta):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def rotation_matrix_y(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def rotation_matrix_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

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
def high_pass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data, axis=0)
def integrate_speeds(speeds):
    for i in range(0, len(speeds), 59):
        speeds[i:i+59] = np.cumsum(speeds[i:i+59], axis=0)
        np.cumsum(speeds, axis=2)
    return speeds

def quaternion_to_rotation_matrix_new(q):
    """
    Convert a quaternion to a rotation matrix, ensuring consistency.
    
    :param q: Quaternion [w, x, y, z]
    :return: 3x3 rotation matrix
    """
    #print(np.array(q).shape)
    #print(np.array(q))
    Rot_matrix = []
    for quat in q:
        w, x, y, z = quat
        # Normalize the quaternion
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        w, x, y, z = quat / norm
        # Ensure w is positive to maintain consistency
        if w < 0:
            w, x, y, z = -w, -x, -y, -z
        # Compute rotation matrix
        R = np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
            [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
        Rot_matrix.append(R)
    return np.array(Rot_matrix)

#%% OLD WAY
import numpy as np
from scipy.spatial.transform import Rotation as R

def correct_euler_angles(euler_array):
    """
    Corrects sudden 360-degree drops in Euler angles.

    Parameters:
    euler_array (np.ndarray): A numpy array of shape (num_acquisitions, num_sensors, euler_angles)

    Returns:
    np.ndarray: A numpy array with corrected Euler angles.
    """
    corrected_euler = euler_array.copy()
    for i in range(1, corrected_euler.shape[0]):  # Start from the second acquisition
        for j in range(corrected_euler.shape[1]):  # Iterate over each sensor
            for k in range(corrected_euler.shape[2]):  # Iterate over each Euler angle (roll, pitch, yaw)
                if i % 59 == 0:  # Skip the first sample of each movement
                    continue
                else:
                    # Calculate the difference between the current and previous angle
                    diff = corrected_euler[i, j, k] - corrected_euler[i - 1, j, k]
                    # Check if there's a drop of about 360 degrees and correct it
                    if diff > 180:
                        corrected_euler[i, j, k] -= 360
                    elif diff < -180:
                        corrected_euler[i, j, k] += 360
    return corrected_euler
def integrate_speeds(speeds):
    for i in range(0, len(speeds), 59):
        speeds[i:i+59] = np.cumsum(speeds[i:i+59], axis=0)
        np.cumsum(speeds, axis=2)
    return speeds
def create_windows(dataset, window_length, overlap):
    # window length in this case means the number of samples per window
    # overlap is the number of samples that overlap between windows
    windows = []
    windows_same_movement = []
    # Check if the number of samples per movement is divisible by the window length
    if dataset[0].shape[1] % window_length == 0:
        for movement in dataset:
            num_samples = movement.shape[1]
            num_windows = (num_samples - window_length) // (window_length - overlap) + 1            
            for i in range(num_windows):
                start_idx = i * (window_length - overlap)
                end_idx = start_idx + window_length
                window = movement[:, start_idx:end_idx]
                windows_same_movement.append(window)
                #windows.append(window)
            windows.append(windows_same_movement)
            windows_same_movement = []
    else:
        print("ERROR: The number of samples per movement is not divisible by the window length. Please adjust the window length accordingly.")
    return np.array(windows)
# NEW WAY
dataset_name='prova_senza5_dataset.npy'
label_name='prova_senza5_labels.npy'
num_classes=6
data, label = load_and_log(dataset_name, label_name, no_weight_classification=False, wandb_enabled=False, wandb_project="SL_multiclass")
data[:,:11,:] = notch_filter(bandpass_filter(highpass_filter(data[:,:11,:], 0.5), 0.5, 100), 50, 30)
data = convert_to_SI(data)
windows = create_windows(data, 200, 100)
if len(label) < (len(windows)*len(windows[0])):
    labels = np.repeat(label, len(windows)*len(windows[0]) / len(data))
    #labels = np.repeat(label, len(windows) / len(data))
    label = labels
num_samples = label.shape[0]
one_hot_matrix = np.zeros((num_samples, num_classes))
label = label.astype(int)
for i in range(num_samples):
    index = label[i]
    one_hot_matrix[i, index] = 1.
label = one_hot_matrix

quat = extract_quaternions_new(np.array(windows)) # mov(windows), channels, samples
quat_shape0, quat_shape1 = quat.shape[:2]
quat = quat.reshape(quat_shape0*quat_shape1, 3, 200, 4)
rotation_matrices = np.array([[R.from_quat(sensors).as_matrix() for sensors in windows] for windows in quat])

num_movements, num_sensors = rotation_matrices.shape[:2]
averaged_rotation_matrices = np.mean(rotation_matrices, axis=2)  # Average across the samples
speeds = np.zeros((num_movements, num_sensors, 3))
for m in range(num_movements):
    for s in range(num_sensors):
        speeds[m, s] = np.array(R.from_matrix(averaged_rotation_matrices[m, s]).as_euler('xyz', degrees=True))
        #speeds[m, s] = rotation_matrix_to_euler_new(averaged_rotation_matrices[m,s])#np.array(R.from_matrix(averaged_rotation_matrices[m, s]).as_euler('xyz', degrees=True))
#speeds = correct_euler_angles(speeds)
#positions = integrate_speeds(speeds)
positions = correct_euler_angles(speeds)#(positions)
angles = np.zeros((num_movements, 5))
for m in range(num_movements):
    angles[m, 0] = positions[m, 2, 2]  # shoulder abduction 
    angles[m, 1] = positions[m, 2, 0]  # shoulder flexion
    angles[m, 2] = positions[m, 2, 1]  # shoulder rotation
    angles[m, 3] = positions[m, 1, 0] - positions[m, 2, 0]  # elbow
    angles[m, 4] = positions[m, 0, 1]  # wrist
num_windows_same_movement = 59
first_angles = np.empty((int(num_movements/num_windows_same_movement),5))
last_angles = np.empty((int(num_movements/num_windows_same_movement),5))
j=0
for i in range(0, len(angles), num_windows_same_movement):
    first_angles[j] = angles[i]
    last_angles[j] = angles[i + num_windows_same_movement-1]
    j += 1
num_angles = angles.shape[1]
if num_angles == 9:
    calibration_angles = [0, 0, 0, 90, 0, 0, 0, 0, 0] # config.calibration_angles
elif num_angles == 5:
    calibration_angles = [0, 0, 0, 90, 0]
rest_angles = []
for i in range(len(first_angles)):
    if int(labels[i]) != 5: # 5 is the label for rest
        diff = calibration_angles - first_angles[i]
        angles[i*num_windows_same_movement : (i+1)*num_windows_same_movement] += diff
    else:
        # I take the last value of the angles of the previous window and I sum it to all the sample of the rest window
        diff = last_angles[i-1]
        angles[i*num_windows_same_movement : (i+1)*num_windows_same_movement] += diff
        rest_angles.append(angles[i*num_windows_same_movement : (i+1)*num_windows_same_movement])

def normalize_angles(imu_dataset):
    # Iterate over each window in the list
    for window in imu_dataset:
        # Update the window with the normalized values
        window[:] = window / 180.0
    return imu_dataset

#old_angles = normalize_angles(old_angles)
#new_angles = normalize_angles(new_angles)

new_angles = angles
# PLOT COMPARISON OF OLD AND NEW EULER ANGLES OVERLAPPED
import matplotlib.pyplot as plt
j = 0
for t in range(len(rest_angles)):
    num_plots = 5
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 18), sharex=True, sharey=True)
    time = range(quat_shape1)
    ang = rest_angles[j]
    for i in range(num_plots):
        axs[i].plot(time, ang[:, i], label='New')
        axs[i].legend()
        axs[i].grid(True)
    j+=1
    plt.show()

#old_angles = normalize_angles(old_angles)
#new_angles = normalize_angles(new_angles)

# PLOT COMPARISON OF OLD AND NEW EULER ANGLES OVERLAPPED
import matplotlib.pyplot as plt
j = 0
for t in range(10):
    num_plots = 5
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 18), sharex=True, sharey=True)
    time = range(quat_shape1)
    for i in range(num_plots):
        start = j*quat_shape1
        axs[i].plot(time, new_angles[start:start+quat_shape1, i], label='New')
        axs[i].legend()
        axs[i].grid(True)
    j+=1
    plt.show()

import matplotlib.pyplot as plt
j = 10
for t in range(10):
    num_plots = 5
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 18), sharex=True, sharey=True)
    time = range(quat_shape1)
    for i in range(num_plots):
        start = j*quat_shape1
        axs[i].plot(time, new_angles[start:start+quat_shape1, i], label='New')
        axs[i].legend()
        axs[i].grid(True)
    j+=1
    plt.show()
j = 20
for t in range(10):
    num_plots = 5
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 18), sharex=True, sharey=True)
    time = range(quat_shape1)
    for i in range(num_plots):
        start = j*quat_shape1
        axs[i].plot(time, new_angles[start:start+quat_shape1, i], label='New')
        axs[i].legend()
        axs[i].grid(True)
    j+=1
    plt.show()

import matplotlib.pyplot as plt
j = 30
for t in range(10):
    num_plots = 5
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 18), sharex=True, sharey=True)
    time = range(quat_shape1)
    for i in range(num_plots):
        start = j*quat_shape1
        axs[i].plot(time, new_angles[start:start+quat_shape1, i], label='New')
        axs[i].legend()
        axs[i].grid(True)
    j+=1
    plt.show()
j = 40
for t in range(10):
    num_plots = 5
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 18), sharex=True, sharey=True)
    time = range(quat_shape1)
    for i in range(num_plots):
        start = j*quat_shape1
        axs[i].plot(time, new_angles[start:start+quat_shape1, i], label='New')
        axs[i].legend()
        axs[i].grid(True)
    j+=1
    plt.show()

import matplotlib.pyplot as plt
j = 50
for t in range(10):
    num_plots = 5
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 18), sharex=True, sharey=True)
    time = range(quat_shape1)
    for i in range(num_plots):
        start = j*quat_shape1
        axs[i].plot(time, new_angles[start:start+quat_shape1, i], label='New')
        axs[i].legend()
        axs[i].grid(True)
    j+=1
    plt.show()
j = 60
for t in range(10):
    num_plots = 5
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 18), sharex=True, sharey=True)
    time = range(quat_shape1)
    for i in range(num_plots):
        start = j*quat_shape1
        axs[i].plot(time, new_angles[start:start+quat_shape1, i], label='New')
        axs[i].legend()
        axs[i].grid(True)
    j+=1
    plt.show()

import matplotlib.pyplot as plt
j = 70
for t in range(10):
    num_plots = 5
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 18), sharex=True, sharey=True)
    time = range(quat_shape1)
    for i in range(num_plots):
        start = j*quat_shape1
        axs[i].plot(time, new_angles[start:start+quat_shape1, i], label='New')
        axs[i].legend()
        axs[i].grid(True)
    j+=1
    plt.show()
j = 80
for t in range(10):
    num_plots = 5
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 18), sharex=True, sharey=True)
    time = range(quat_shape1)
    for i in range(num_plots):
        start = j*quat_shape1
        axs[i].plot(time, new_angles[start:start+quat_shape1, i], label='New')
        axs[i].legend()
        axs[i].grid(True)
    j+=1
    plt.show()

import matplotlib.pyplot as plt
j = 90
for t in range(10):
    num_plots = 5
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 18), sharex=True, sharey=True)
    time = range(quat_shape1)
    for i in range(num_plots):
        start = j*quat_shape1
        axs[i].plot(time, new_angles[start:start+quat_shape1, i], label='New')
        axs[i].legend()
        axs[i].grid(True)
    j+=1
    plt.show()
j = 100
for t in range(10):
    num_plots = 5
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 18), sharex=True, sharey=True)
    time = range(quat_shape1)
    for i in range(num_plots):
        start = j*quat_shape1
        axs[i].plot(time, new_angles[start:start+quat_shape1, i], label='New')
        axs[i].legend()
        axs[i].grid(True)
    j+=1
    plt.show()

import matplotlib.pyplot as plt
j = 110
for t in range(10):
    num_plots = 5
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 18), sharex=True, sharey=True)
    time = range(quat_shape1)
    for i in range(num_plots):
        start = j*quat_shape1
        axs[i].plot(time, new_angles[start:start+quat_shape1, i], label='New')
        axs[i].legend()
        axs[i].grid(True)
    j+=1
    plt.show()

#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vqf import offlineVQF
from pathlib import Path
from numpy.linalg import norm
# Function to concatenate CSV files and convert to NumPy array
def concatenate_csv(files):
    data_frames = []
    # Read each CSV file and append its contents to the list
    for file in files:
        df = pd.read_csv(file, header=None)#, low_memory=False)
        #df[993927] = pd.to_numeric(df[993927], errors='coerce').astype(np.float32)
        data_frames.append(df)
    # Concatenate the CSV dataframes vertically
    concatenated_data = pd.concat(data_frames, axis=1)
    # Convert to NumPy array
    concatenated_array = concatenated_data.to_numpy()
    return concatenated_array
def extract_and_store_classes(data, time_instants, sample_frequency=2000, fixed_segment_length=6000):
    # Initialize 3D array to store time series segments
    num_movements = len(time_instants)
    data_classes = []
    
    for i, instant in enumerate(time_instants):
        # Calculate start and end indices for taking 8000 samples
        start_index = int(instant * sample_frequency)
        end_index = min(start_index + fixed_segment_length, data.shape[1])
        
        # If the segment is less than 8000 samples, pad with zeros
        if end_index - start_index < fixed_segment_length:
            print("ERROR") 
        
        # Take 8000 samples from start_index to end_index
        data_classes.append(data[:, start_index:end_index])
        print(data_classes[i].shape)

        #
        # NEXT LINE IS JUST TO ADD THE CROPPED DATA AND ASSIGN THEM THE REST LABEL
        #
        #data_classes.append(data[:, end_index:end_index+fixed_segment_length])
      
    return data_classes
data_list = []
label_list = []
for i in range(1,7):
    directory_path = Path('C:/Users/claud/Desktop/LocoD/SavedData/DataCSV')
    csv_files = list(directory_path.glob(f'**/signal{i}.csv'))
    data = pd.read_csv(csv_files[0], header=None).to_numpy()
    label_directory_path = Path('C:/Users/claud/Desktop/LocoD/SavedData/LabelCSV')
    label_files = list(label_directory_path.glob(f'**/label{i}.csv'))
    label = pd.read_csv(label_files[0], header=None).to_numpy()
    data_list.append(extract_and_store_classes(data, label[1, :]))
    label_list.append(np.array(label[0, :]))
data_list = np.array(data_list)
label_list = np.array(label_list)

import numpy as np
from scipy.spatial.transform import Rotation as R

def correct_euler_angles(euler_array):
    """
    Corrects sudden 360-degree drops in Euler angles.

    Parameters:
    euler_array (np.ndarray): A numpy array of shape (num_acquisitions, num_sensors, euler_angles)

    Returns:
    np.ndarray: A numpy array with corrected Euler angles.
    """
    corrected_euler = euler_array.copy()
    for i in range(1, corrected_euler.shape[0]):  # Start from the second acquisition
        for j in range(corrected_euler.shape[1]):  # Iterate over each sensor
            for k in range(corrected_euler.shape[2]):  # Iterate over each Euler angle (roll, pitch, yaw)
                if i % 59 == 0:  # Skip the first sample of each movement
                    continue
                else:
                    # Calculate the difference between the current and previous angle
                    diff = corrected_euler[i, j, k] - corrected_euler[i - 1, j, k]
                    # Check if there's a drop of about 360 degrees and correct it
                    if diff > 180:
                        corrected_euler[i, j, k] -= 360
                    elif diff < -180:
                        corrected_euler[i, j, k] += 360
    return corrected_euler
def integrate_speeds(speeds):
    for i in range(0, len(speeds), 59):
        speeds[i:i+59] = np.cumsum(speeds[i:i+59], axis=0)
        np.cumsum(speeds, axis=2)
    return speeds
def create_windows(dataset, window_length, overlap):
    # window length in this case means the number of samples per window
    # overlap is the number of samples that overlap between windows
    windows = []
    windows_same_movement = []
    # Check if the number of samples per movement is divisible by the window length
    if dataset[0].shape[1] % window_length == 0:
        for movement in dataset:
            num_samples = movement.shape[1]
            num_windows = (num_samples - window_length) // (window_length - overlap) + 1            
            for i in range(num_windows):
                start_idx = i * (window_length - overlap)
                end_idx = start_idx + window_length
                window = movement[:, start_idx:end_idx]
                windows_same_movement.append(window)
                #windows.append(window)
            windows.append(windows_same_movement)
            windows_same_movement = []
    else:
        print("ERROR: The number of samples per movement is not divisible by the window length. Please adjust the window length accordingly.")
    return np.array(windows)
# NEW WAY
def pipeline_prova(data, label):
    num_classes=6
    #data, label
    data[:,:11,:] = notch_filter(bandpass_filter(highpass_filter(data[:,:11,:], 0.5), 0.5, 100), 50, 30)
    data = convert_to_SI(data)
    windows = create_windows(data[:,:,:], 200, 100)
    if len(label) < (len(windows)*len(windows[0])):
        labels = np.repeat(label, len(windows)*len(windows[0]) / len(data))
        #labels = np.repeat(label, len(windows) / len(data))
        label = labels
    num_samples = label.shape[0]
    one_hot_matrix = np.zeros((num_samples, num_classes))
    label = label.astype(int)
    for i in range(num_samples):
        index = label[i]
        one_hot_matrix[i, index - 1] = 1.
    label = one_hot_matrix
    quat = extract_quaternions_new(np.array(windows)) # mov(windows), channels, samples
    quat_shape0, quat_shape1 = quat.shape[:2]
    quat = quat.reshape(quat_shape0*quat_shape1, 3, 200, 4)
    rotation_matrices = np.array([[[quaternion_to_rotation_matrix(sensor) for sensor in windows] for windows in movement] for movement in quat])

    num_movements, num_sensors = rotation_matrices.shape[:2]
    averaged_rotation_matrices = np.mean(rotation_matrices, axis=2)  # Average across the samples
    speeds = np.zeros((num_movements, num_sensors, 3))
    for m in range(num_movements):
        for s in range(num_sensors):
            speeds[m, s] = np.array(R.from_matrix(averaged_rotation_matrices[m, s]).as_euler('xyz', degrees=True))
            #speeds[m, s] = rotation_matrix_to_euler_new(averaged_rotation_matrices[m,s])#np.array(R.from_matrix(averaged_rotation_matrices[m, s]).as_euler('xyz', degrees=True))
    #speeds = correct_euler_angles(speeds)
    first_angles = np.empty((20,5))
    #last_angles = np.empty((20,5))
    j=0
    for i in range(0, len(speeds), 59):
        first_angles[j] = speeds[i]
        #last_angles[j] = new_angles[i+58]
        j += 1
    calibration_angles = [0, 0, 0, 90, 0]
    for i in range(len(first_angles)):
        diff = calibration_angles - first_angles[i]
        speeds[i*59 : (i+1)*59] += diff
    positions = speeds
    positions = correct_euler_angles(positions)
    angles = np.zeros((num_movements, 5))
    for m in range(num_movements):
        angles[m, 0] = positions[m, 2, 2]  # shoulder abduction 
        angles[m, 1] = positions[m, 2, 0]  # shoulder flexion
        angles[m, 2] = positions[m, 2, 1]  # shoulder rotation
        angles[m, 3] = positions[m, 1, 0] - positions[m, 2, 0]  # elbow
        angles[m, 4] = positions[m, 0, 1]  # wrist
    new_angles = angles
    return new_angles

new_angles_prova = []
for data, label in zip(data_list, label_list):
    new_angles_prova.append(pipeline_prova(data,label))
new_angles_prova = np.array(new_angles_prova)

for angles in new_angles_prova:
    # PLOT COMPARISON OF OLD AND NEW EULER ANGLES OVERLAPPED
    import matplotlib.pyplot as plt
    j = 5
    for t in range(10):
        num_plots = 5
        fig, axs = plt.subplots(num_plots, 1, figsize=(12, 9), sharex=True, sharey=True)
        time = range(59)
        for i in range(num_plots):
            start = j*59
            axs[i].plot(time, angles[start:start+59, i], label='New')
            axs[i].legend()
            axs[i].grid(True)
        j+=1
        plt.show()

#%% FOR ONLINE 

class IMUProcessor:
    def __init__(self):
        # Initialize VQF objects for the three sensors
        self.vqf_wrist = None
        self.vqf_upper = None
        self.vqf_arm = None

    def reset_vqf(self):
        # Reset the VQF objects (for a new movement)
        self.vqf_wrist = VQF(gyrTs=0.0005, accTs=0.0005)
        self.vqf_upper = VQF(gyrTs=0.0005, accTs=0.0005)
        self.vqf_arm = VQF(gyrTs=0.0005, accTs=0.0005)

    def extract_quaternions_online(self, window, is_consecutive):
        if not is_consecutive or self.vqf_wrist is None:
            # If the window is not consecutive or VQF objects are not initialized, reset them
            self.reset_vqf()
        
        quats = np.zeros((window.shape[0], 3, window.shape[2], 4))  # Assuming window shape (29, 200)

        for idx, data in enumerate(window):
            for i in range(window.shape[2]):  # Iterate over each timestamp in the window
                # Update VQF for each sensor
                self.vqf_wrist.update(gyr=np.ascontiguousarray(data[20:23, i]), acc=np.ascontiguousarray(data[17:20, i]))
                self.vqf_upper.update(gyr=np.ascontiguousarray(data[14:17, i]), acc=np.ascontiguousarray(data[11:14, i]))
                self.vqf_arm.update(gyr=np.ascontiguousarray(data[26:29, i]), acc=np.ascontiguousarray(data[23:26, i]))
                
                # Store quaternions for each sensor
                quats[idx, 0, i] = self.vqf_wrist.getQuat6D()
                quats[idx, 1, i] = self.vqf_upper.getQuat6D()
                quats[idx, 2, i] = self.vqf_arm.getQuat6D()

        return quats

# Example usage
imu_processor = IMUProcessor()

def process_online_window(window, is_consecutive):
    # Step 1: Extract quaternions from the current window
    quats = imu_processor.extract_quaternions_online(window, is_consecutive)
    
    # Step 2: Convert quaternions to rotation matrices
    quat_shape0, quat_shape1 = quats.shape[:2]
    rotation_matrices = np.array([[quaternion_to_rotation_matrix(q) for q in samples] for samples in quats.reshape(quat_shape0 * quat_shape1, 3, 200, 4)])
    
    # Step 3: Extract angles from rotation matrices
    euler_angles = extract_angles_from_rot_matrix(rotation_matrices)
    
    # Step 4: Normalize angles for further processing
    imu_vector = normalize_angles(euler_angles)
    
    return imu_vector
