import numpy as np
from scipy.spatial.transform import Rotation as R
from vqf import offlineVQF, VQF
from .preprocessing_and_normalization import normalize_EMG_all_channels, normalize_angles
from scipy.signal import butter, filtfilt

previous_label = np.array([0, 0, 0, 0, 1])
previous_angles = np.zeros((5))
previous_pose = np.zeros((3,3))
diff = np.zeros((5))
vqf_w = VQF(gyrTs = 0.0005, accTs = 0.0005)
vqf_u = VQF(gyrTs = 0.0005, accTs = 0.0005)
vqf_a = VQF(gyrTs = 0.0005, accTs = 0.0005)
num_emg_channels = 9

def extract_online_quaternions(window, new_vqf, freq = None):
    global vqf_w
    global vqf_u
    global vqf_a
    num_samples = window.shape[1]
    if freq is not None and freq != 0:
        Ts = 1/freq
    else:
        Ts = 0.0005
    if new_vqf:
        vqf_w = VQF(gyrTs = Ts, accTs = Ts)
        vqf_u = VQF(gyrTs = Ts, accTs = Ts)
        vqf_a = VQF(gyrTs = Ts, accTs = Ts)
    quats = np.zeros((3, num_samples, 4))
    for i in range(num_samples):
        vqf_w.update(gyr=np.ascontiguousarray(window[num_emg_channels + 3: num_emg_channels + 6,i]), acc=np.ascontiguousarray(window[num_emg_channels: num_emg_channels + 3,i]))
        vqf_u.update(gyr=np.ascontiguousarray(window[num_emg_channels + 9: num_emg_channels + 12,i]), acc=np.ascontiguousarray(window[num_emg_channels + 6: num_emg_channels + 9,i]))
        vqf_a.update(gyr=np.ascontiguousarray(window[num_emg_channels + 15: num_emg_channels + 18,i]), acc=np.ascontiguousarray(window[num_emg_channels + 12: num_emg_channels + 15,i]))
        quats[0, i] = vqf_w.getQuat6D()
        quats[1, i] = vqf_u.getQuat6D()
        quats[2, i] = vqf_a.getQuat6D()
    return quats

def correct_euler_angles(euler_array):
    global previous_pose
    corrected_euler = euler_array.copy()
    for j in range(corrected_euler.shape[0]):  # Iterate over each sensor
        for k in range(corrected_euler.shape[1]):  # Iterate over each Euler angle (roll, pitch, yaw)
            # Calculate the difference between the current and previous angle
            diff = corrected_euler[j, k] - previous_pose[j, k]
            # Check if there's a drop of about 360 degrees and correct it
            if diff > 180:
                corrected_euler[j, k] -= 360
            elif diff < -180:
                corrected_euler[j, k] += 360
    previous_pose = corrected_euler
    return corrected_euler

def extract_angles_from_rot_matrix(rotation_matrice, first_window, label):
    global diff
    global previous_angles
    global previous_label
    num_sensors = len(rotation_matrice)
    averaged_rotation_matrix = np.mean(rotation_matrice, axis=1)  # Average across the samples
    angular_pose = np.zeros((num_sensors, 3))
    for s in range(num_sensors):
        angular_pose[s] = np.array(R.from_matrix(averaged_rotation_matrix[s]).as_euler('xyz', degrees=True))
    # I NEED PREVIOUS LOOP EULER ANGLE
    angular_pose = correct_euler_angles(angular_pose)
    angles = np.zeros((9)) # config.input_shape_imu
    angles[0] = angular_pose[2, 2]  # as shoulder abduction 
    angles[1] = angular_pose[2, 0]  # shoulder flexion
    angles[2] = angular_pose[2, 1]  # shoulder rotation
    angles[3] = angular_pose[1, 0] - angular_pose[2, 0]  # elbow
    angles[4] = angular_pose[0, 1]  # wrist
    # 
    angles[5] = angular_pose[1, 1]
    angles[6] = angular_pose[1, 2]
    angles[7] = angular_pose[0, 0]
    angles[8] = angular_pose[0, 2]
    num_angles = int(angles.shape[0])
    if label[0,4] != 1: # no rest, one hot encoded label
        if first_window:
            #last_angles = np.empty(int(num_angles))
            if num_angles == 9:
                calibration_angles = [0, 0, 0, 90, 0, 0, 0, 0, 0] # config.calibration_angles
            elif num_angles == 5:
                calibration_angles = [0, 0, 0, 90, 0]
            diff = calibration_angles - angles
            #last_angles[i] = new_angles[(i*num_windows_same_movement) + num_windows_same_movement-1]
            #new_angles[i*num_windows_same_movement : (i+1)*num_windows_same_movement] += last_angles[i] #ONLY FOR REST!!!
        angles += diff
    else:
        if not np.array_equal(label, previous_label):
            diff = previous_angles - angles
        angles += diff
    previous_angles = angles
    previous_label = label.copy()
    return angles

def get_online_feature_vector(window, label, first_window, freq, model_type):
    global previous_label
    if first_window: #label != previous_label: # first_window ?
        quat = extract_online_quaternions(window, True, freq) # (num_sensors, num_samples, 4)
    else:
        quat = extract_online_quaternions(window, False, freq)
    #previous_label = label
    # (sensors, samples, 3x3)
    rotation_matrix = np.array([R.from_quat(sensor).as_matrix() for sensor in quat])
    angles = extract_angles_from_rot_matrix(rotation_matrix, first_window, label) # label != previous ?
    imu_vector = normalize_angles(angles)
    emg_vector = extract_features(window[:num_emg_channels,:])
    return emg_vector, imu_vector

def extract_features(window):
    features_list = []
    num_channels = window.shape[0]  # Get the number of channels in the window
    for channel_idx in range(num_channels):
        channel_data = window[channel_idx, :]
        channel_features = []
        # Mean Absolute Value
        channel_features.append(np.mean(np.abs(channel_data)))
        # Number of Zero Crossings
        num_zero_crossings = np.sum(np.diff(np.sign(channel_data)) != 0)
        channel_features.append(num_zero_crossings)
        # Number of Slope Sign Changes
        slope_sign_changes = np.sum(np.diff(np.sign(np.diff(channel_data))) != 0)
        channel_features.append(slope_sign_changes)
        # Wave Length
        wave_length = np.sum(np.abs(np.diff(channel_data)))
        channel_features.append(wave_length)
        # Root Mean Square
        #channel_features.append(np.sqrt(np.mean(np.square(channel_data))))
        # Variance
        #channel_features.append(np.var(channel_data))
        # Maximum Value
        #channel_features.append(np.max(channel_data))
        features_list.append(channel_features)
    # Convert the features list to a numpy array and transpose it
    return np.array(features_list)