import numpy as np
from scipy.spatial.transform import Rotation as R
from vqf import offlineVQF, VQF
from cbpr_master_thesis.preprocessing_and_normalization import normalize_EMG_all_channels, normalize_angles
from scipy.signal import butter, filtfilt

num_emg_channels = 9

# Check if window lenght (number of samples per window) is an integer multiple of the number of samples per movement
# then create windows been sure that each window belongs to the same movement
def create_windows(dataset, window_length, overlap, labels):
    # window length in this case means the number of samples per window
    # overlap is the number of samples that overlap between windows
    windows = []
    windows_same_movement = []
    label_windows = []
    # Check if the number of samples per movement is divisible by the window length
    if dataset.shape[2] % window_length == 0:
        for movement, label in zip(dataset, labels):
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
            label_windows.append(np.full(num_windows, label))
    else:
        print("ERROR: The number of samples per movement is not divisible by the window length. Please adjust the window length accordingly.")
    return np.array(windows), np.array(label_windows)

def extract_quaternions_new(windows):
    # Initialize VQF objects for each sensor
    #vqf_wrist = VQF(gyrTs=0.0005, accTs=0.0005)
    #vqf_upper = VQF(gyrTs=0.0005, accTs=0.0005)
    #vqf_arm = VQF(gyrTs=0.0005, accTs=0.0005)
    # Initialize output array with the desired shape
    quaternions = []
    for window in windows:
        # Reset VQF objects at the start of each recording
        #vqf_wrist.resetState()
        #vqf_upper.resetState()
        #vqf_arm.resetState()
        vqf_wrist = VQF(gyrTs=0.0005, accTs=0.0005)
        vqf_upper = VQF(gyrTs=0.0005, accTs=0.0005)
        vqf_arm = VQF(gyrTs=0.0005, accTs=0.0005)
        quats = np.zeros((windows.shape[1], 3, windows.shape[3], 4))
        for idx, data in enumerate(window):
            for i in range(windows.shape[3]):
                vqf_wrist.update(gyr=np.ascontiguousarray(data[num_emg_channels + 3: num_emg_channels + 6,i]), acc=np.ascontiguousarray(data[num_emg_channels: num_emg_channels + 3,i]))
                vqf_upper.update(gyr=np.ascontiguousarray(data[num_emg_channels + 9: num_emg_channels + 12,i]), acc=np.ascontiguousarray(data[num_emg_channels + 6: num_emg_channels + 9,i]))
                vqf_arm.update(gyr=np.ascontiguousarray(data[num_emg_channels + 15: num_emg_channels + 18,i]), acc=np.ascontiguousarray(data[num_emg_channels + 12: num_emg_channels + 15,i]))
                quats[idx, 0, i] = vqf_wrist.getQuat6D()
                quats[idx, 1, i] = vqf_upper.getQuat6D()
                quats[idx, 2, i] = vqf_arm.getQuat6D()
        quaternions.append(quats)
    return np.array(quaternions)

def extract_quaternions(dataset):
    num_samples = dataset.shape[2]
    vqf_wrist = VQF(gyrTs = 0.0005, accTs = 0.0005)
    vqf_upper = VQF(gyrTs = 0.0005, accTs = 0.0005)
    vqf_arm = VQF(gyrTs = 0.0005, accTs = 0.0005)
    quats = np.zeros((dataset.shape[0], 3, num_samples, 4))
    for idx, data in enumerate(dataset):
        for i in range(num_samples):
            vqf_wrist.update(gyr=np.ascontiguousarray(data[14:17,i]), acc=np.ascontiguousarray(data[11:14,i]))
            vqf_upper.update(gyr=np.ascontiguousarray(data[20:23,i]), acc=np.ascontiguousarray(data[17:20,i]))
            vqf_arm.update(gyr=np.ascontiguousarray(data[26:29,i]), acc=np.ascontiguousarray(data[23:26,i]))
            quats[idx, 0, i] = vqf_wrist.getQuat6D()
            quats[idx, 1, i] = vqf_upper.getQuat6D()
            quats[idx, 2, i] = vqf_arm.getQuat6D()
    return quats

def correct_euler_angles(euler_array):
    corrected_euler = euler_array.copy()
    for i in range(1, corrected_euler.shape[0]):  # Start from the second acquisition
        for j in range(corrected_euler.shape[1]):  # Iterate over each sensor
            for k in range(corrected_euler.shape[2]):  # Iterate over each Euler angle (roll, pitch, yaw)
                # Calculate the difference between the current and previous angle
                diff = corrected_euler[i, j, k] - corrected_euler[i - 1, j, k]
                # Check if there's a drop of about 360 degrees and correct it
                if diff > 180:
                    corrected_euler[i, j, k] -= 360
                elif diff < -180:
                    corrected_euler[i, j, k] += 360
    return corrected_euler

def extract_angles_from_rot_matrix(rotation_matrices, num_windows_same_movement, label):
    num_windows, num_sensors = rotation_matrices.shape[:2]
    averaged_rotation_matrices = np.mean(rotation_matrices, axis=2)  # Average across the samples
    angular_pose = np.zeros((num_windows, num_sensors, 3))
    for m in range(num_windows):
        for s in range(num_sensors):
            angular_pose[m, s] = np.array(R.from_matrix(averaged_rotation_matrices[m, s]).as_euler('xyz', degrees=True))
    angular_pose = correct_euler_angles(angular_pose)
    angles = np.zeros((num_windows, 9)) # config.input_shape_imu
    for m in range(num_windows):
        angles[m, 0] = angular_pose[m, 2, 2]  # represent shoulder abduction 
        angles[m, 1] = angular_pose[m, 2, 0]  # shoulder flexion
        angles[m, 2] = angular_pose[m, 2, 1]  # shoulder rotation
        angles[m, 3] = angular_pose[m, 1, 0] - angular_pose[m, 2, 0]  # elbow
        angles[m, 4] = angular_pose[m, 0, 1]  # wrist
        # 
        angles[m, 5] = angular_pose[m, 1, 1]
        angles[m, 6] = angular_pose[m, 1, 2]
        angles[m, 7] = angular_pose[m, 0, 0]
        angles[m, 8] = angular_pose[m, 0, 2]
    num_angles = angles.shape[1]
    first_angles = np.empty((int(num_windows / num_windows_same_movement), num_angles))
    last_angles = np.empty((int(num_windows / num_windows_same_movement), num_angles))
    j=0
    for i in range(0, len(angles), num_windows_same_movement):
        first_angles[j] = angles[i]
        last_angles[j] = angles[i + num_windows_same_movement-1]
        j += 1
    if num_angles == 9:
        calibration_angles = [0, 0, 0, 90, 0, 0, 0, 0, 0] # config.calibration_angles
    elif num_angles == 5:
        calibration_angles = [0, 0, 0, 90, 0]
    label = label.reshape(int(num_windows/num_windows_same_movement), num_windows_same_movement, label.shape[1])
    for i in range(len(first_angles)):
        if label[i,0,5] != 1:
            diff = calibration_angles - first_angles[i]
            angles[i*num_windows_same_movement : (i+1)*num_windows_same_movement] += diff
    j = 0
    for i in range(0, len(angles), num_windows_same_movement):
        last_angles[j] = angles[i + num_windows_same_movement-1]
        j += 1
    for i in range(len(first_angles)):
        if label[i,0,5] == 1:# I take the last value of the angles of the previous window and I sum it to all the sample of the rest window
            diff = last_angles[i-1] - first_angles[i]
            angles[i*num_windows_same_movement : (i+1)*num_windows_same_movement] += diff
        #last_angles[i] = new_angles[(i*num_windows_same_movement) + num_windows_same_movement-1]
        #new_angles[i*num_windows_same_movement : (i+1)*num_windows_same_movement] += last_angles[i] #ONLY FOR REST!!!
    return angles

def get_new_feature_vector(windows, labels):
    movements, windows_same_movement, num_channels, num_samples = windows.shape
    #quat = extract_quaternions(np.array(windows)) # mov(windows), channels, samples
    quat = extract_quaternions_new(windows) # (movements, windows_same_movement, num_channels, num_samples, 4)
    # from (movements, windows_same_movement, sensors, samples, quaternions) to (windows, sensors, samples, quaternions)
    quat = quat.reshape(quat.shape[0]*quat.shape[1], quat.shape[2], quat.shape[3], quat.shape[4])
    # (windows, sensors, samples, 3x3)
    rotation_matrices = np.array([[R.from_quat(sensors).as_matrix() for sensors in windows] for windows in quat])
    angles = extract_angles_from_rot_matrix(rotation_matrices, windows_same_movement, labels)
    imu_vector = normalize_angles(angles)
    # qua devo aggiungere max emg
    emg_vector = normalize_EMG_all_channels(extract_EMG_features(windows.reshape(movements*windows_same_movement, num_channels, num_samples)))#, np.load('C:/Users/claud/Desktop/LocoD/SavedData/Dataset/max_data.npy'))
    return emg_vector, imu_vector

def extract_EMG_features(windows):
    all_features = []
    for window in windows:
        window_features = extract_features(window[:num_emg_channels,:])
        all_features.append(window_features)
    return all_features

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
