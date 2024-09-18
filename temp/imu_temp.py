import numpy as np
from vqf import VQF
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt

def high_pass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data, axis=0)

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
            vqf_wrist.update(gyr=np.ascontiguousarray(data[20:23,i]), acc=np.ascontiguousarray(data[17:20,i]))
            vqf_upper.update(gyr=np.ascontiguousarray(data[14:17,i]), acc=np.ascontiguousarray(data[11:14,i]))
            vqf_arm.update(gyr=np.ascontiguousarray(data[26:29,i]), acc=np.ascontiguousarray(data[23:26,i]))
            quats_wrist.append(vqf_wrist.getQuat6D())
            quats_upper.append(vqf_upper.getQuat6D())
            quats_arm.append(vqf_arm.getQuat6D())
        quat.append(np.array([quats_wrist, quats_upper, quats_arm]))
    return np.array(quat)

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
# I need to combine quat_to_rot, angles_from_rot
def new_quaternion_to_rotation_matrix(quat): # I call it one step before 
    # [sensors, samples, 4] [0-1-2, 100, 4]
    R1 = []
    R2 = []
    R3 = []

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
    return np.array(euler_angle_displacements)

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

def integrate_speeds(speeds):
    num_movements, num_sensors, _, num_angles = speeds.shape
    angles = np.zeros((num_movements, num_sensors, _, num_angles))
    for m in range(num_movements):
        for s in range(num_sensors):
            for i in range(num_angles):
                angles[m, s, :, i] = np.cumsum(speeds[m, s, :, i])
    return angles

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


#########################################
# ChatGPT

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt

def extract_quaternions(dataset):
    num_samples = dataset.shape[2]
    vqf_wrist = VQF(gyrTs=0.0005, accTs=0.0005)
    vqf_upper = VQF(gyrTs=0.0005, accTs=0.0005)
    vqf_arm = VQF(gyrTs=0.0005, accTs=0.0005)
    quats = np.zeros((dataset.shape[0], 3, num_samples, 4))
    
    for idx, data in enumerate(dataset):
        for i in range(num_samples):
            vqf_wrist.update(gyr=data[20:23, i], acc=data[17:20, i])
            vqf_upper.update(gyr=data[14:17, i], acc=data[11:14, i])
            vqf_arm.update(gyr=data[26:29, i], acc=data[23:26, i])
            quats[idx, 0, i] = vqf_wrist.getQuat6D()
            quats[idx, 1, i] = vqf_upper.getQuat6D()
            quats[idx, 2, i] = vqf_arm.getQuat6D()
    
    return quats

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
            vqf_wrist.update(gyr=np.ascontiguousarray(data[20:23,i]), acc=np.ascontiguousarray(data[17:20,i]))
            vqf_upper.update(gyr=np.ascontiguousarray(data[14:17,i]), acc=np.ascontiguousarray(data[11:14,i]))
            vqf_arm.update(gyr=np.ascontiguousarray(data[26:29,i]), acc=np.ascontiguousarray(data[23:26,i]))
            quats_wrist.append(vqf_wrist.getQuat6D())
            quats_upper.append(vqf_upper.getQuat6D())
            quats_arm.append(vqf_arm.getQuat6D())
        quat.append(np.array([quats_wrist, quats_upper, quats_arm]))
    return np.array(quat)

def quaternion_to_rotation_matrix(quat):
    return R.from_quat(quat).as_matrix()

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

def extract_angles_from_rot_matrix(rotation_matrices):
    num_movements, num_sensors, num_samples = rotation_matrices.shape[:3]
    speeds = np.zeros((num_movements, num_sensors, num_samples, 3))
    
    for m in range(num_movements):
        for s in range(num_sensors):
            speeds[m, s] = calculate_euler_angle_displacements(rotation_matrices[m, s])
    
    positions = integrate_speeds(speeds)
    angles = np.zeros((num_movements, num_samples, 5))
    
    for m in range(num_movements):
        for i in range(num_samples):
            angles[m, i, 0] = positions[m, 2, i, 2]  # shoulder abduction 
            angles[m, i, 1] = positions[m, 2, i, 0]  # shoulder flexion
            angles[m, i, 2] = positions[m, 2, i, 1]  # shoulder rotation
            angles[m, i, 3] = positions[m, 1, i, 0] - positions[m, 2, i, 0]  # elbow
            angles[m, i, 4] = positions[m, 0, i, 1]  # wrist
    
    return angles

def calculate_euler_angle_displacements(rotation_matrices, axis=None):
    if axis is None:
        num_samples = rotation_matrices.shape[0]
        euler_angle_displacements = np.zeros((num_samples, 3))
        
        for i in range(num_samples):
            if i == 0:
                R_rel = rotation_matrices[i] * 0.01
            else:
                R_rel = np.dot(rotation_matrices[i], np.linalg.inv(rotation_matrices[i-1]))
            euler_angle_displacements[i] = rotation_matrix_to_euler_angles(R_rel)
    else:
    
    return euler_angle_displacements

def rotation_matrix_to_euler_angles(rotation_matrix):
    r = R.from_matrix(rotation_matrix)
    return r.as_euler('xyz', degrees=True)

def integrate_speeds(speeds):
    return np.cumsum(speeds, axis=2)

def integrate_speeds(speeds):
    num_movements, num_sensors, num_angles = speeds.shape
    angles = np.zeros((num_movements, num_sensors, num_angles))
    for m in range(num_movements):
        for s in range(num_sensors):
            for i in range(num_angles):
                angles[m, s, i] = np.cumsum(speeds[m, s, i])
    return angles

def high_pass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data, axis=1)

def bias_correction(euler_angles):
    num_movements, num_samples, num_angles = euler_angles.shape
    bias = np.zeros_like(euler_angles)
    trend = np.zeros_like(euler_angles)
    desired_trend = np.zeros_like(euler_angles)
    t = np.arange(num_samples)
    
    for m in range(num_movements):
        for i in range(num_angles):
            bias[m, :, i] = high_pass_filter(euler_angles[m, :, i], cutoff=0.5, fs=2000.0, order=4)
            p = np.polyfit(t, euler_angles[m, :, i], 1)
            trend[m, :, i] = np.polyval(p, t)
            p = np.polyfit(t, bias[m, :, i], 1)
            desired_trend[m, :, i] = np.polyval(p, t)
    
    angles_corrected = euler_angles - (trend - desired_trend)
    
    for m in range(num_movements):
        for i in range(num_samples):
            angles_corrected[m, i] -= (angles_corrected[m, i, 0] + angles_corrected[m, i, -1]) / 2
            angles_corrected[m, i, 3] += 90
    
    return angles_corrected

# Example pipeline usage
def process_pipeline(dataset):
    quat = extract_quaternions(dataset)  # Extract quaternions
    rotation_matrices = np.array([[quaternion_to_rotation_matrix(q) for q in samples] for samples in quat])
    euler_angles = extract_angles_from_rot_matrix(rotation_matrices)  # Extract angles from rotation matrices
    angles = bias_correction(euler_angles)  # Correct bias in angles
    return angles

# Assuming 'windows' is your dataset with shape [movements, channels, samples]
# angles = process_pipeline(np.array(windows))
