from scipy.signal import butter, lfilter, freqz
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.signal import medfilt
from scipy.signal import wiener
from scipy.signal import hilbert
from scipy.signal import resample
from scipy.signal import decimate
from scipy.signal import detrend
from scipy.signal import correlate
from scipy.signal import spectrogram
from scipy.signal import welch
from scipy.signal import periodogram
from scipy.signal import csd

# %% OLD Idea

# KALMAN FILTER
from pyquaternion import Quaternion
class KalmanFilter:
    def __init__(self, sample_period, process_noise, measurement_noise, a=1, i=0, j=0, k=0):
        self.sample_period = sample_period
        self.q = Quaternion(a, i, j, k)
        self.P = np.eye(4) * 0.1  # Initial covariance
        self.Q = np.eye(4) * process_noise  # Process noise
        self.R = np.eye(3) * measurement_noise  # Measurement noise

    def update(self, gyroscope, accelerometer):
        # Prediction step
        q1, q2, q3, q4 = self.q.elements
        gx, gy, gz = gyroscope

        F = np.array([
            [1, -0.5*self.sample_period*gx, -0.5*self.sample_period*gy, -0.5*self.sample_period*gz],
            [0.5*self.sample_period*gx, 1, 0.5*self.sample_period*gz, -0.5*self.sample_period*gy],
            [0.5*self.sample_period*gy, -0.5*self.sample_period*gz, 1, 0.5*self.sample_period*gx],
            [0.5*self.sample_period*gz, 0.5*self.sample_period*gy, -0.5*self.sample_period*gx, 1]
        ])

        self.q = Quaternion(F @ self.q.elements)
        self.P = F @ self.P @ F.T + self.Q

        # Measurement update step
        q1, q2, q3, q4 = self.q.elements
        hx = np.array([2*(q2*q4 - q1*q3), 2*(q1*q2 + q3*q4), q1*q1 - q2*q2 - q3*q3 + q4*q4])
        z = accelerometer / np.linalg.norm(accelerometer)
        y = z - hx

        H = np.array([
            [-2*q3, 2*q4, -2*q1, 2*q2],
            [2*q2, 2*q1, 2*q4, 2*q3],
            [2*q1, -2*q2, -2*q3, 2*q4]
        ])

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        dq = K @ y
        dq_quat = Quaternion(1, *dq[:3])
        self.q = (self.q * dq_quat).normalised
        self.P = (np.eye(4) - K @ H) @ self.P

    def get_quaternion(self):
        return self.q

def calculate_quaternions_kalman(data):
    quat_kalman = []
    for data_move in data:
        #kalman1 = KalmanFilter(sample_period=0.0005, process_noise=1e-5, measurement_noise=10)
        #kalman2 = KalmanFilter(sample_period=0.0005, process_noise=1e-5, measurement_noise=10)
        kalman3 = KalmanFilter(sample_period=0.0005, process_noise=1e-5, measurement_noise=10, a=0.7071, i=0.7071, j=0, k=0)
        for i in range(0, data_move.shape[1], 10):
            end_idx = min(i + 10, data_move.shape[1])
            """
            range_len = end_idx - i
            gyroscope_data1 = np.mean(data_move[3:6, i:end_idx], axis=1)
            accelerometer_data1 = np.mean(data_move[:3, i:end_idx], axis=1)
            gyroscope_data2 = np.mean(data_move[9:12, i:end_idx], axis=1)
            accelerometer_data2 = np.mean(data_move[6:9, i:end_idx], axis=1)
            """
            gyroscope_data3 = np.mean(data_move[3:6, i:end_idx], axis=1)
            accelerometer_data3 = np.mean(data_move[:3, i:end_idx], axis=1)        
            #kalman1.update(gyroscope_data1, accelerometer_data1)
            #kalman2.update(gyroscope_data2, accelerometer_data2)
            kalman3.update(gyroscope_data3, accelerometer_data3)
            #quat1 = kalman1.get_quaternion()
            #quat2 = kalman2.get_quaternion()
            quat3 = kalman3.get_quaternion()
            quat_kalman.append(quat3) #quat1, quat2, 
    # quat_kalman = [6000,3] where each element is a quaternion: n+i+j+k
    quat_kalman = np.array(quat_kalman)
    quat_k = np.zeros((5000, 4))
    for i in range(5000):
        q = quat_kalman[i]
        quat_k[i, 0] = q.w
        quat_k[i, 1] = q.x
        quat_k[i, 2] = q.y
        quat_k[i, 3] = q.z
    """
    for i in range(6000):
        for j in range(3):
            q = quat_kalman[i, j]
            quat_k[j, i, 0] = q.w  # Scalar part
            quat_k[j, i, 1] = q.x  # Vector part x
            quat_k[j, i, 2] = q.y  # Vector part y
            quat_k[j, i, 3] = q.z  # Vector part z
    """
    return quat_k


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

data, label = load_and_log('test_flexion_elbow_dataset.npy', 'test_flexion_elbow_labels.npy', no_weight_classification=False, wandb_enabled=False, wandb_project="SL_multiclass")
data = apply_lowpass_filter(data, cutoff=50, fs=2000.0, order=5)

data = integrate_gyroscope_imu(data)



def calculate_quaternions_imu_no_avg(dataset):
    Ts = 0.0005  # Sampling period in seconds
    avg_gyr_upper_all = []
    avg_acc_upper_all = []
    avg_gyr_wrist_all = []
    avg_acc_wrist_all = []
    avg_gyr_arm_all = []
    avg_acc_arm_all = []

    # Loop over the dataset to calculate and store averages
    for data in dataset:
        avg_gyr_upper_all.append(data[3:6, :])
        avg_acc_upper_all.append(data[:3, :])
        avg_gyr_wrist_all.append(data[9:12, :])
        avg_acc_wrist_all.append(data[6:9, :])
        avg_gyr_arm_all.append(data[15:18, :])
        avg_acc_arm_all.append(data[12:15, :])
    avg_gyr_arm_all = avg_gyr_arm_all[0]
    avg_acc_arm_all = avg_acc_arm_all[0]
    avg_gyr_wrist_all = avg_gyr_wrist_all[0]
    avg_acc_wrist_all = avg_acc_wrist_all[0]
    avg_gyr_upper_all = avg_gyr_upper_all[0]
    avg_acc_upper_all = avg_acc_upper_all[0]
    avg_gyr_upper_all = np.array(avg_gyr_upper_all).reshape(dataset.shape[2] ,3)
    avg_acc_upper_all = np.array(avg_acc_upper_all).reshape(dataset.shape[2] ,3)
    avg_gyr_wrist_all = np.array(avg_gyr_wrist_all).reshape(dataset.shape[2] ,3)
    avg_acc_wrist_all = np.array(avg_acc_wrist_all).reshape(dataset.shape[2] ,3)
    avg_gyr_arm_all = np.array(avg_gyr_arm_all).reshape(dataset.shape[2] ,3)
    avg_acc_arm_all = np.array(avg_acc_arm_all).reshape(dataset.shape[2] ,3)

    # Reshape the data to match the expected input format for offlineVQF
    gyr_upper = np.ascontiguousarray(np.array(avg_gyr_upper_all))
    acc_upper = np.ascontiguousarray(np.array(avg_acc_upper_all))
    gyr_wrist = np.ascontiguousarray(np.array(avg_gyr_wrist_all))
    acc_wrist = np.ascontiguousarray(np.array(avg_acc_wrist_all))
    gyr_arm = np.ascontiguousarray(np.array(avg_gyr_arm_all))
    acc_arm = np.ascontiguousarray(np.array(avg_acc_arm_all))

    # Compute the quaternions using the averaged data for each accelerometer
    out_upper = offlineVQF(gyr_upper, acc_upper, mag=None, Ts=Ts, params=None)
    out_wrist = offlineVQF(gyr_wrist, acc_wrist, mag=None, Ts=Ts, params=None)
    out_arm = offlineVQF(gyr_arm, acc_arm, mag=None, Ts=Ts, params=None)

    # Extract the quaternions and store them
    quaternion = np.array([out_upper['quat6D'], out_wrist['quat6D'], out_arm['quat6D']])
    return quaternion
def calculate_quaternions_imu_no_avg(dataset):
    Ts = 0.0005  # Sampling period in seconds
    # Loop over the dataset to calculate and store averages (not in this case, we are not averaging the data)
    quats = []
    biases = []
    biases_sigma = []
    for data in dataset:
        avg_gyr_upper_all = []
        avg_acc_upper_all = []
        avg_gyr_wrist_all = []
        avg_acc_wrist_all = []
        avg_gyr_arm_all = []
        avg_acc_arm_all = []
        avg_gyr_wrist_all.append(data[3:6, :])
        avg_acc_wrist_all.append(data[:3, :])
        avg_gyr_upper_all.append(data[9:12, :])
        avg_acc_upper_all.append(data[6:9, :])
        avg_gyr_arm_all.append(data[15:18, :])
        avg_acc_arm_all.append(data[12:15, :])
        avg_gyr_upper_all = np.array(avg_gyr_upper_all).reshape(dataset.shape[2] ,3)
        avg_acc_upper_all = np.array(avg_acc_upper_all).reshape(dataset.shape[2] ,3)
        avg_gyr_wrist_all = np.array(avg_gyr_wrist_all).reshape(dataset.shape[2] ,3)
        avg_acc_wrist_all = np.array(avg_acc_wrist_all).reshape(dataset.shape[2] ,3)
        avg_gyr_arm_all = np.array(avg_gyr_arm_all).reshape(dataset.shape[2] ,3)
        avg_acc_arm_all = np.array(avg_acc_arm_all).reshape(dataset.shape[2] ,3)
        # Reshape the data to match the expected input format for offlineVQF
        gyr_upper = np.ascontiguousarray(np.array(avg_gyr_upper_all))
        acc_upper = np.ascontiguousarray(np.array(avg_acc_upper_all))
        gyr_wrist = np.ascontiguousarray(np.array(avg_gyr_wrist_all))
        acc_wrist = np.ascontiguousarray(np.array(avg_acc_wrist_all))
        gyr_arm = np.ascontiguousarray(np.array(avg_gyr_arm_all))
        acc_arm = np.ascontiguousarray(np.array(avg_acc_arm_all))
        # Compute the quaternions using the averaged data for each accelerometer
        out_upper = offlineVQF(gyr=gyr_upper, acc=acc_upper, mag=None, Ts=Ts, params=None)
        out_wrist = offlineVQF(gyr=gyr_wrist, acc=acc_wrist, mag=None, Ts=Ts, params=None)
        out_arm = offlineVQF(gyr=gyr_arm, acc=acc_arm, mag=None, Ts=Ts, params=None)
        # Extract the quaternions and store them
        quaternion = np.array([out_wrist['quat6D'], out_upper['quat6D'], out_arm['quat6D']])
        # The bias here are just for upper arm
        bias, bias_sigma = np.array(out_arm['bias']), np.array(out_arm['biasSigma'])
        quats.append(quaternion)
        biases.append(bias)
        biases_sigma.append(bias_sigma)
    return np.array(quats), np.array(biases), np.array(biases_sigma)
'''
    avg_gyr_upper_all = np.array(avg_gyr_upper_all).reshape(dataset.shape[2] ,3)
    avg_acc_upper_all = np.array(avg_acc_upper_all).reshape(dataset.shape[2] ,3)
    avg_gyr_wrist_all = np.array(avg_gyr_wrist_all).reshape(dataset.shape[2] ,3)
    avg_acc_wrist_all = np.array(avg_acc_wrist_all).reshape(dataset.shape[2] ,3)
    avg_gyr_arm_all = np.array(avg_gyr_arm_all).reshape(dataset.shape[2] ,3)
    avg_acc_arm_all = np.array(avg_acc_arm_all).reshape(dataset.shape[2] ,3)

    # Reshape the data to match the expected input format for offlineVQF
    gyr_upper = np.ascontiguousarray(np.array(avg_gyr_upper_all))
    acc_upper = np.ascontiguousarray(np.array(avg_acc_upper_all))
    gyr_wrist = np.ascontiguousarray(np.array(avg_gyr_wrist_all))
    acc_wrist = np.ascontiguousarray(np.array(avg_acc_wrist_all))
    gyr_arm = np.ascontiguousarray(np.array(avg_gyr_arm_all))
    acc_arm = np.ascontiguousarray(np.array(avg_acc_arm_all))

    # Compute the quaternions using the averaged data for each accelerometer
    out_upper = offlineVQF(gyr_upper, acc_upper, mag=None, Ts=Ts, params=None)
    out_wrist = offlineVQF(gyr_wrist, acc_wrist, mag=None, Ts=Ts, params=None)
    out_arm = offlineVQF(gyr_arm, acc_arm, mag=None, Ts=Ts, params=None)

    # Extract the quaternions and store them
    quaternion = np.array([out_upper['quat6D'], out_wrist['quat6D'], out_arm['quat6D']])
    return quaternion
'''
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
    print(f"gyr_upper shape: {gyr_upper.shape}")
    print(f"acc_upper shape: {acc_upper.shape}")

    # Compute the quaternions using the averaged data for each accelerometer
    out_upper = offlineVQF(acc_upper, gyr_upper, mag=None, Ts=Ts, params=None)
    out_wrist = offlineVQF(acc_wrist, gyr_wrist, mag=None, Ts=Ts, params=None)
    out_arm = offlineVQF(acc_arm, gyr_arm, mag=None, Ts=Ts, params=None)

    # Extract the quaternions and store them
    quaternion = np.array([out_upper['quat6D'], out_wrist['quat6D'], out_arm['quat6D']])
    return quaternion

def initial_orientation_sensor3(quat):
    theta = np.pi / 2  # 90 degrees in radians, s3 is 90 degree around x-axis wrt s2 and s1
    q_90x = np.array([np.cos(theta / 2), np.sin(theta / 2), 0, 0])
    q_90x_conj = np.array([np.cos(theta / 2), -np.sin(theta / 2), 0, 0])
    #quat = np.array(quat)
    # Apply the rotation
    q_temp = quaternion_multiply(q_90x, quat)
    q_result = quaternion_multiply(q_temp, q_90x_conj)
    return q_result

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

# quat = 3 20000 4
# quaternion calculation 3 20000 4
quat = calculate_quaternions_imu_no_avg(data)
quat = np.array(quat)
for i in range(quat.shape[1]):
    quat[2,i,:] = initial_orientation_sensor3(quat[2,i,:])


# R1 = wrist
# R2 = forearm
# R3 = upper arm
R1 = []
R2 = []
R3 = []
for i in range(quat.shape[1]):
    r1 = Rotation.from_quat(quat[0,i,:]).as_matrix()
    r2 = Rotation.from_quat(quat[1,i,:]).as_matrix()
    r3 = Rotation.from_quat(quat[2,i,:]).as_matrix()
    R1.append(np.array(r1))
    R2.append(np.array(r2))
    R3.append(np.array(r3))
R1 = np.array(R1)
R2 = np.array(R2)
R3 = np.array(R3)

R2 = remove_y_rotation(R2, data[0,9:11,:]) 

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
angle_elbow = apply_single_lowpass(np.array(angle_elbow), cutoff=200, fs=2000.0, order=5)
angle_wrist = apply_single_lowpass(np.array(angle_wrist), cutoff=200, fs=2000.0, order=5)

'''
dataaa = []
dataaa.append(angle_elbow)
dataaa.append(angle_wrist)
dataaa = apply_lowpass_filter(dataaa, cutoff=50, fs=2000.0, order=5)
'''

fig, axs = plt.subplots(2, 1, figsize=(12, 18), sharex=True)
time = range(int(len(angle_elbow)))
axs[0].plot(time, angle_elbow)
axs[0].set_ylabel('Angle X-axis')
axs[0].grid(True)
axs[0].set_title('Subplot 4')
axs[1].plot(time, angle_wrist)
axs[1].set_ylabel('Angle Y-axis')
axs[1].grid(True)
axs[1].set_title('Subplot 5')
fig.suptitle('Subplots of Data', y=0.92)
plt.show()

#%% PLOTS

# Create a figure and a single set of axes
fig, ax = plt.subplots(figsize=(12, 6))
time = range(100)
data = [range(100), range(100, 200), range(200, 300), range(300, 400), range(400, 500)]
for i in range(len(data)):
    ax.plot(time, data[i], label=f'Dataset {i+1}')
ax.legend()
ax.set_title('Overlayed Plots')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.grid(True)
fig.suptitle('Overlayed Plots of Data', y=0.95)
plt.show()

fig, axs = plt.subplots(5, 1, figsize=(12, 18), sharex=True)
time = range(100)
data = [range(100), range(100, 200), range(200, 300), range(300, 400), range(400, 500)]
for i in range(5):
    axs[i].plot(time, data[i])
    axs[i].set_ylabel(f'Dataset {i+1} values')
    axs[i].grid(True)
    axs[i].set_title(f'Subplot {i+1}')
fig.suptitle('Subplots of Data', y=0.92)
plt.show()


