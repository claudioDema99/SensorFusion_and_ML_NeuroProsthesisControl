import numpy as np
from scipy import signal

num_emg_channels = 9

def highpass_filter(signal_data, cutoff_frequency, sampling_frequency=2000):
    nyquist_frequency = 0.5 * sampling_frequency
    highpass_cutoff = cutoff_frequency / nyquist_frequency
    b, a = signal.butter(4, highpass_cutoff, 'highpass', analog=False)
    filtered_signal = []
    for data in signal_data:
        channels = []
        for channel in data:
            filter_signal = signal.filtfilt(b, a, channel)
            channels.append(filter_signal)
        filtered_signal.append(channels)
    return filtered_signal

def lowpass_filter(signal_data, cutoff_frequency, sampling_frequency=2000):
    nyquist_frequency = 0.5 * sampling_frequency
    lowpass_cutoff = cutoff_frequency / nyquist_frequency
    b, a = signal.butter(4, lowpass_cutoff, 'lowpass', analog=False)
    filtered_signal = []
    for data in signal_data:
        channels = []
        for channel in data:
            filter_signal = signal.filtfilt(b, a, channel)
            channels.append(filter_signal)
        filtered_signal.append(channels)
    return filtered_signal

def bandpass_filter(signal_data, low_cutoff_frequency, high_cutoff_frequency, sampling_frequency=2000):
    nyquist_frequency = 0.5 * sampling_frequency
    lowpass_cutoff = low_cutoff_frequency / nyquist_frequency
    highpass_cutoff = high_cutoff_frequency / nyquist_frequency
    b, a = signal.butter(4, [lowpass_cutoff, highpass_cutoff], 'bandpass', analog=False)
    filtered_signal = []
    for data in signal_data:
        channels = []
        for channel in data:
            filter_signal = signal.filtfilt(b, a, channel)
            channels.append(filter_signal)
        filtered_signal.append(channels)
    return filtered_signal

def notch_filter(signal_data, notch_frequency, Q, sampling_frequency=2000):
    notch = notch_frequency / (0.5 * sampling_frequency)
    b, a = signal.iirnotch(notch, Q)
    filtered_signal = []
    for data in signal_data:
        channels = []
        for channel in data:
            filter_signal = signal.filtfilt(b, a, channel)
            channels.append(filter_signal)
        filtered_signal.append(channels)
    return filtered_signal

def normalize_EMG_all_channels(dataset, max_values = None, save=False):
    channel_max = np.full(num_emg_channels, -np.inf)
    #channel_min = np.full(11, np.inf)
    data = np.array(dataset)
    data_2D = data.transpose(1, 0, 2)
    data_2D = data_2D.reshape(num_emg_channels, -1)
    if max_values is None:
        for i in range(data_2D.shape[0]):
            channel_max[i] = np.max(data_2D[i,:])
            #channel_min[i] = np.min(data_2D[i,:])
    else:
        channel_max = max_values
    normalized_data = []
    for window in dataset:
        channels = []
        for i, channel in enumerate(window):
            normalized_channel = channel / channel_max[i]
            if np.isnan(normalized_channel).any():
                normalized_channel = np.zeros(channel.shape)
            channels.append(normalized_channel)
        normalized_data.append(np.array(channels))
    if save:
        # Save the max values for each channel
        print("Max EMG values saved")
        np.save('params/max_emg.npy', channel_max)
    return normalized_data

def normalize_angles(imu_dataset):
    # Iterate over each window in the list
    for window in imu_dataset:
        # Update the window with the normalized values
        window[:] = window / 180.0
    return imu_dataset

def normalize_raw_imu(data, save=False):
    if len(data.shape) == 3:
        # Normalize each channel
        # Compute the mean and standard deviation for each channel
        mean = np.mean(data, axis=(0, 2), keepdims=True)  # Mean over [windows, samples]
        std = np.std(data, axis=(0, 2), keepdims=True)    # Std dev over [windows, samples]
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        # Apply Z-score normalization
        normalized_data = (data - mean) / std
    elif len(data.shape) == 2:
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        std = np.where(std == 0, 1, std)
        normalized_data = (data - mean) / std
    if save:
        # Save the mean and standard deviation for each channel
        print("Mean and std saved")
        np.save('params/mean_raw_imu.npy', mean)
        np.save('params/std_raw_imu.npy', std)
    return normalized_data

def save_max_emg_values(dataset):
    max_emg = np.load('C:/Users/claud/Desktop/LocoD/SavedData/Dataset/max_emg.npy')
    if len(np.array(dataset).shape) == 2:
        data = np.array(dataset)
    elif len(np.array(dataset).shape) == 3:
        data = np.array(dataset).transpose(1, 0, 2)
        data = data.reshape(num_emg_channels, -1)
    max_data = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        max_data[i] = np.max(data[i,:])
    max_emg = np.maximum(max_emg, max_data)
    np.save('max_emg.npy', max_emg)
    return max_emg

def convert_to_SI(dataset):
    for data in dataset:
        # accelerometer data from g to m/s^2
        data[num_emg_channels: num_emg_channels+3,:] = data[num_emg_channels: num_emg_channels+3,:] * 9.81
        data[num_emg_channels+6: num_emg_channels+9,:] = data[num_emg_channels+6: num_emg_channels+9,:] * 9.81
        data[num_emg_channels+12: num_emg_channels+15,:] = data[num_emg_channels+12: num_emg_channels+15,:] * 9.81
        # gyro data from deg/s to rad/s
        data[num_emg_channels+3: num_emg_channels+6,:] = np.deg2rad(data[num_emg_channels+3: num_emg_channels+6,:])
        data[num_emg_channels+9: num_emg_channels+12,:] = np.deg2rad(data[num_emg_channels+9: num_emg_channels+12,:])
        data[num_emg_channels+15: num_emg_channels+18,:] = np.deg2rad(data[num_emg_channels+15: num_emg_channels+18,:])
    return dataset