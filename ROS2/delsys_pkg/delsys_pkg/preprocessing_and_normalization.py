import numpy as np
from scipy import signal

num_emg_channels = 9

def highpass_filter(data, cutoff_frequency, sampling_frequency=2000):
    nyquist_frequency = 0.5 * sampling_frequency
    highpass_cutoff = cutoff_frequency / nyquist_frequency
    b, a = signal.butter(4, highpass_cutoff, 'highpass', analog=False)
    channels = []
    for channel in data:
        filter_signal = signal.filtfilt(b, a, channel)
        channels.append(filter_signal)
    return channels

def bandpass_filter(data, low_cutoff_frequency, high_cutoff_frequency, sampling_frequency=2000):
    nyquist_frequency = 0.5 * sampling_frequency
    lowpass_cutoff = low_cutoff_frequency / nyquist_frequency
    highpass_cutoff = high_cutoff_frequency / nyquist_frequency
    b, a = signal.butter(4, [lowpass_cutoff, highpass_cutoff], 'bandpass', analog=False)
    channels = []
    for channel in data:
        filter_signal = signal.filtfilt(b, a, channel)
        channels.append(filter_signal)
    return channels

def notch_filter(data, notch_frequency, Q, sampling_frequency=2000):
    notch = notch_frequency / (0.5 * sampling_frequency)
    b, a = signal.iirnotch(notch, Q)
    channels = []
    for channel in data:
        filter_signal = signal.filtfilt(b, a, channel)
        channels.append(filter_signal)
    return channels

def normalize_EMG_all_channels(dataset, max_values):
    channel_max = np.full(num_emg_channels, -np.inf)
    #channel_min = np.full(11, np.inf)
    data_2D = np.array(dataset)
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
    return normalized_data

def normalize_angles(imu_dataset):
    # Iterate over each window in the list
    imu_dataset = imu_dataset / 180.0
    return imu_dataset

# TOCHECK
def normalize_raw_imu(data, mean, std):
    if len(data.shape) == 2:
        normalized_data = (data - mean) / std
    return normalized_data.squeeze()

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

def convert_to_SI(data):
    # accelerometer data from g to m/s^2
    data[num_emg_channels: num_emg_channels+3,:] = data[num_emg_channels: num_emg_channels+3,:] * 9.81
    data[num_emg_channels+6: num_emg_channels+9,:] = data[num_emg_channels+6: num_emg_channels+9,:] * 9.81
    data[num_emg_channels+12: num_emg_channels+15,:] = data[num_emg_channels+12: num_emg_channels+15,:] * 9.81
    # gyro data from deg/s to rad/s
    data[num_emg_channels+3: num_emg_channels+6,:] = np.deg2rad(data[num_emg_channels+3: num_emg_channels+6,:])
    data[num_emg_channels+9: num_emg_channels+12,:] = np.deg2rad(data[num_emg_channels+9: num_emg_channels+12,:])
    data[num_emg_channels+15: num_emg_channels+18,:] = np.deg2rad(data[num_emg_channels+15: num_emg_channels+18,:])
    return data

def count_classes(labels):
    """
    Helper function to count occurrences of each class in the labels.
    Returns a dictionary with classes (as tuples) as keys and their counts as values.
    """
    unique, counts = np.unique(labels, axis=0, return_counts=True)
    return {tuple(label): count for label, count in zip(unique, counts)}

def undersample_majority_class_first_n(emg_data, labels, imu_data=None, target_samples=None):
    """
    Undersample the majority class (assumed to be [1,0,0,0,0]) by keeping the first N samples.
    Works with or without IMU data.
    """
    # Ensure labels are of integer type for consistent comparison
    labels = labels.astype(int)
    
    # Get the class distribution
    class_counts = count_classes(labels)
    print("Original class distribution:", class_counts)

    # Identify the majority class (as a tuple)
    majority_class = (1, 0, 0, 0, 0)

    # Ensure the majority class exists in the data
    if majority_class not in class_counts:
        raise ValueError(f"Majority class {majority_class} not found in the dataset.")

    majority_count = class_counts[majority_class]
    
    if target_samples is None:
        # If not specified, use the count of the second most common class
        target_samples = sorted(class_counts.values(), reverse=True)[1]

    print(f"Target samples for majority class: {target_samples}")

    # Convert majority_class back to numpy array for comparison
    majority_class_array = np.array(majority_class)

    # Find indices of majority class samples
    majority_indices = np.where(np.all(labels == majority_class_array, axis=1))[0]

    # Keep only the first 'target_samples' indices
    kept_majority_indices = majority_indices[:target_samples]

    # Find indices of other classes
    other_indices = np.where(np.any(labels != majority_class_array, axis=1))[0]

    # Combine indices
    all_kept_indices = np.concatenate([kept_majority_indices, other_indices])

    # Sort indices to maintain original order
    all_kept_indices.sort()

    # Use these indices to select data
    balanced_emg = emg_data[all_kept_indices]
    balanced_labels = labels[all_kept_indices]

    if imu_data is not None:
        balanced_imu = imu_data[all_kept_indices]
    else:
        balanced_imu = None

    print("Balanced class distribution:", count_classes(balanced_labels))

    return balanced_emg, balanced_imu, balanced_labels
