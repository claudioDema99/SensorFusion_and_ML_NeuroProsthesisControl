#%%
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from cbpr_master_thesis.preprocessing_and_normalization import notch_filter, bandpass_filter, highpass_filter, normalize_EMG_all_channels, convert_to_SI, normalize_raw_imu
from cbpr_master_thesis.feature_extraction import create_windows, extract_EMG_features
from cbpr_master_thesis.data_analysis import undersample_majority_class_first_n, split_into_movements, calculate_accuracy_vs_time, plot_accuracy_vs_time, group_same_movement_type, calculate_movement_type_accuracy, plot_movement_accuracies

#%% PLOT FUNCTIONS

def split_into_movements(labels, predictions):
    """
    Splits the labels and predictions into different movements.

    Parameters:
    - labels: Aray of true labels.
    - predictions: Array of predictions.

    Returns:
    - movements: List of tuples, where each tuple is (movement_labels, movement_predictions).
    """
    movements = []
    current_movement_labels = []
    current_movement_predictions = []
    
    object_labels = {1, 2, 3, 4}  # Define the object labels
    rest_label = 0  # Define the rest label
    
    label_counter = 0  # Counter for unique object labels encountered in the current movement
    last_label = None  # Track the last object label

    for i in range(len(labels)):
        current_label = labels[i]
        current_prediction = predictions[i]

        scalar_label = np.argmax(current_label)
        # Increase the counter when encountering a new object label different from the previous one
        if scalar_label in object_labels and scalar_label != last_label:
            label_counter += 1
            last_label = scalar_label

        # When we've seen 4 objects (1, 2, 3, 4) and the rest label, a movement is complete
        if label_counter == 5:
            # Append the completed movement to the list of movements
            movements.append((np.array(current_movement_labels), np.array(current_movement_predictions)))
            # Reset for the next movement
            current_movement_labels = []
            current_movement_predictions = []
            label_counter = 0
            last_label = None  # Reset the last label for the new movement

        # Track the current label in the movement
        current_movement_labels.append(scalar_label)
        current_movement_predictions.append(current_prediction)
    
    # Append the last movement if any labels remain after the loop
    if current_movement_labels:
        movements.append((np.array(current_movement_labels), np.array(current_movement_predictions)))
    
    return movements

def calculate_accuracy_vs_time(movements):
    """
    Calculates accuracy over time for each movement.

    Parameters:
    - movements: List of tuples, where each tuple is (movement_labels, movement_predictions).

    Returns:
    - accuracies: List of accuracy values over time for each movement.
    """
    accuracies = []
    
    for movement_labels, movement_predictions in movements:
        accuracy = []
        for t in range(1, len(movement_labels) + 1):
            correct = np.sum(movement_labels[:t] == movement_predictions[:t])
            accuracy.append(correct / t)
        
        accuracies.append(accuracy)
    
    return accuracies

def plot_accuracy_vs_time(accuracies):
    """
    Plots accuracy vs time for each movement.

    Parameters:
    - accuracies: List of accuracy values over time for each movement.
    """
    plt.figure(figsize=(10, 6))
    
    for i, accuracy in enumerate(accuracies):
        plt.plot(accuracy, label=f'Movement {i+1}')
    
    plt.title('Accuracy vs Time for Each Movement')
    plt.xlabel('Time (Inference Steps)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def group_same_movement_type(movements):
    type1 = []
    type2 = []
    type3 = []
    type4 = []
    j = 0
    for i in range(1, len(movements)+1):
        if i == 1+j:
            type1.append(movements[i-1])
        elif i == 2+j:
            type2.append(movements[i-1])
        elif i == 3+j:
            type3.append(movements[i-1])
        elif i == 4+j:
            type4.append(movements[i-1])
            j += 4
    return type1, type2, type3, type4

def calculate_movement_type_accuracy(movements: List[List[List[int]]]) -> float:
    accuracies = []
    for movement in movements:  # Iterate over the 5 movements
        labels = movement[0]  # First list contains labels
        predictions = movement[1]  # Second list contains predictions
        
        # Ensure labels and predictions have the same length
        if len(labels) != len(predictions):
            raise ValueError("Labels and predictions must have the same length")
        
        correct = sum(1 for label, pred in zip(labels, predictions) if label == pred)
        accuracy = correct / len(labels)
        accuracies.append(accuracy)
    
    return sum(accuracies) / len(accuracies)

def plot_movement_accuracies(accuracies):
    movement_types = list(accuracies.keys())
    avg_accuracies = [np.mean(acc) for acc in accuracies.values()]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Overall accuracies
    ax1.bar(movement_types, avg_accuracies)
    ax1.set_ylabel('Average Accuracy')
    ax1.set_title('Overall Movement Type Accuracies')
    ax1.set_ylim(0, 1)
    
    # Detailed accuracies
    x = np.arange(len(movement_types))
    width = 0.15
    for i in range(3):
        movement_accuracies = [acc[i] for acc in accuracies.values()]
        ax2.bar(x + i*width, movement_accuracies, width, label=f'Movement {i+1}')
    
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Detailed Movement Accuracies')
    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels(movement_types)
    ax2.legend()
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()

#%%

# Calculate and plot accuracy vs time for the first model
folder_name = "1_30_08"
recording_number = "1"
file_path = "//wsl.localhost/Ubuntu/home/dema/CBPR_Recording_Folders/"+folder_name+"/DATASET_ANGLE_ESTIMATION_"+recording_number+"/angles_estimation_dataset.npz"
file_path2 = "//wsl.localhost/Ubuntu/home/dema/CBPR_Recording_Folders/"+folder_name+"/DATASET_RAW_IMU_"+recording_number+"/raw_imu_dataset.npz"
file_path3 = "//wsl.localhost/Ubuntu/home/dema/CBPR_Recording_Folders/"+folder_name+"/DATASET_EMG_"+recording_number+"/emg_dataset.npz"
current_path = "current_data_angles.npz"
data = np.load(current_path, allow_pickle=True)
emg, imu, label, pred1, pred2, pred3 = (data['emg'], data['imu'], data['label'], data['prediction_0'], data['prediction_1'], data['prediction_2'])

# output of split_into_movements: [movements, label/prediction]

# movements realted to the predicition of the first model
movement1, movement2, movement3, movement4 = group_same_movement_type(split_into_movements(label, pred1))
# movements realted to the predicition of the second model
movement1_second, movement2_second, movement3_second, movement4_second = group_same_movement_type(split_into_movements(label, pred2))
# movements realted to the predicition of the third model
movement1_third, movement2_third, movement3_third, movement4_third = group_same_movement_type(split_into_movements(label, pred3))

accuracy_type1 = calculate_movement_type_accuracy(movement1)
accuracy_type2 = calculate_movement_type_accuracy(movement2)
accuracy_type3 = calculate_movement_type_accuracy(movement3)
accuracy_type4 = calculate_movement_type_accuracy(movement4)
accuracy_type1_second = calculate_movement_type_accuracy(movement1_second)
accuracy_type2_second = calculate_movement_type_accuracy(movement2_second)
accuracy_type3_second = calculate_movement_type_accuracy(movement3_second)
accuracy_type4_second = calculate_movement_type_accuracy(movement4_second)
accuracy_type1_third = calculate_movement_type_accuracy(movement1_third)
accuracy_type2_third = calculate_movement_type_accuracy(movement2_third)
accuracy_type3_third = calculate_movement_type_accuracy(movement3_third)
accuracy_type4_third = calculate_movement_type_accuracy(movement4_third)

# Example usage
accuracies = {
    'Type 1': [accuracy_type1, accuracy_type1_second, accuracy_type1_third],
    'Type 2': [accuracy_type2, accuracy_type2_second, accuracy_type2_third],
    'Type 3': [accuracy_type3, accuracy_type3_second, accuracy_type3_third],
    'Type 4': [accuracy_type4, accuracy_type4_second, accuracy_type4_third]
}

plot_movement_accuracies(accuracies)

'''
accuracies1 = calculate_accuracy_vs_time(movements_angles_first_model)
plot_accuracy_vs_time(accuracies1)
accuracies2 = calculate_accuracy_vs_time(movements_angles_second_model)
plot_accuracy_vs_time(accuracies2)
accuracies3 = calculate_accuracy_vs_time(movements_angles_third_model)
plot_accuracy_vs_time(accuracies3)
'''

# %%

current_path = "current_data_angles.npz"
current_path = "current_data_raw_imu.npz"
current_path = "current_data_emg.npz"
data = np.load(current_path, allow_pickle=True)
emg, label, pred = (data['emg'], data['label'], data['predictions'])

# output of split_into_movements: [movements, label/prediction]

# movements realted to the predicition of the first model
movement1, movement2, movement3, movement4 = group_same_movement_type(split_into_movements(label, pred))

accuracy_type1 = calculate_movement_type_accuracy(movement1)
accuracy_type2 = calculate_movement_type_accuracy(movement2)
accuracy_type3 = calculate_movement_type_accuracy(movement3)
accuracy_type4 = calculate_movement_type_accuracy(movement4)

# Example usage
accuracies = {
    'Type 1': [accuracy_type1],
    'Type 2': [accuracy_type2],
    'Type 3': [accuracy_type3],
    'Type 4': [accuracy_type4]
}

plot_movement_accuracies(accuracies)
# %%
