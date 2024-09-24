#%% IMPORT LIBRARIES
from typing import List
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pandas as pd
from scipy import signal
import pickle
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from cbpr_master_thesis.preprocessing_and_normalization import highpass_filter, bandpass_filter, notch_filter, normalize_EMG_all_channels, normalize_raw_imu, convert_to_SI
from cbpr_master_thesis.feature_extraction import extract_EMG_features, create_windows

#%% NORMALIZATION PARAMS

def store_normalization_params():
    file_path = "C:/Users/claud/Desktop/raw_dataset.npz"
    dataset = np.load(file_path, allow_pickle=True)
    emg, imu = (dataset['emg'], dataset['imu'])
    emg = emg.reshape(9, 100*emg.shape[0]) # 11
    imu = imu.reshape(18, 100*imu.shape[0])
    # if they have the same number of columns, then I concatenate them vertically with vstack
    data = np.vstack((emg, imu))
    # I add a component in the first dimension because I consider all of them belonging to the same movement
    data = np.expand_dims(data, axis=0)
    data[:,:11,:] = notch_filter(bandpass_filter(highpass_filter(data[:,:11,:], 0.5), 0.5, 100), 50, 30)
    # Convert the IMU data to SI units
    data = convert_to_SI(data)
    # (movements, windows_same_movement, channels, samples) (movements, windows_same_movement)
    window_length = 200
    while data.shape[2] % window_length != 0:
        window_length -= 1
    overlap = round(window_length / 2)
    label = np.array([1])
    windows, label = create_windows(data, window_length, overlap, label)
    windows = windows.squeeze()
    imu_vector = normalize_raw_imu(windows[:,11:,:])
    # Compute the mean across windows and samples for each channel
    emg_vector = np.array(normalize_EMG_all_channels(extract_EMG_features(windows)))
    return

#store_normalization_params()

#%% PLOT FUNCTIONS

def split_into_movements(labels, predictions):
    """
    Splits the labels and predictions into different movements.

    Parameters:
    - labels: Array of true labels.
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

#%% UNDERSAMPLING FUNCTIONS

def count_classes(labels):
    """Count occurrences of each class in the dataset."""
    unique, counts = np.unique(labels, axis=0, return_counts=True)
    return dict(zip(map(tuple, unique), counts))

def undersample_majority_class_first_n(emg_data, imu_data, labels, target_samples=None):
    """Undersample the majority class (assumed to be [1,0,0,0,0]) by keeping the first N samples."""
    class_counts = count_classes(labels)
    #print("Original class distribution:", class_counts)
    # Identify the majority class (assumed to be [1,0,0,0,0])
    majority_class = tuple([1.0,0.0,0.0,0.0,0.0])
    if majority_class not in class_counts:
        raise ValueError("Majority class [1,0,0,0,0] not found in the dataset")
    majority_count = class_counts[majority_class]
    if target_samples is None:
        # If not specified, use the count of the second most common class
        target_samples = sorted(class_counts.values(), reverse=True)[1]
    #print(f"Target samples for majority class: {target_samples}")
    # Find indices of majority class samples
    majority_indices = np.where(np.all(labels == majority_class, axis=1))[0]
    # Keep only the first 'target_samples' indices
    kept_majority_indices = majority_indices[:target_samples]
    # Find indices of other classes
    other_indices = np.where(np.any(labels != majority_class, axis=1))[0]
    # Combine indices
    all_kept_indices = np.concatenate([kept_majority_indices, other_indices])
    # Sort indices to maintain original order
    all_kept_indices.sort()
    # Use these indices to select data
    balanced_emg = emg_data[all_kept_indices]
    balanced_imu = imu_data[all_kept_indices]
    balanced_labels = labels[all_kept_indices]
    #print("Balanced class distribution:", count_classes(balanced_labels))
    return balanced_emg, balanced_imu, balanced_labels

#%% DATA EXTRACTION AND BALANCING

base_path = "C:/Users/claud/Desktop/CBPR_Recording_Folders/"
def extract_and_balance(participant_folder, recording_numbers):
    data = {'emg_angles': [], 'imu_angles': [], 'label_angles': [], 
            'emg_raw_imu': [], 'imu_raw_imu': [], 'label_raw_imu': [], 
            'emg_emg': [], 'imu_emg': [], 'label_emg': []}
    for recording_number in range(recording_numbers):
        file_paths = {
            'angles': os.path.join(base_path, participant_folder, f"DATASET_ANGLE_ESTIMATION_{str(recording_number)}", "angles_estimation_dataset.npz"),
            'raw_imu': os.path.join(base_path, participant_folder, f"DATASET_RAW_IMU_{str(recording_number)}", "raw_imu_dataset.npz"),
            'emg': os.path.join(base_path, participant_folder, f"DATASET_EMG_{str(recording_number)}", "emg_dataset.npz")
        }
        for file_type, file_path in file_paths.items():
            loaded_data = np.load(file_path, allow_pickle=True)
            for key in ['emg', 'imu', 'label']:
                data_key = f"{key}_{file_type}"
                if data_key in data:
                    if recording_number == 0:
                        data[data_key].append(loaded_data[key])
                    else:
                        data[data_key][0] = np.concatenate((data[data_key][0], loaded_data[key]), axis=0)
    data['emg_angles'], data['imu_angles'], data['label_angles'] = undersample_majority_class_first_n(data['emg_angles'][0], data['imu_angles'][0], data['label_angles'][0])
    data['emg_raw_imu'], data['imu_raw_imu'], data['label_raw_imu'] = undersample_majority_class_first_n(data['emg_raw_imu'][0], data['imu_raw_imu'][0], data['label_raw_imu'][0])
    data['emg_emg'], data['imu_emg'], data['label_emg'] = undersample_majority_class_first_n(data['emg_emg'][0], data['imu_emg'][0], data['label_emg'][0])
    return data

#%% Confusion Matrix, Accuracy, Classification Report and F1-scores

def data_analysis():
    base_path = "C:/Users/claud/Desktop/CBPR_Recordings/"
    datasets = ["ffnn_angles_dataset.npz", "lstm_angles_dataset.npz", "cnn_angles_dataset.npz", 
                "ffnn_raw_imu_dataset.npz", "lstm_raw_imu_dataset.npz", "cnn_raw_imu_dataset.npz", 
                "ffnn_emg_dataset.npz", "lstm_emg_dataset.npz", "cnn_emg_dataset.npz"]
    participant_folders = ["1_30_08/", "2_30_08/", "3_30_08/", "1_31_08/", "2_31_08/", "3_31_08/", 
                           "1_02_09/", "2_02_09/", "1_04_09/", "2_04_09/", "3_04_09/", "1_05_09/", 
                           "2_05_09/", "3_05_09/", "1_06_09/"]

    # Dictionary to store metrics
    metrics_dict = {
        'Participant': [],
        'Model': [],
        'Data_Type': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': []
    }

    # Dictionary to store confusion matrices
    confusion_matrices = {}

    for participant_folder in participant_folders:
        for dataset in datasets:
            # Construct full path
            full_path = os.path.join(base_path, participant_folder, dataset)
            # Load the dataset
            data = np.load(full_path)
            y_true = data['label']
            y_pred = data['prediction']
            model, data_type = dataset.split('_')[:2]

            # If y_true and y_pred are one-hot encoded, convert them to class labels
            if len(y_true.shape) > 1 and y_true.shape[1] > 1:
                y_true = np.argmax(y_true, axis=1)

            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                y_pred = np.argmax(y_pred, axis=1)

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            confusion_matrices[f'{participant_folder[:-1]}_{model}_{data_type}'] = cm

            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{model.upper()} - {data_type.capitalize()} - Confusion Matrix')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            cm_filename = f"{participant_folder[:-1]}_{model}_{data_type}_confusion_matrix.png"
            plt.savefig(os.path.join(base_path, participant_folder, cm_filename))
            plt.close()

            # Accuracy and classification report
            acc = accuracy_score(y_true, y_pred)
            report = classification_report(y_true, y_pred, output_dict=True)

            # Store metrics
            metrics_dict['Participant'].append(participant_folder[:-1])
            metrics_dict['Model'].append(model.upper())
            metrics_dict['Data_Type'].append(data_type.capitalize())
            metrics_dict['Accuracy'].append(acc)
            metrics_dict['Precision'].append(report['weighted avg']['precision'])
            metrics_dict['Recall'].append(report['weighted avg']['recall'])
            metrics_dict['F1-Score'].append(report['weighted avg']['f1-score'])

            # Print the report for reference
            print(f'{participant_folder[:-1]} - {model.upper()} - {data_type.capitalize()} - Classification Report:')
            print(classification_report(y_true, y_pred))

            # Plot F1-scores
            class_f1_scores = {f'Class {i}': report[str(i)]['f1-score'] for i in range(len(report) - 3)}
            plt.figure(figsize=(10, 6))
            plt.bar(class_f1_scores.keys(), class_f1_scores.values())
            plt.title(f'{model.upper()} - {data_type.capitalize()} - F1-scores for each class')
            plt.ylabel('F1-score')
            plt.ylim(0, 1)
            for i, (k, v) in enumerate(class_f1_scores.items()):
                plt.text(i, v, f'{v:.4f}', ha='center', va='bottom')
            f1_filename = f"{participant_folder[:-1]}_{model}_{data_type}_f1_scores.png"
            plt.savefig(os.path.join(base_path, participant_folder, f1_filename))
            plt.close()

            print(f"Analysis complete for {participant_folder[:-1]} - {model.upper()} - {data_type.capitalize()}")
            print("----------------------------------------")

    # Convert metrics dictionary to DataFrame for easier manipulation
    metrics_df = pd.DataFrame(metrics_dict)
    # Save metrics DataFrame to CSV or Excel
    metrics_df.to_csv("metrics.csv", index=False)
    # or: metrics_df.to_excel("metrics.xlsx", index=False)
    # Save confusion matrices dictionary using pickle
    with open("confusion_matrices.pkl", "wb") as f:
        pickle.dump(confusion_matrices, f)
    return metrics_df, confusion_matrices

'''
# Load CSV
metrics_df = pd.read_csv("metrics.csv")

# Load Excel
metrics_df = pd.read_excel("metrics.xlsx")
with open("confusion_matrices.pkl", "rb") as f:
    confusion_matrices = pickle.load(f)
'''

# %% PLOTS FROM DATA ANALYSIS

def plot_accuracy_boxplots(metrics_df, model_type):
    # Filter data for the specified model type (FFNN, CNN, LSTM)
    filtered_df = metrics_df[metrics_df['Model'] == model_type]
    
    # Create boxplot for accuracy across different input types (EMG, Angles, IMU)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Data_Type', y='Accuracy', data=filtered_df)
    plt.title(f'Accuracy Distribution for {model_type}')
    plt.ylabel('Accuracy')
    plt.xlabel('Input Type')
    plt.show()

def plot_performance_lineplot(metrics_df, model_type):
    # Filter data for the specified model type
    filtered_df = metrics_df[metrics_df['Model'] == model_type]

    # Create a line plot for each metric (Accuracy, Precision, Recall, F1-Score)
    plt.figure(figsize=(12, 6))
    
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        sns.lineplot(x='Data_Type', y=metric, data=filtered_df, label=metric)
    
    plt.title(f'Performance Across Input Types for {model_type}')
    plt.ylabel('Metric Value')
    plt.xlabel('Input Type')
    plt.legend(title='Metrics')
    plt.show()

def plot_f1_scores_barchart(metrics_df, model_type):
    # Filter data for the specified model type
    filtered_df = metrics_df[metrics_df['Model'] == model_type]
    
    # Iterate over the unique input types (EMG, Angles, IMU)
    for input_type in filtered_df['Data_Type'].unique():
        # Filter data for each input type
        class_f1_df = filtered_df[filtered_df['Data_Type'] == input_type]
        
        # Extract F1-scores for each class
        f1_data = []
        for i in range(5):  # Assuming you have 5 classes (0, 1, 2, 3, 4)
            class_column = 'F1-Score'
            if class_column in class_f1_df.columns:
                f1_score = class_f1_df[class_column].values[0]  # Get the F1-score for class i
                f1_data.append({'Class': f'Class {i}', 'F1-Score': f1_score})
            else:
                print(f'Column {class_column} not found in DataFrame')

        # Convert the list to a DataFrame
        f1_df = pd.DataFrame(f1_data)

        # Create bar plot for F1-score for each class
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Class', y='F1-Score', data=f1_df)
        plt.title(f'{model_type} - F1-scores for Each Class ({input_type})')
        plt.ylabel('F1-Score')
        plt.ylim(0, 1)
        plt.xlabel('Class')
        plt.show()

'''
# Example usage for FFNN, CNN, and LSTM
plot_accuracy_boxplots(metrics_df, 'FFNN')
plot_accuracy_boxplots(metrics_df, 'CNN')
plot_accuracy_boxplots(metrics_df, 'LSTM')
# Example usage for FFNN, CNN, and LSTM
plot_performance_lineplot(metrics_df, 'FFNN')
plot_performance_lineplot(metrics_df, 'CNN')
plot_performance_lineplot(metrics_df, 'LSTM')
# Example usage for FFNN, CNN, and LSTM
plot_f1_scores_barchart(metrics_df, 'FFNN')
plot_f1_scores_barchart(metrics_df, 'CNN')
plot_f1_scores_barchart(metrics_df, 'LSTM')
'''

#%% ANOVA and Tukey's HSD

def comparison_analysis(metrics_df):
    rows = []
    for index, row in metrics_df.iterrows():
        rows.append({
            'Participant': row['Participant'],
            'Classification_Method': row['Model'],
            'Input_Modality': row['Data_Type'],
            'Accuracy': row['Accuracy']
        })
    data = pd.DataFrame(rows)

    # Create a formula for the two-way ANOVA
    model = ols('Accuracy ~ C(Classification_Method) * C(Input_Modality)', data=data).fit()

    # Perform the ANOVA
    anova_table = sm.stats.anova_lm(model, typ=2)

    print(anova_table)

    # Perform Tukey HSD test for post-hoc analysis
    tukey = pairwise_tukeyhsd(endog=data['Accuracy'],
                            groups=data['Classification_Method'], 
                            alpha=0.05)

    print(tukey)

    # Create a line plot to show interaction between factors
    sns.pointplot(x='Input_Modality', y='Accuracy', hue='Classification_Method', data=data, dodge=True, markers=["o", "s", "D"], linestyles=["-", "--", ":"])
    plt.title('Interaction Plot: Accuracy by Classification Method and Input Modality')
    plt.show()
    return data

# Plot Heatmap for Accuracy values
def plot_heatmap(accuracy_matrix):
    plt.figure(figsize=(8,6))
    sns.heatmap(accuracy_matrix, annot=True, cmap='coolwarm', fmt='.3f', linewidths=0.5)
    plt.title('Heatmap: Accuracy Across Classification Methods and Input Modalities')
    plt.xlabel('Input Modality')
    plt.ylabel('Classification Method')
    plt.show()

# Bar Plot for Mean Accuracy with Error Bars (Standard Deviation)
def bar_chart_with_error_bars(data):
    plt.figure(figsize=(10,6))
    sns.barplot(x='Classification_Method', 
                y='Accuracy', 
                hue='Input_Modality', 
                data=data, 
                ci='sd',  # Standard deviation as error bars
                capsize=.2)  # Caps on error bars
    plt.title('Bar Chart: Accuracy by Classification Method and Input Modality')
    plt.ylabel('Mean Accuracy')
    plt.xlabel('Classification Method')
    plt.legend(title='Input Modality')
    plt.show()

'''
# Interaction Plot for Accuracy by Classification Method and Input Modality
def interaction_plot(data):
    sns.pointplot(x='Input_Modality', 
                  y='Accuracy', 
                  hue='Classification_Method', 
                  data=data, 
                  dodge=True, 
                  markers=["o", "s", "D"], 
                  linestyles=["-", "--", ":"],
                  ci='sd')  # ci='sd' adds standard deviation as error bars
    plt.title('Interaction Plot: Accuracy by Classification Method and Input Modality')
    plt.ylabel('Mean Accuracy')
    plt.xlabel('Input Modality')
    plt.legend(title='Classification Method')
    plt.show()
'''