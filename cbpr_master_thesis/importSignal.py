#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vqf import offlineVQF
from pathlib import Path
from numpy.linalg import norm

# %%

'''
Problem: I have different csv files stored in a folder from MatLab (LocoD) and I want not only to import them but also to concatenate them in a single numpy array.
Here I do this for both the data and the labels.
'''

# Function to concatenate CSV files and convert to NumPy array
def concatenate_csv(files):
    data_frames = []
    # Read each CSV file and append its contents to the list
    for file in files:
        df = pd.read_csv(file, header=None)
        data_frames.append(df)
    # Concatenate the CSV dataframes vertically
    concatenated_data = pd.concat(data_frames, axis=1)
    # Convert to NumPy array
    concatenated_array = concatenated_data.to_numpy()
    return concatenated_array

def concatenate_csv_label(files):
    data_frames = []
    lengths = []  # To store the lengths of each CSV file

    # Read each CSV file and append its contents to the list
    for file in files:
        df = pd.read_csv(file, header=None)
        data_frames.append(df)
        # Count the number of values in the current file and store it in lengths array
        lengths.append(len(df))

    # Concatenate the CSV dataframes vertically
    concatenated_data = pd.concat(data_frames, axis=1)
    # Convert to NumPy array
    concatenated_array = concatenated_data.to_numpy()

    # Adjust values in the concatenated array according to the specified logic
    cumulative_sum = lengths[0]  # Initialize the cumulative sum with the first length
    total_increment = 0  # Initialize the total increment to zero

    for i in range(1, len(lengths)):  # Start from the second length
        increment = lengths[i-1] * 3  # Calculate the increment based on the previous length
        total_increment += increment  # Add the increment to the total increment

        # Update the section of concatenated_array by adding the total increment
        # I don't know which is the correct one
        concatenated_array[1, cumulative_sum:cumulative_sum + lengths[i]] += total_increment
        concatenated_array[cumulative_sum:cumulative_sum + lengths[i]] += total_increment

        # Update the cumulative sum for the next iteration
        cumulative_sum += lengths[i]

    return concatenated_array

# Path to the directory containing CSV files
directory_path = Path('C:/Users/claud/Desktop/LocoD/SavedData/DataCSV') # MODIFY
# Get a list of all CSV files in the directory (but not in the subdirectories)
csv_files = list(directory_path.glob('*.csv'))
# Concatenate the CSV files and convert to NumPy array
if len(csv_files) == 0:
    print("No CSV files found in the directory!")  
elif len(csv_files) == 1:
    data = pd.read_csv(csv_files[0], header=None).to_numpy()
    # Print the shape of the concatenated array
    print(data.shape)
else:
    data = concatenate_csv(csv_files)
    # Print the shape of the concatenated array
    print(data.shape)

# Do the same for the time-label and label files
label_directory_path = Path('C:/Users/claud/Desktop/LocoD/SavedData/LabelCSV') 
label_files = list(label_directory_path.glob('*.csv'))
if len(label_files) == 0:
    print("No label files found in the directory!")
elif len(label_files) == 1:
    label = pd.read_csv(label_files[0], header=None).to_numpy()
    print(label.shape)
else:
    label = concatenate_csv_label(label_files)
    print(label.shape)

#%%

'''
Problem: now I have all the row data, also the samples that are not useful for the classification task. 
I want to extract the samples that are useful for the classification task, i.e. the samples that starts from the beginning of the movement (time instant when the operator
has clicked the button) and ends after 4 seconds. (2000 Hz * 4 seconds = 6000 samples per movement)
But now I dont want to store each movement in a separate plane, but I store each movement as an element of a list, where each element is a single 2D numpy array.
'''

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


# labels: contains only 1 label for each movement
# data: list of 2D numpy arrays, each array contains the samples of a single movement
labels = np.array(label[0, :])
dataset = extract_and_store_classes(data, label[1, :])

def insert_rest_after_each_element(arr):
    # Create a new array with double the length of the original array
    extended_array = np.empty(len(arr) * 2, dtype=arr.dtype)
    
    # Assign values to the new array
    extended_array[0::2] = arr  # Original elements at even indices
    extended_array[1::2] = 5    # Insert '4' at odd indices
    
    return extended_array

#labels = insert_rest_after_each_element(labels)
print(np.array(dataset).shape)
print(np.array(labels).shape)

#%%
'''
move the dataset needed inside the folder DataCSV and LabelCSV paying attention at the order of the files
Then run above cells to import the data and labels
Then run the following cell modifying the name of the npy files saved

binary problems:
    borraccia vuota e piena:------------------------------binary1 (borraccia 1 e piena 4)
    borraccia vuota e mouse:------------------------------binary2 (borraccia 1 and mouse 2)
    borraccia vuota e tazza:------------------------------binary3 (borraccia 1 and tazza 3)
    mouse e tazza
multiclass problems:
    borraccia vuota, piena, mouse:------------------------multi1 (borraccia 1, piena 4, mouse 2)
    borraccia vuota, piena, tazza:------------------------multi2 (borraccia 1, piena 4, tazza 3)
    borraccia vuota, piena, mouse, tazza:-----------------multi3 (borraccia 1, piena 4, mouse 2, tazza 3)
'''
# data 3D numpy array
# Path to save the file
save_path = 'C:/Users/claud/Desktop/LocoD/SavedData/Dataset/prova_dataset.npy'
save_path_label = 'C:/Users/claud/Desktop/LocoD/SavedData/Dataset/prova_labels.npy'
# Save the 3D array to file
np.save(save_path, dataset)
np.save(save_path_label, labels)
print("3D NumPy array saved successfully!")

# %%

# Function to concatenate CSV files and convert to NumPy array
def concatenate_csv(file):
    data_frames = []
    # Read each CSV file and append its contents to the list
    df = pd.read_csv(file, header=None)
    data_frames.append(df)
    return np.array(np.squeeze(data_frames, axis=0))
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
    return data_classes

# Path to the directory containing CSV files and get a list of all CSV files in the directory (but not in the subdirectories)
directory_path = Path('C:/Users/claud/Desktop/LocoD/SavedData/DataCSV') # MODIFY
csv_files = list(directory_path.glob('*.csv'))
# Do the same for the time-label and label files
label_directory_path = Path('C:/Users/claud/Desktop/LocoD/SavedData/LabelCSV') 
label_files = list(label_directory_path.glob('*.csv'))
# Lists for storing the dataset and labels
dataset = []
labels = []
for file_csv, file_label in zip(csv_files, label_files):
    print(f"Processing {file_csv} and {file_label}")
    # Concatenate the CSV files and convert to NumPy array
    data = concatenate_csv(file_csv)
    # Concatenate the time-label and label files and convert to NumPy array
    label = concatenate_csv(file_label)
    # Do something with the data and labels
    labels.append(label[0,:])
    dataset.append(extract_and_store_classes(data, label[1, :]))
for i in range(len(dataset)):
    print(np.array(dataset[i]).shape)
    print(np.array(labels[i]).shape)
# THIS SOLVES DIMENSIONALITY PROBLEM
for data in dataset:
    data = np.array(data)
dataset = np.concatenate(dataset, axis=0)
labels = np.concatenate(labels, axis=0)
print(dataset.shape)
print(labels.shape)
# data 3D numpy array
# Path to save the file
save_path = 'C:/Users/claud/Desktop/LocoD/SavedData/Dataset/prova_dataset.npy'
save_path_label = 'C:/Users/claud/Desktop/LocoD/SavedData/Dataset/prova_labels.npy'
# Save the 3D array to file
np.save(save_path, dataset)
np.save(save_path_label, labels)
print("3D NumPy array saved successfully!")
# %%
