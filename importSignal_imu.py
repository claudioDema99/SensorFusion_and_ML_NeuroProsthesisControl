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

# Path to the directory containing CSV files
directory_path = Path('C:/Users/claud/Desktop/LocoD/SavedData/DataTest') # MODIFY
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
label_directory_path = Path('C:/Users/claud/Desktop/LocoD/SavedData/LabelTest') 
label_files = list(label_directory_path.glob('*.csv'))
if len(label_files) == 0:
    print("No label files found in the directory!")
elif len(label_files) == 1:
    label = pd.read_csv(label_files[0], header=None).to_numpy()
    print(label.shape)
else:
    label = concatenate_csv(label_files)
    print(label.shape)

#%%

'''
Problem: now I have all the row data, also the samples that are not useful for the classification task. 
I want to extract the samples that are useful for the classification task, i.e. the samples that starts from the beginning of the movement (time instant when the operator
has clicked the button) and ends after 6 seconds. (2000 Hz * 6 seconds = 12000 samples per movement)
But now I dont want to store each movement in a separate plane, but I store each movement as an element of a list, where each element is a single 2D numpy array.

10 seconds = 20000 samples
25 seconds = 50000 samples
15 seconds = 30000 samples
30 seconds = 60000 samples
'''

def extract_and_store_classes(data, time_instants, sample_frequency=2000, fixed_segment_length=30000):
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


# labels: contains only 1 label for each movement
# data: list of 2D numpy arrays, each array contains the samples of a single movement
labels = np.array(label[0, :])
dataset = extract_and_store_classes(data, label[1, :])

#%%
# data 3D numpy array
# Path to save the file
save_path = 'C:/Users/claud/Desktop/LocoD/SavedData/Dataset/random_positions_dataset.npy'
save_path_label = 'C:/Users/claud/Desktop/LocoD/SavedData/Dataset/random_positions_labels.npy'
# Save the 3D array to file
np.save(save_path, dataset)
np.save(save_path_label, labels)
print("3D NumPy array saved successfully!")

# %% 