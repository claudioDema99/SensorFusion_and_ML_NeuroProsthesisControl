#%% load data
import numpy as np

base_path = "C:/Users/claud/Desktop/CBPR_Recordings/"
datasets = ["ffnn_angles_dataset.npz", "lstm_angles_dataset.npz", "cnn_angles_dataset.npz", 
            "ffnn_raw_imu_dataset.npz", "lstm_raw_imu_dataset.npz", "cnn_raw_imu_dataset.npz", 
            "ffnn_emg_dataset.npz", "lstm_emg_dataset.npz", "cnn_emg_dataset.npz"]
participant_folders = ["1_30_08/", "2_30_08/", "3_30_08/", "1_31_08/", "2_31_08/", "3_31_08/", 
                        "1_02_09/", "2_02_09/", "1_04_09/", "2_04_09/", "3_04_09/", "1_05_09/", 
                        "2_05_09/", "3_05_09/", "1_06_09/"]
data = np.load(base_path+participant_folders[0]+datasets[0])

#%% create boolean lists: True if prediction == label, False otherwise
# ffnn_angles is a list of 15 dictionaries, one for each participant
# each dictionary has 4 lists, that are not the trajectories but the different labels (objects)
# I still need to divide each list into the 4 different trajectories,
# and then for each trajectory I have a list of boolean per object (label)

def get_booleans(data):
    indexes_zero = []
    indexes_one = []
    indexes_two = []
    indexes_three = []
    indexes_four = []
    if data['label'].ndim > 1:
        label_indices = np.argmax(data['label'], axis=1)
    else:
        label_indices = data['label']
    # Store indices for each class
    for idx, label in enumerate(label_indices):
        if label == 0:
            indexes_zero.append(idx)
        elif label == 1:
            indexes_one.append(idx)
        elif label == 2:
            indexes_two.append(idx)
        elif label == 3:
            indexes_three.append(idx)
        elif label == 4:
            indexes_four.append(idx)
    # Create empty boolean lists for each class
    boolean_one = []
    boolean_two = []
    boolean_three = []
    boolean_four = []
    # For class 1: check if predictions at indices from indexes_one are equal to 1
    for idx in indexes_one:
        boolean_one.append(data['prediction'][idx] == 1)
    # For class 2: check if predictions at indices from indexes_two are equal to 2
    for idx in indexes_two:
        boolean_two.append(data['prediction'][idx] == 2)
    # For class 3: check if predictions at indices from indexes_three are equal to 3
    for idx in indexes_three:
        boolean_three.append(data['prediction'][idx] == 3)
    # For class 4: check if predictions at indices from indexes_four are equal to 4
    for idx in indexes_four:
        boolean_four.append(data['prediction'][idx] == 4)
    return {1:boolean_one, 2:boolean_two, 3:boolean_three, 4:boolean_four}

ffnn_angles = []
lstm_angles = []
cnn_angles = []

for participant in participant_folders:
    ffnn_angles.append(get_booleans(np.load(base_path+participant+datasets[0])))
    lstm_angles.append(get_booleans(np.load(base_path+participant+datasets[1])))
    cnn_angles.append(get_booleans(np.load(base_path+participant+datasets[2])))

# %% to resemple the data to 'number_of_samples' samples

def resample_to_n(boolean_list, number_of_samples=100):
    """Resample a list of booleans to a fixed length of 100 by segment majority voting."""
    length = len(boolean_list)
    resampled_list = []
    
    for i in range(number_of_samples):
        # Define the segment bounds in the original list
        start = int(i * length / number_of_samples)
        end = int((i + 1) * length / number_of_samples)
        
        # Take the majority in each segment
        segment = boolean_list[start:end]
        majority = sum(segment) > len(segment) / 2  # True if more than half are True
        resampled_list.append(majority)
    #return resampled_list
    return np.repeat(resampled_list, 4).tolist()

# Apply resampling for each participant's trajectory
resampled_data = []
for participant in lstm_angles:
    resampled_participant = {
        "trajectory_1": resample_to_n(participant[1], 25),
        "trajectory_2": resample_to_n(participant[2], 25),
        "trajectory_3": resample_to_n(participant[3], 25),
        "trajectory_4": resample_to_n(participant[4], 25),
    }
    resampled_data.append(resampled_participant)

# %% plot the resampled data

import matplotlib.pyplot as plt

# Define colors for each trajectory
colors = ["blue", "green", "orange", "red"]
trajectory_labels = ["Trajectory 1", "Trajectory 2", "Trajectory 3", "Trajectory 4"]

# Create a subplot for each participant
num_participants = len(resampled_data)
fig, axes = plt.subplots(num_participants, 1, figsize=(10, 2 * num_participants), sharex=True)

# Iterate through each participant and plot each trajectory's boolean values
for i, participant in enumerate(resampled_data):
    ax = axes[i] if num_participants > 1 else axes  # In case of a single plot
    for j, (trajectory, values) in enumerate(participant.items()):
        # Convert boolean values to 0 and 1 for plotting
        values_numeric = [1 if v else 0 for v in values]
        ax.plot(values_numeric, label=trajectory_labels[j], color=colors[j], alpha=0.7)
        
    # Customize each participant's subplot
    ax.set_ylim(-0.1, 1.1)  # Set y-axis range to match boolean scale (0 to 1)
    ax.set_ylabel(f"Participant {i + 1}")
    ax.legend(loc="upper right")
    ax.set_yticks([0, 1])  # Display y-ticks at 0 and 1
    ax.grid(True)

# Add a shared x-label and title
plt.xlabel("Resampled Time Points")
plt.suptitle("Algorithm Performance by Participant and Trajectory")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit title
plt.show()

#%% score and plot

# Assuming `resampled_data` is the list of dictionaries where each dictionary has four trajectories
scored_data = []

# Iterate through each participant in resampled_data
for participant in resampled_data:
    # Extract lists for each trajectory
    trajectory_1 = participant["trajectory_1"]
    trajectory_2 = participant["trajectory_2"]
    trajectory_3 = participant["trajectory_3"]
    trajectory_4 = participant["trajectory_4"]
    
    # Initialize a list to store the score for each sample point
    participant_score = []
    
    # Iterate through each sample point
    for i in range(len(trajectory_1)):
        # Calculate score by summing `True` values across the four trajectories at the same sample point
        score = sum([trajectory_1[i], trajectory_2[i], trajectory_3[i], trajectory_4[i]])
        participant_score.append(score)
    
    # Append the score list for this participant to scored_data
    scored_data.append(participant_score)

# Now `scored_data` contains the score list for each participant, with values from 0 to 4 for each sample point.

import matplotlib.pyplot as plt

# Plot each participant's score list
num_participants = len(scored_data)
fig, axes = plt.subplots(num_participants, 1, figsize=(10, 2 * num_participants), sharex=True)

# Iterate through each participant's score list and plot
for i, scores in enumerate(scored_data):
    ax = axes[i] if num_participants > 1 else axes  # Handle case of a single plot
    ax.plot(scores, color="purple", marker="o", alpha=0.6)
    
    # Customize the subplot for each participant
    ax.set_ylim(-0.1, 4.1)  # y-axis range from 0 to 4
    ax.set_ylabel(f"Participant {i + 1}")
    ax.set_yticks([0, 1, 2, 3, 4])  # Show only integers 0 to 4
    ax.grid(True)

# Add shared x-label and title
plt.xlabel("Resampled Time Points")
plt.suptitle("Score per Sample Point for Each Participant (0 to 4 Correct Recognitions)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for title
plt.show()

#%% final global score data

num_samples = len(scored_data[0])
global_score = [0] * num_samples

# Sum scores across all participants for each sample point
for participant_scores in scored_data:
    for i in range(num_samples):
        global_score[i] += participant_scores[i]

import matplotlib.pyplot as plt

# Plot the global score
plt.figure(figsize=(10, 4))
plt.plot(global_score, color="purple", marker="o", alpha=0.7)

# Customize the plot
plt.ylim(0, 61)  # Set y-axis range from 0 to max possible score (number of participants)
plt.xlabel("Resampled Time Points")
plt.ylabel("Global Score (Summed Recognitions)")
plt.title("Global Algorithm Performance Over Time (Summed Across Participants)")
plt.grid(True)
plt.show()

# %%

import matplotlib.colors as mcolors

# Add small random noise to each value in global_score
noise = np.random.normal(0, 1, len(global_score))  # Noise with mean=0, std deviation=1
global_score = global_score + noise  # Add noise to global_score

global_scores = [
    global_score[0:25],      # First 25 elements
    global_score[25:50],     # Elements 25-49
    global_score[50:75],     # Elements 50-74
    global_score[75:100]     # Elements 75-99
]

# Create a linear space for x-axis values (now just for 25 points)
x = np.linspace(0, 24, 25)

# Create figure with 4 subplots
fig, axes = plt.subplots(4, 1, figsize=(10, 8))
fig.suptitle('Scores by Trajectory Segment', y=1.02)

# Define a custom red-to-green colormap
red_to_green = mcolors.LinearSegmentedColormap.from_list("red_green", ["red", "green"])

# Create a normalization that will be used for all plots
norm = plt.Normalize(0, 60)  # Set fixed range from 0 to 60

# Plot each segment
for i, (ax, scores) in enumerate(zip(axes, global_scores)):
    sc = ax.scatter(x, np.zeros_like(x), c=scores, cmap=red_to_green, 
                   s=1000, marker='s', norm=norm)  # Added norm parameter
    ax.axis('off')
    ax.set_title(f'Segment {i+1}')

# Add color bar with legend (shared for all plots)
cbar = fig.colorbar(sc, ax=axes, orientation="horizontal", pad=0.1, aspect=30, shrink=0.5)
cbar.set_label("Score (0-60)")

plt.tight_layout()
plt.show()
# %%
