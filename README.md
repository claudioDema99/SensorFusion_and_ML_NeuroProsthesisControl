# Sensor fusion and ML for neuroprosthesis control

Code from my MSc thesis in Robotics Engineering (University of Genoa, 2024), done during an internship at the Center for Bionics and Pain Research (CBPR) in Gothenburg.

Thesis: *Sensor Fusion for Enhancing Prosthesis Control: Overcoming the Limb Position Effect and Dynamic Movement Variability Using EMG and IMU Data.* Supervisors: Prof. Maura Casadio, Dr. Fabian Just.

## The problem

With myoelectric prostheses the same grasp produces different EMG patterns depending on the arm posture (limb position effect), so a classifier trained in one posture degrades in others. The idea here: use three IMUs (wrist, forearm, upper arm) to estimate nine arm orientation angles and give them to the classifier together with the EMG features, instead of raw IMU readings.

## Setup

- 15 healthy participants, 6 EMG + 3 IMU sensors (Delsys Trigno), 27 channels at 2 kHz
- 5 classes: paper plate, ceramic plate, empty bottle, full bottle, rest
- three inputs (EMG, EMG + raw IMU, EMG + estimated angles) x three models (FFNN, CNN, LSTM)
- small models on purpose: the target prosthesis had 1 MB of flash for the model

## Results

Offline, participant-specific, mean over the 15 participants (thesis ch. 5):

| Model | EMG | EMG + raw IMU | EMG + angles |
|---|--:|--:|--:|
| FFNN | 23.9% | 30.0% | 88.4% |
| CNN | 22.5% | 30.2% | 87.6% |
| LSTM | 22.1% | 34.1% | 82.0% |

Things that did not work as well:

- on unseen participants (5-fold, 12 train / 3 test) accuracy drops to 33-35%, chance is 20%
- online, the ROS2 system runs end to end but live accuracy was below the offline numbers
- all participants were non-amputees, nothing here is validated on prosthesis users

## Repo layout

| Path | What |
|---|---|
| `Pipeline.py`, `CNN_pipeline.py`, `LSTM_pipeline.py` | offline experiments for FFNN, CNN, LSTM |
| `Pipeline_hyperparams.py` | hyperparameter search |
| `cbpr_master_thesis/` | preprocessing, feature extraction, orientation angles, models, analysis |
| `ROS2/delsys_pkg/` | ROS2 nodes: streaming, recording, training, live prediction, GUI |
| `ROS2/interfaces/` | custom msg/srv |

Acquisition used LocoD, a MATLAB platform developed at CBPR, with my changes for streaming to ROS2. It is not included here.

No data is included: recordings come from human participants under the study's ethics approval.

## Running

```bash
pip install -r requirements.txt
export THESIS_DATA_ROOT=/path/to/data   # default: ./data
python Pipeline.py        # FFNN
python CNN_pipeline.py    # CNN
python LSTM_pipeline.py   # LSTM
```

The scripts are research scripts: the experiment to run is chosen by editing the calls at the bottom of each file. They expect the recordings under `$THESIS_DATA_ROOT/CBPR_Recording_Folders/<session>/` and log runs to Weights & Biases.

ROS2 part:

```bash
cp -r ROS2/delsys_pkg ROS2/interfaces <ros2_ws>/src/
cd <ros2_ws> && colcon build --packages-select delsys_pkg interfaces
source install/setup.bash
export CBPR_RECORDINGS_DIR=~/CBPR_Recording_Folders
ros2 launch delsys_pkg delsys_launch.py
```

It needs the Delsys hardware and the MATLAB streaming side to do anything useful.
