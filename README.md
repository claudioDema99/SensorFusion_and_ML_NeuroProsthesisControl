# Sensor Fusion and Machine Learning for Neural Prosthesis Control

**MSc thesis project (2024)** | University of Genoa and Center for Bionics and Pain Research (CBPR), Gothenburg, Sweden  
**Author:** Claudio Demaria  
**Thesis:** *Sensor Fusion for Enhancing Prosthesis Control: Overcoming the Limb Position Effect and Dynamic Movement Variability Using EMG and IMU Data*

## Overview

This research investigates the **limb position effect** in myoelectric prosthesis control: the same intended grasp can produce different electromyography (EMG) patterns as the arm changes posture or moves under load. Models relying exclusively on EMG may therefore misclassify grasping activity across different arm positions.

I developed a sensor-fusion pipeline that combines forearm/arm **EMG** with **inertial measurement unit (IMU)** measurements. Instead of passing only raw IMU readings to the classifier, the approach estimates **nine arm-orientation angles** from the accelerometer and gyroscope signals, then combines these features with EMG. The hypothesis is that a task-relevant, physically meaningful orientation representation improves classification with compact neural networks.

The project includes the experimental ML pipelines, offline comparisons, data acquisition and preprocessing, and a **ROS2-based system for sensor streaming, recording, training, and live prediction visualization**. The research was conducted in the context of transradial prosthesis control, although the experimental participants were **15 non-amputee volunteers**.

> **Key result:** In the primary *offline, participant-specific* evaluation, the feedforward neural network reached **88.4% accuracy with EMG + estimated angles**, compared with **23.9% for EMG alone** and **30.0% for EMG + raw IMU**. These are not unseen-participant or deployed prosthesis performance figures.

## Research question and contributions

1. **Orientation estimation:** estimate nine arm-orientation angles using three IMUs on the wrist, forearm and upper arm, with orientation/bias estimation based on accelerometer and gyroscope signals.
2. **Controlled input comparison:** compare EMG alone, EMG + raw IMU, and EMG + estimated angles while holding each model architecture fixed.
3. **Architectural comparison:** evaluate feedforward neural networks (FFNN), convolutional neural networks (CNN), and long short-term memory networks (LSTM): **nine model/input configurations**.
4. **Experimental validation:** analyze results from 15 participants, run statistical comparisons (two-way ANOVA and Tukey HSD), and test cross-participant generalization using subject-disjoint five-fold cross-validation.
5. **Research software:** implement offline pipelines and ROS2/MATLAB integration for acquisition, synchronized data processing, storage, and model inference.

## Experimental setup

- **Participants:** 15 healthy, non-amputee volunteers.
- **Sensors:** six EMG sensors and three IMU sensors (wrist, forearm, upper arm), acquired using the Delsys Trigno system.
- **Task:** reaching, grasping, and moving objects with different shapes and weights: paper and ceramic plates, plus empty and water-filled bottles. A rest/no-object state was also included, yielding a five-class recognition problem.
- **Signals:** the system handled a high-frequency stream (27 channels at 2,000 Hz, as described in the thesis).
- **Model constraint:** the laboratory's target prosthesis had **1 MB of flash available for the embedded model**; this motivated evaluation of compact architectures. The thesis does not establish successful deployment of the trained classifier onto that prosthesis.

### Processing and model pipeline

```text
Delsys Trigno: EMG + 3 IMUs
             |
             v
 Acquisition / preprocessing / time windows
             |
       +-----+-------------------+
       |                         |
       v                         v
 EMG feature extraction    IMU processing
                             |         |
                        Raw readings  Orientation / bias estimation
                                         |
                                   Nine arm angles
       |                         |         |
       +------------+------------+---------+
                    |
         Three input representations
     (EMG / EMG+raw IMU / EMG+angles)
                    |
             FFNN / CNN / LSTM
                    |
       Offline evaluation / live inference
```

The IMU orientation features are computed before the classifier; the FFNN/CNN/LSTM models do not themselves implement the orientation-estimation algorithm.

## Quantitative results

**Primary offline evaluation: mean metrics across the 15 participants.** Values are reported in Chapter 5, Tables 5.1–5.3 of the MSc thesis.

| Model | Inputs | Accuracy | Precision | Recall | F1-score |
|:--|:--|--:|--:|--:|--:|
| FFNN | EMG | 23.9% | 12.0% | 23.9% | 13.2% |
| FFNN | EMG + raw IMU | 30.0% | 30.7% | 29.9% | 25.8% |
| **FFNN** | **EMG + estimated angles** | **88.4%** | **88.9%** | **88.4%** | **88.4%** |
| CNN | EMG | 22.5% | 11.4% | 22.5% | 11.2% |
| CNN | EMG + raw IMU | 30.2% | 29.2% | 30.2% | 28.6% |
| **CNN** | **EMG + estimated angles** | **87.6%** | **88.4%** | **87.6%** | **87.6%** |
| LSTM | EMG | 22.1% | 8.8% | 22.1% | 11.1% |
| LSTM | EMG + raw IMU | 34.1% | 32.1% | 34.2% | 30.1% |
| **LSTM** | **EMG + estimated angles** | **82.0%** | **82.9%** | **82.0%** | **82.0%** |

The best FFNN result represents a **64.5-percentage-point accuracy improvement** over EMG-only input. This is a comparison within the reported offline setup, not a claim of improvement on external populations or real prosthesis users.

Two-way ANOVA and Tukey's HSD analyses in the thesis supported the advantage of the angle-based representation across architectures. The thesis does not provide enough detail here to justify quoting exact p-values in this README.

### Subject-independent evaluation

The thesis also evaluated the angle-based representation under **five-fold cross-validation with participants held out**: each fold trained on 12 participants and tested on the other three.

| Model | Mean held-out-participant accuracy (5 folds) |
|:--|--:|
| FFNN | 34.37% |
| CNN | 34.96% |
| LSTM | 32.90% |

These results are substantially lower than the participant-specific offline results and highlight the difficulty of **generalizing to unseen users**. Five-class chance-level accuracy would be 20% under balanced random guessing; these figures should not be presented as a practical solution to user-independent prosthesis control.

### Online / real-time performance

A ROS2 infrastructure was implemented for streaming and live inference, but the thesis reports that **online classification performance fell short of expectations**. Possible contributors include real-time noise and latency, movement variability, and limitations imposed by compact models. Accordingly, this repository represents a **research prototype**, not a clinically validated or production-ready prosthesis controller.

## Repository structure

The following paths reflect the public repository at the time this README was prepared:

| Path | Purpose |
|:--|:--|
| `Pipeline.py` | FFNN offline experiment pipeline |
| `CNN_pipeline.py` | CNN offline experiment pipeline |
| `LSTM_pipeline.py` | LSTM offline experiment pipeline |
| `Pipeline_hyperparams.py` | Hyperparameter search/experiment support |
| `cbpr_master_thesis/` | Shared Python components for preprocessing, feature extraction, models, and analysis |
| `MATLAB/` | Adaptations to the LocoD-based acquisition/recording and MATLAB-to-ROS2 streaming logic |
| `ROS2/` | ROS2 packages supporting data streaming, recording, inference, model management, and the GUI |
| `params/` | Participant-specific normalization parameters |

The MATLAB acquisition work builds upon **LocoD**, an existing platform developed at CBPR. This repository contains project-specific modifications; it is not an original implementation of LocoD from scratch.

## Running the ROS2 infrastructure

> The commands below document the repository's existing workflow; I have not re-run the hardware-dependent system in a clean environment. Model artifacts, raw participant datasets, compatible software versions, and the Delsys hardware may be needed for the full workflow.

1. Install a compatible ROS2 distribution and the workspace dependencies required by the repository's packages.
2. Copy the `delsys_pkg` and `interfaces` packages from `ROS2/` into the `src/` directory of your ROS2 workspace.
3. Build and source the workspace:

   ```bash
   cd <your-ros2-workspace>
   colcon build --packages-select delsys_pkg interfaces
   source install/setup.bash
   ```

4. Connect the Delsys Trigno acquisition system and configure MATLAB / the ROS2 Toolbox as required by the project's sensor streaming setup.
5. Launch the ROS2 application:

   ```bash
   ros2 launch delsys_pkg delsys_launch.py
   ```

The GUI supports working/live-prediction, recording, and training workflows. See the thesis's Chapter 4 for the ROS2 state machine and node responsibilities.

### Offline experiments

The offline FFNN, CNN, and LSTM scripts load recorded data, construct the three input representations, train/evaluate models, and produce performance analysis. Inputs and dataset paths are environment-specific; consult the scripts and `params/` before running them. **No turnkey reproducibility guarantee or verified dependency lockfile is claimed here.**

## Limitations and research status

- **Healthy-volunteer cohort:** experiments were carried out with non-amputees; results must not be generalized to people with limb loss without dedicated validation.
- **Participant-specific versus unseen participants:** 82.0–88.4% reflects the primary offline analysis, while subject-disjoint evaluation reached roughly 33–35%.
- **Online performance:** real-time inference infrastructure exists, but online accuracy was not yet satisfactory.
- **Hardware deployment:** architecture choices considered a 1 MB flash constraint; successful on-device prosthesis deployment was not demonstrated.
- **Data availability:** the repository contains research code; clinical/human-subject data or complete model artifacts should not be assumed publicly available.
- **Next steps:** strengthen online robustness, improve unseen-user generalization, and evaluate with prosthesis users under appropriate approvals.

## Thesis and attribution

**Claudio Demaria (2024).** *Sensor Fusion for Enhancing Prosthesis Control: Overcoming the Limb Position Effect and Dynamic Movement Variability Using EMG and IMU Data.* MSc in Robotics Engineering, University of Genoa. Research undertaken at the Center for Bionics and Pain Research, Gothenburg, Sweden. Supervisors: Prof. Maura Casadio and Dr. Fabian Just.

This README summarizes results from the thesis (not an independently peer-reviewed publication). The Delsys acquisition interface used adaptations of the pre-existing LocoD framework. The study's participant data were collected with research ethics approval as described in the thesis.

**Contact:** [Claudio Demaria](https://github.com/claudioDema99)
