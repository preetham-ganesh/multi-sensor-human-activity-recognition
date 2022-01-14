# Multi Sensor-based Human Activity Recognition using OpenCV and Sensor Fusion

Author: [Preetham Ganesh](https://www.linkedin.com/in/preethamganesh/)

## Contents

## Description

- It is an application for video-based classification of actions in UTD-MHAD dataset which attempts to capture minute changes in the subject's actions.
- The project aims to develop general ML models and Ensemble models that classifies actions based on videos.
- The project also aims at finding which combination of modalities in the dataset (RGB, Depth, Inertial) produce best possible results.

## Requirement Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements

Requires: Python 3.6.

```bash
# Clone this repository
git clone https://github.com/preetham-ganesh/multi-sensor-human-activity-recognition.git
cd multi-sensor-human-activity-recognition

# Create a Conda environment with dependencies
conda env create -f environment.yml
conda activate mshar_env
pip install -r requirements.txt
```

## Data Pre-processing

### Original Data

- The data was downloaded from UTD-MHAD website [[Link]](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html).
- The dataset contains videos from 3 modalities, namely: RGB, Depth, and Inertial.
- The downloaded dataset should be saved in the 'data' directory under 'original_data' sub-directory which should be created. 
- The total number of actions, subjects and takes in the dataset are 27, 8, 4.

### Skeleton Point Extraction

```bash
python3 skeleton_points_extraction.py
```

### Depth & Inertial Conversion

```bash
python3 depth_conversion.py
python3 inertial_conversion.py
```

### Data Normalization

```bash
python3 data_normalization.py
```

## Model Training & Testing

```bash
python3 model_training_testing.py
python3 find_best_model_per_combination.py
```

## Future Work

## Support

For any queries regarding the repository contact 'preetham.ganesh2015@gmail.com'.

## License

[MIT](https://choosealicense.com/licenses/mit/)
