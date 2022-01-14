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

### Best ML model for each Modality combination

| Modalities | Skeleton Pose Model | ML Model | Parameters | Accuracy | Precision | Recall | F1 |
| - | - | - | - | - | - | - | - |
| RGB |	COCO | Gradient Boosting | n_estimators=100, max_depth=3 | 53.271 | 57.358 | 65.517 | 57.711 |
| RGB | MPI | Gradient Boosting | n_estimators=100, max_depth=4 | 52.336 | 57.159 | 61.538 | 57.035 |
| Depth | COCO | Support Vector | kernel=rbf | 38.318 | 39.534 | 43.158 | 38.788 |
| Depth | MPI | Gradient Boosting | n_estimators=60, max_depth=3 | 38.318 | 40.838 | 41.414 | 36.825 |
| Inertial | COCO | Support Vector | kernel=rbf | 65.421 | 66.003 | 70.707 | 66.765 |
| Inertial | MPI | Support Vector | kernel=rbf | 65.421 | 66.003 | 70.707 | 66.765 |
| RGB, Depth | COCO | Gradient Boosting | n_estimators=100, max_depth=2 | 52.336 | 63.536 | 58.947 | 55.374 |
| RGB, Depth | MPI | Gradient Boosting | n_estimators=80, max_depth=5 | 58.879 | 61.658 | 63.636 | 60.432 |
| RGB, Inertial | COCO | Gradient Boosting | n_estimators=100, max_depth=4 | 64.486| 70.309 | 69.697 | 65.223 |
| RGB, Inertial | MPI | Gradient Boosting | n_estimators=100, max_depth=5 | 70.093 | 76.755 | 72.816 | 71.602 |
| Depth, Inertial | COCO | Support Vector | kernel=poly | 57.944 | 60.58 | 60.194 | 56.682 |
| Depth, Inertial | MPI | Support Vector | kernel=poly | 52.336 | 59.137 | 58.947 | 53.151 |
| RGB, Depth, Inertial | COCO | Gradient Boosting | n_estimators=100, max_depth=3 | 59.813 | 65.695 | 67.368 | 62.902 |
| RGB, Depth, Inertial | MPI | Gradient Boosting | n_estimators=100, max_depth=5 | 64.486 | 74.075 | 72.632 | 70.376 |

## Future Work

## Support

For any queries regarding the repository contact 'preetham.ganesh2015@gmail.com'.

## License

[MIT](https://choosealicense.com/licenses/mit/)
