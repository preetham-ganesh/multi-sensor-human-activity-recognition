# authors_name = 'Preetham Ganesh'
# project_title = 'Multi Sensor-based Human Activity Recognition using OpenCV and Sensor Fusion'
# email = 'preetham.ganesh2015@gmail.com'

import pandas as pd
import numpy as np
from skeleton_points_extraction import choose_caffe_model_files
from skeleton_points_extraction import exports_processed_data


def min_max_normalizer(processed_sensor_values: list):
    """Performs min-max normalization on the processed sensor values.

        Args:
            processed_sensor_values: Sensor values processed in previous

        Returns:
            Min-max normalized sensor values for the current sensor.
    """
    minimum_value = min(processed_sensor_values)
    maximum_value = max(processed_sensor_values)

    # If all the values in the processed sensor values is 0 then the normalization will not be done else normalization
    # will be performed.
    if maximum_value - minimum_value == 0:
        normalized_sensor_values = processed_sensor_values
    else:
        normalized_sensor_values = [(processed_sensor_values[i] - minimum_value) / (maximum_value - minimum_value)
                                    for i in range(len(processed_sensor_values))]
    return normalized_sensor_values


def per_video_data_normalizer(skeleton_point_information: pd.DataFrame,
                              skeleton_pose_model: str,
                              data_version: str,
                              modality: str,
                              data_name: str):
    _, _, n_skeleton_points = choose_caffe_model_files(skeleton_pose_model)
    skeleton_point_column_names = ['frame']
    skeleton_point_column_names += ['rgb_x_{}'.format(i) for i in range(n_skeleton_points)]
    skeleton_point_column_names += ['rgb_y_{}'.format(i) for i in range(n_skeleton_points)]
    skeleton_point_column_names += ['rgb_z_{}'.format(i) for i in range(n_skeleton_points)]
    skeleton_point_column_names += ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
    for i in range(len(skeleton_point_column_names)):
        skeleton_point_information[skeleton_point_column_names[i]] = 0


def data_normalizer(n_actions: int,
                    n_subjects: int,
                    n_takes: int,
                    skeleton_pose_models: list):
    x = 0


def main():
    n_actions = 27
    n_subjects = 8
    n_takes = 4
    skeleton_pose_models = ['coco', 'mpi']
    data_normalizer(n_actions, n_subjects, n_takes, skeleton_pose_models)


if __name__ == '__main__':
    main()
