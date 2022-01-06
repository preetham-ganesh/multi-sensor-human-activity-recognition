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
        normalized_sensor_values = [0 for _ in range(len(processed_sensor_values))]
    else:
        normalized_sensor_values = [(processed_sensor_values[i] - minimum_value) / (maximum_value - minimum_value)
                                    for i in range(len(processed_sensor_values))]
    return normalized_sensor_values


def per_video_data_normalizer(skeleton_point_information: pd.DataFrame,
                              skeleton_pose_model: str,
                              data_version: str,
                              modality: str,
                              data_name: str):
    """Performs min-max normalization on the skeleton point information for the current video, and filters the dataframe
    columns based on the modality. Exports the dataframe into a CSV file.

        Args:
            skeleton_point_information: Current version of skeleton point information for the current video.
            skeleton_pose_model: Model name which will be used to import model details.
            data_version: Current version of the dataframe.
            modality: Current modality of the dataframe.
            data_name: Name with which the dataframe should be saved.

        Returns:
            None
    """
    # Imports number of skeleton points model based on the skeleton_pose_model given as input.
    _, _, n_skeleton_points = choose_caffe_model_files(skeleton_pose_model)

    # Column names for the converted skeleton point information.
    skeleton_point_column_names = list(skeleton_point_information.columns)

    # Iterates across skeleton point column names for performing data normalization
    for i in range(len(skeleton_point_column_names)):
        if skeleton_point_column_names[i] != 'frame':
            skeleton_point_information[skeleton_point_column_names[i]] = min_max_normalizer(
                skeleton_point_information[skeleton_point_column_names[i]])

    # Exports the updated version of the skeleton point information into a CSV file.
    exports_processed_data(skeleton_point_information, data_version, modality, '{}_{}'.format(data_name,
                                                                                              skeleton_pose_model))


def data_normalizer(n_actions: int,
                    n_subjects: int,
                    n_takes: int,
                    skeleton_pose_models: list,
                    modalities: list):
    """Performs min-max normalization on the skeleton point information for all actions, subjects, and takes.

        Args:
            n_actions: Total number of actions in the original dataset.
            n_subjects: Total number of subjects in the original dataset.
            n_takes: Total number of takes in the original dataset.
            skeleton_pose_models: Model names which will be used to import model details.
            modalities: List of modalities available in the dataset.

        Returns:
            None.
    """
    data_version = 'normalized_data'

    # Iterates across all actions, subjects, takes, skeleton pose models and modalities in the dataset.
    for i in range(1, n_actions + 1):
        for j in range(1, n_subjects + 1):
            for k in range(1, n_takes + 1):
                for m in range(len(skeleton_pose_models)):
                    for n in range(len(modalities)):
                        data_name = 'a{}_s{}_t{}'.format(i, j, k)

                        # Imports skeleton point information file and performs min-max normalization.
                        try:
                            skeleton_point_information = pd.read_csv(
                                '../data/processed_data/{}/{}_{}.csv'.format(modalities[n], data_name,
                                                                             skeleton_pose_models[m]))
                            per_video_data_normalizer(skeleton_point_information, skeleton_pose_models[m], data_version,
                                                      modalities[n], data_name)
                        except FileNotFoundError:
                            print('Video file for {}_{} does not exist.'.format(data_name, skeleton_pose_models[m]))
                        print()


def main():
    print()
    n_actions = 27
    n_subjects = 8
    n_takes = 4
    skeleton_pose_models = ['coco', 'mpi']
    modalities = ['rgb', 'depth', 'inertial']
    data_normalizer(n_actions, n_subjects, n_takes, skeleton_pose_models, modalities)


if __name__ == '__main__':
    main()
