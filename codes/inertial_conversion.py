# authors_name = 'Preetham Ganesh'
# project_title = 'Multi Sensor-based Human Activity Recognition using OpenCV and Sensor Fusion'
# email = 'preetham.ganesh2015@gmail.com'


import pandas as pd
import numpy as np
from scipy.io import loadmat
from skeleton_points_extraction import choose_caffe_model_files
from skeleton_points_extraction import exports_processed_data


def per_video_inertial_converter(inertial_file: np.ndarray,
                                 skeleton_pose_model: str,
                                 data_version: str,
                                 modality: str,
                                 data_name: str,
                                 skeleton_point_information: pd.DataFrame):
    """Converts inertial information (based on model name) from the MATLAB file given as input. Adds the converted
    information to the existing skeleton point information and exports the dataframe into a CSV file.

        Args:
            inertial_file: MATLAB inertial file for the current video.
            skeleton_pose_model: Model name which will be used to import model details.
            data_version: Current version of the dataframe.
            modality: Current modality of the dataframe.
            data_name: Name with which the dataframe should be saved.
            skeleton_point_information: Current version of skeleton point information for the current video.

        Returns:
            None.
    """
    # Imports number of skeleton points model based on the skeleton_pose_model given as input.
    _, _, n_skeleton_points = choose_caffe_model_files(skeleton_pose_model)

    # Column names for the converted inertial information.
    inertial_columns = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']

    # Iterates across the column names for inertial sensors information.
    for i in range(len(inertial_columns)):

        # Filters information of current inertial sensor.
        current_inertial = inertial_file[:, i]
        current_inertial_moving_average = []

        # Iterates across the frames of current inertial sensor information and computes moving average.
        for j in range(0, inertial_file.shape[0], 7):
            current_inertial_moving_average.append(round(float(np.mean(current_inertial[j: j + 4])), 5))
            current_inertial_moving_average.append(round(float(np.mean(current_inertial[j + 3: j + 7])), 5))

        # Adds the moving average information for the current inertial sensor information to the skeleton point
        # information
        skeleton_point_information[inertial_columns[i]] = current_inertial_moving_average[:len(
            skeleton_point_information)]

    # Exports the updated version of the skeleton point information into a CSV file.
    exports_processed_data(skeleton_point_information, data_version, modality, '{}_{}'.format(data_name,
                                                                                              skeleton_pose_model))


def inertial_converter(n_actions: int,
                       n_subjects: int,
                       n_takes: int,
                       skeleton_pose_models: list):
    modality = 'inertial'
    data_version = 'processed_data'

    # Iterates across all actions, subjects and takes in the dataset.
    for i in range(1, n_actions + 1):
        for j in range(1, n_subjects + 1):
            for k in range(1, n_takes + 1):
                for m in range(len(skeleton_pose_models)):
                    data_name = 'a{}_s{}_t{}'.format(i, j, k)
                    inertial_file = loadmat(
                        '../data/original_data/{}/{}_{}.mat'.format(modality.title(), data_name, modality))
                    skeleton_file = pd.read_csv('../data/processed_data/{}/{}_{}.csv'.format('rgb', data_name,
                                                                                             skeleton_pose_models[m]))
                    #print(data_name, inertial_file['d_iner'].shape[0], len(skeleton_file),
                    #      inertial_file['d_iner'].shape[0] / len(skeleton_file))
                    per_video_inertial_converter(inertial_file['d_iner'], skeleton_pose_models[m], data_version, modality,
                                                 data_name, skeleton_file)


def main():
    print()
    n_actions = 1
    n_subjects = 1
    n_takes = 1
    skeleton_pose_models = ['coco', 'mpi']
    inertial_converter(n_actions, n_subjects, n_takes, skeleton_pose_models)


if __name__ == '__main__':
    main()

