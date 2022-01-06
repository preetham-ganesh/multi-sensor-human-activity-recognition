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

    # Column names for the converted inertial information dictionary.
    inertial_columns = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
    inertial_information = {inertial_columns[i]: [] for i in range(len(inertial_columns))}
    inertial_information['frame'] = [i for i in range(len(skeleton_point_information))]

    # Iterates across the column names for inertial sensors information.
    for i in range(len(inertial_columns)):

        # Filters information of current inertial sensor.
        current_inertial = inertial_file[:, i]
        current_inertial_moving_average = []

        # Iterates across the frames of current inertial sensor information and computes moving average.
        for j in range(0, inertial_file.shape[0], 3):
            current_inertial_moving_average.append(round(float(np.mean(current_inertial[j: j + 3])), 5))

        # Adds the moving average information for the current inertial sensor information to the skeleton point
        # information
        inertial_information[inertial_columns[i]] = current_inertial_moving_average[:len(skeleton_point_information)]

    # Converts inertial information dictionary into a pandas dataframe.
    inertial_information_df = pd.DataFrame(inertial_information, columns=['frame'] + inertial_columns)

    # Exports the updated version of the skeleton point information into a CSV file.
    exports_processed_data(inertial_information_df, data_version, modality, '{}_{}'.format(data_name,
                                                                                           skeleton_pose_model))


def inertial_converter(n_actions: int,
                       n_subjects: int,
                       n_takes: int,
                       skeleton_pose_models: list):
    """Converts MATLAB inertial information and adds them to the skeleton point information for all actions, subjects,
    and takes.

        Args:
            n_actions: Total number of actions in the original dataset.
            n_subjects: Total number of subjects in the original dataset.
            n_takes: Total number of takes in the original dataset.
            skeleton_pose_models: Model names which will be used to import model details.

        Returns:
            None.

        Raises:
            FileNotFoundError: If a particular video file is not found.
    """
    modality = 'inertial'
    data_version = 'processed_data'

    # Iterates across all actions, subjects and takes in the dataset.
    for i in range(1, n_actions + 1):
        for j in range(1, n_subjects + 1):
            for k in range(1, n_takes + 1):
                for m in range(len(skeleton_pose_models)):
                    data_name = 'a{}_s{}_t{}'.format(i, j, k)

                    # Imports MATLAB inertial file and the skeleton point information for the current action, subject &
                    # take.
                    try:
                        inertial_file = loadmat('../data/original_data/{}/{}_{}.mat'.format(modality.title(), data_name,
                                                                                            modality))
                        skeleton_file = pd.read_csv('../data/processed_data/{}/{}_{}.csv'.format('depth', data_name,
                                                                                                 skeleton_pose_models[m]))
                        per_video_inertial_converter(inertial_file['d_iner'], skeleton_pose_models[m], data_version,
                                                     modality, data_name, skeleton_file)
                    except FileNotFoundError:
                        print('Video file for {}_{} does not exist.'.format(data_name, skeleton_pose_models[m]))
                    print()


def main():
    print()
    n_actions = 27
    n_subjects = 8
    n_takes = 4
    skeleton_pose_models = ['coco', 'mpi']
    inertial_converter(n_actions, n_subjects, n_takes, skeleton_pose_models)


if __name__ == '__main__':
    main()
