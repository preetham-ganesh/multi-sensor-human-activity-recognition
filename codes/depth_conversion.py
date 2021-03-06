# authors_name = 'Preetham Ganesh'
# project_title = 'Multi Sensor-based Human Activity Recognition using OpenCV and Sensor Fusion'
# email = 'preetham.ganesh2015@gmail.com'


import pandas as pd
import numpy as np
from scipy.io import loadmat
from skeleton_points_extraction import choose_caffe_model_files
from skeleton_points_extraction import exports_processed_data


def per_video_depth_converter(depth_file: np.ndarray,
                              skeleton_pose_model: str,
                              data_version: str,
                              modality: str,
                              data_name: str,
                              skeleton_point_information: pd.DataFrame):
    """Converts depth information (based on model name) from the MATLAB file given as input. Adds the converted
    information to the existing skeleton point information and exports the dataframe into a CSV file.

        Args:
            depth_file: MATLAB depth file for the current video.
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

    # Column names for the converted depth information dictionary.
    depth_information = {'rgb_z_{}'.format(i): [] for i in range(n_skeleton_points)}
    depth_information['frame'] = [i for i in range(len(skeleton_point_information))]

    # Iterates across frames to convert depth information
    for i in range(len(skeleton_point_information)):

        # Filters current frame's depth information and resizes it.
        frame_depth = depth_file[:, :, i]

        # Filter current frame's skeleton point information
        current_frame_skeleton_points = skeleton_point_information[skeleton_point_information['frame'] == i]

        # Iterates across the skeleton points to extract depth information.
        for j in range(n_skeleton_points):
            skeleton_point_x_value = int(current_frame_skeleton_points['rgb_x_{}'.format(j)]) // 2
            skeleton_point_y_value = int(current_frame_skeleton_points['rgb_y_{}'.format(j)]) // 2
            current_skeleton_point_depth_value = frame_depth[skeleton_point_y_value][skeleton_point_x_value]
            depth_information['rgb_z_{}'.format(j)].append(current_skeleton_point_depth_value / 100)

    # Iterates across the depth information, imputes mean values if there are any zeros, and adds them to current
    # version of skeleton point information
    for i in range(n_skeleton_points):
        current_depth_information = depth_information['rgb_z_{}'.format(i)]
        n_non_zeros = len(current_depth_information) - current_depth_information.count(0)

        # In some cases the depth camera would have registered depth as 0 in all the frames for a particular skeleton
        # point
        if n_non_zeros != 0:
            imputed_depth_information = [round(sum(current_depth_information) / n_non_zeros, 2)
                                         if current_depth_information[j] == 0 else current_depth_information[j] for j in
                                         range(len(current_depth_information))]
            depth_information['rgb_z_{}'.format(i)] = imputed_depth_information

    # Converts depth information dictionary into a pandas dataframe.
    depth_information_df = pd.DataFrame(depth_information, columns=['frame'] + ['rgb_z_{}'.format(i) for i in
                                                                                range(n_skeleton_points)])

    # Exports the updated version of the skeleton point information into a CSV file.
    exports_processed_data(depth_information_df, data_version, modality, '{}_{}'.format(data_name, skeleton_pose_model))


def depth_converter(n_actions: int,
                    n_subjects: int,
                    n_takes: int,
                    skeleton_pose_models: list):
    """Converts MATLAB depth information and adds them to the skeleton point information for all actions, subjects, and
    takes.

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
    modality = 'depth'
    data_version = 'processed_data'

    # Iterates across all actions, subjects and takes in the dataset.
    for i in range(1, n_actions + 1):
        for j in range(1, n_subjects + 1):
            for k in range(1, n_takes + 1):
                for m in range(len(skeleton_pose_models)):
                    data_name = 'a{}_s{}_t{}'.format(i, j, k)

                    # Imports MATLAB depth file and the skeleton point information for the current action, subject and
                    # take.
                    try:
                        depth_file = loadmat('../data/original_data/{}/{}_{}.mat'.format(modality.title(), data_name,
                                                                                         modality))
                        skeleton_point_information = pd.read_csv('../data/{}/rgb/{}_{}.csv'.format(data_version,
                                                                                                   data_name,
                                                                                                   skeleton_pose_models[m]))
                        per_video_depth_converter(depth_file['d_depth'], skeleton_pose_models[m], data_version,
                                                  modality, data_name, skeleton_point_information)
                    except FileNotFoundError:
                        print('Video file for {}_{} does not exist.'.format(data_name, skeleton_pose_models[m]))
                    print()


def main():
    print()
    n_actions = 27
    n_subjects = 8
    n_takes = 4
    skeleton_pose_models = ['coco', 'mpi']
    depth_converter(n_actions, n_subjects, n_takes, skeleton_pose_models)


if __name__ == '__main__':
    main()
