# authors_name = 'Preetham Ganesh'
# project_title = 'Multi Sensor-based Human Activity Recognition using OpenCV and Sensor Fusion'
# email = 'preetham.ganesh2015@gmail.com'


import cv2
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
    """Converts depth information (based on model name) from the matlab file given as input. Adds the converted
    information to the existing skeleton point information and exports the dataframe into a CSV file.

        Args:
            depth_file: Matlab depth file for the current video.
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

    # Iterates across frames to convert depth information
    for i in range(len(skeleton_point_information)):

        # Filters current frame's depth information and resizes it.
        frame_depth = depth_file[:, :, i]
        frame_depth_resized = cv2.resize(frame_depth, dsize=(640, 480))

        # Filter current frame's skeleton point information
        current_frame_skeleton_points = skeleton_point_information[skeleton_point_information['frame'] == i]

        # Iterates across the skeleton points to extract depth information.
        for j in range(n_skeleton_points):
            skeleton_point_x_value = int(current_frame_skeleton_points['rgb_x_{}'.format(j)])
            skeleton_point_y_value = int(current_frame_skeleton_points['rgb_y_{}'.format(j)])
            current_skeleton_point_depth_value = frame_depth_resized[skeleton_point_x_value][skeleton_point_y_value]
            depth_information['rgb_z_{}'.format(j)].append(current_skeleton_point_depth_value)

    # Iterates across the depth information to add them to current version of skeleton point information
    for i in range(n_skeleton_points):
        skeleton_point_information['rgb_z_{}'.format(i)] = depth_information['rgb_z_{}'.format(i)]

    # Exports the updated version of the skeleton point information into a CSV file.
    exports_processed_data(skeleton_point_information, data_version, modality, '{}_{}'.format(data_name,
                                                                                              skeleton_pose_model))


def depth_converter(n_actions: int,
                    n_subjects: int,
                    n_takes: int,
                    skeleton_pose_models: list):
    depth_file = loadmat('../data/original_data/Depth/a1_s1_t1_depth.mat')
    data_name = 'a1_s1_t1'
    skeleton_point_data = pd.read_csv('../data/processed_data/rgb/{}_{}.csv'.format(data_name, skeleton_pose_models[0]))
    per_video_depth_converter(depth_file['d_depth'], 'coco', 'processed_data', 'depth', 'a1_s1_t1', skeleton_point_data)



def main():
    print()
    n_actions = 5
    n_subjects = 5
    n_takes = 4
    skeleton_pose_models = ['coco', 'mpi']
    depth_converter(n_actions, n_subjects, n_takes, skeleton_pose_models)


if __name__ == '__main__':
    main()
