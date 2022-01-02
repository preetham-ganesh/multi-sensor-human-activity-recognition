# authors_name = 'Preetham Ganesh'
# project_title = 'Multi Sensor-based Human Activity Recognition using OpenCV and Sensor Fusion'
# email = 'preetham.ganesh2015@gmail.com'


import cv2
import pandas as pd
import time
import os


def choose_caffe_model_files(skeleton_pose_model: str):
    """Returns Caffe model files name based on the given input model name.

        Args:
            skeleton_pose_model: Caffe model name which contains either 'COCO' or 'MPI'.

        Returns:
            A tuple that contains the names for proto & weights files and the number of skeleton points.
    """
    if skeleton_pose_model == 'coco':
        proto_file_name = 'pose_deploy_linevec.prototxt'
        weights_file_name = 'pose_iter_440000.caffemodel'
        n_skeleton_points = 18
    else:
        proto_file_name = 'pose_deploy_linevec_faster_4_stages.prototxt'
        weights_file_name = 'pose_iter_160000.caffemodel'
        n_skeleton_points = 14
    return proto_file_name, weights_file_name, n_skeleton_points


def check_directory_existence(directory_path: str,
                              sub_directory: str):
    """Concatenates directory path and sub_directory. Checks if the directory path exists; if not, it will create the
    directory.

        Args:
            directory_path: Current directory path
            sub_directory: Directory that needs to be checked if it exists or not.

        Return:
            Newly concatenated directory path
    """
    directory_path = '{}/{}'.format(directory_path, sub_directory)

    # If the concatenated directory path does not exist then the sub directory is created.
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    return directory_path


def exports_processed_data(data: pd.DataFrame,
                           data_version: str,
                           modality: str,
                           data_name: str):
    """Exports processed data into CSV files to the location based data_version and modality.

        Args:
            data: Pre-processed dataframe for the current action, subject and take.
            data_version: Current version of the dataframe.
            modality: Current modality of the dataframe.
            data_name: Name with which the dataframe should be saved.

        Returns:
            None.
    """
    # Checks for the existence of the directory path
    directory_path = check_directory_existence('../data', data_version)
    directory_path = check_directory_existence(directory_path, modality)

    # Concatenates file name & directory path, and saves the dataframe as the CSV file.
    file_path = '{}/{}.csv'.format(directory_path, data_name)
    data.to_csv(file_path, index=False)


def per_video_skeleton_point_extractor(video_capture: cv2.VideoCapture,
                                       skeleton_pose_model: str,
                                       model_location: str,
                                       data_version: str,
                                       modality: str,
                                       data_name: str):
    """Extracts skeleton points (based on model name) from the video given as input. Exports the extracted skeleton
    point dataframe into a CSV file.

        Args:
            video_capture: Video taken as input given by the user or current action, subject and take.
            skeleton_pose_model: Model name which will be used to import model details.
            model_location: Location where the pretrained model files are saved.
            data_version: Current version of the dataframe.
            modality: Current modality of the dataframe.
            data_name: Name with which the dataframe should be saved.

        Returns:
            None.
    """
    # Imports model details based on the skeleton_pose_model given as input. Uses the details to import the caffe model.
    proto_file_name, weights_file_name, n_skeleton_points = choose_caffe_model_files(skeleton_pose_model)
    proto_file_location = '{}/{}'.format(model_location, proto_file_name)
    weights_file_location = '{}/{}'.format(model_location, weights_file_name)
    caffe_net = cv2.dnn.readNet(weights_file_location, proto_file_location)

    # Column names for the extracted skeleton points dictionary.
    skeleton_point_column_names = ['frame']
    skeleton_point_column_names += ['rgb_x_{}'.format(i) for i in range(n_skeleton_points)]
    skeleton_point_column_names += ['rgb_y_{}'.format(i) for i in range(n_skeleton_points)]
    skeleton_point_information = {skeleton_point_column_names[i]: [] for i in range(len(skeleton_point_column_names))}

    # Iterates across the frames in the loaded video.
    frame_number = 0
    while True:
        return_value, frame = video_capture.read()
        extraction_start_time = time.time()
        if not return_value:
            break

        # Resizes the input frame to model input shape, and passes the input to the caffe model.
        model_input = cv2.dnn.blobFromImage(frame, 1/255, (640, 480), (0, 0, 0), swapRB=False, crop=False)
        caffe_net.setInput(model_input)
        model_output = caffe_net.forward()

        # Extracts the height and width from model output.
        model_output_height, model_output_width = model_output.shape[2], model_output.shape[3]

        # Iterates across the extracted skeleton to pre-process and save it in the dictionary.
        skeleton_point_information['frame'].append(frame_number)
        for i in range(n_skeleton_points):
            probability_map = model_output[0, i, :, :]
            minimum_value, probability, minimum_location, point = cv2.minMaxLoc(probability_map)
            skeleton_point_x_value = (640 * point[0]) / model_output_width
            skeleton_point_y_value = (480 * point[1]) / model_output_height
            skeleton_point_information['rgb_x_{}'.format(i)].append(int(skeleton_point_x_value))
            skeleton_point_information['rgb_y_{}'.format(i)].append(int(skeleton_point_y_value))
        frame_number += 1
        print('video_name={}, frame={}, time_taken={} sec'.format(data_name, frame_number,
                                                                  round(time.time() - extraction_start_time, 3)))

    # Converts the extracted skeleton point dictionary into dataframe, and exports it to CSV file.
    skeleton_point_information_df = pd.DataFrame(skeleton_point_information, columns=skeleton_point_column_names)
    exports_processed_data(skeleton_point_information_df, data_version, modality, data_name)


def skeleton_point_extractor(actions: list,
                             subjects: list,
                             takes: list,
                             skeleton_pose_models: list):
    """per_video_skeleton_point_extractor(1, 1, 1, 'coco', '../results/pretrained_model_files', )
    """
    data_location = '../data/original_data/RGB/'
    video_capture = cv2.VideoCapture('{}/a{}_s{}_t{}_color.avi'.format(data_location, str(1), str(1), str(1)))
    per_video_skeleton_point_extractor(video_capture, 'coco', '../results/pretrained_model_files', 'processed_data',
                                       'rgb', 'a{}_s{}_t{}'.format(str(1), str(1), str(1)))


def main():
    actions = [i for i in range(1, 28)]
    subjects = [i for i in range(1, 9)]
    takes = [i for i in range(1, 5)]
    skeleton_pose_models = ['coco', 'mpi']
    skeleton_point_extractor(actions, subjects, takes, skeleton_pose_models)


if __name__ == '__main__':
    main()
