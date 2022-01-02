# authors_name = 'Preetham Ganesh'
# project_title = 'Multi Sensor-based Human Activity Recognition using OpenCV and Sensor Fusion'
# email = 'preetham.ganesh2015@gmail.com'


import cv2
import pandas as pd


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


def per_video_skeleton_point_extractor(action: int,
                                       subject: int,
                                       take: int,
                                       skeleton_pose_model: str,
                                       pretrained_model_location: str):
    x = 0


def skeleton_point_extractor(actions: list,
                             subjects: list,
                             takes: list,
                             skeleton_pose_models: list):
    x = 0


def main():
    actions = [i for i in range(1, 28)]
    subjects = [i for i in range(1, 9)]
    takes = [i for i in range(1, 5)]
    skeleton_pose_models = ['coco', 'mpi']


if __name__ == '__main__':
    main()
