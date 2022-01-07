# authors_name = 'Preetham Ganesh'
# project_title = 'Multi Sensor-based Human Activity Recognition using OpenCV and Sensor Fusion'
# email = 'preetham.ganesh2015@gmail.com'


import pandas as pd
import itertools
from skeleton_points_extraction import choose_caffe_model_files


def list_combination_generator(modalities: list):
    """Generates combinations for items in the given list.

        Args:
            modalities: List of modalities available in the dataset.

        Returns:
            Combinations of items in the given list.
    """
    modality_combinations = list()

    # Iterates across modalities to generate combinations based on length.
    for length in range(1, len(modalities) + 1):

        # Generate combinations for the current length.
        current_length_combinations = itertools.combinations(modalities, length)

        # Iterates across the generated combinations to convert it into a list.
        for combination in current_length_combinations:
            current_combination_list = list()
            for k in combination:
                current_combination_list.append(k)
            modality_combinations.append(current_combination_list)
    return modality_combinations


def column_name_generator(modality: str,
                          skeleton_pose_model: str):
    # Imports number of skeleton points model based on the skeleton_pose_model given as input.
    _, _, n_skeleton_points = choose_caffe_model_files(skeleton_pose_model)





def data_combiner(n_actions: int,
                  subject_ids: list,
                  n_takes: int,
                  skeleton_pose_model: str):
    # Imports number of skeleton points model based on the skeleton_pose_model given as input.
    _, _, n_skeleton_points = choose_caffe_model_files(skeleton_pose_model)

    """for i in range(1, n_actions + 1):
        for j in subject_ids:
            for k in range(1, n_takes + 1):
                data_name = 'a{}_s{}_t{}_{}'.format(i, j, k, skeleton_pose_model)

                # Imports skeleton point information file and performs min-max normalization.
                try:"""


def main():
    print()
    n_actions = 27
    n_subjects = 8
    n_takes = 4
    skeleton_pose_models = ['coco', 'mpi']
    modalities = ['rgb', 'depth', 'inertial']
    train_subject_ids = [i for i in range(1, 7)]
    validation_subject_ids = [7]
    test_subject_ids = [8]
    data_combiner(n_actions, train_subject_ids, n_takes, skeleton_pose_models[0])


if __name__ == '__main__':
    main()
