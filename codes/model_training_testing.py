# authors_name = 'Preetham Ganesh'
# project_title = 'Multi Sensor-based Human Activity Recognition using OpenCV and Sensor Fusion'
# email = 'preetham.ganesh2015@gmail.com'


import pandas as pd
import itertools
from skeleton_points_extraction import choose_caffe_model_files


def list_combinations_generator(modalities: list):
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


def data_combiner(n_actions: int,
                  subject_ids: list,
                  n_takes: int,
                  modalities: list,
                  skeleton_pose_model: str):
    """Combines skeleton point information for all actions, all takes, given list of subject ids and given list of
    modalities.

        Args:
            n_actions: Total number of actions in the original dataset.
            subject_ids: List of subjects in the current set.
            n_takes: Total number of takes in the original dataset.
            modalities: Current combination of modalities.
            skeleton_pose_model: Current skeleton pose model name which will be used to import skeleton point
                                 information.

        Returns:
            Pandas dataframe which contains the combined skeleton point information for all actions, all takes, given
            list of subject ids and given list of modalities.
    """
    combined_modality_skeleton_information = pd.DataFrame()

    # Iterates across actions, subject_ids, takes, and modalities to combine skeleton point information.
    for i in range(1, n_actions + 1):
        for j in range(len(subject_ids)):
            for k in range(1, n_takes + 1):
                data_name = 'a{}_s{}_t{}'.format(i, subject_ids[j], k)

                # Iterates across modalities to import skeleton point information file and adds it to
                # combined_modality_skeleton_information. If file not found, it moves on to the next combination.
                try:

                    # Imports 1st modality's skeleton point information for current data_name and skeleton_pose_model.
                    current_data_name_modality_information = pd.read_csv('../data/normalized_data/{}/{}_{}.csv'.format(
                        modalities[0], data_name, skeleton_pose_model))
                except FileNotFoundError:
                    continue

                # Since, length of modalities in each combination is different. Hence, if length of modalities is
                # greater than 1, then the imported skeleton point information for other modalities will be merged to
                # the skeleton point information for the first modality.
                if len(modalities) != 1:
                    for m in range(1, len(modalities)):
                        current_skeleton_point_information = pd.read_csv('../data/normalized_data/{}/{}_{}.csv'.format(
                            modalities[m], data_name, skeleton_pose_model))
                        current_data_name_modality_information = pd.merge(current_data_name_modality_information,
                                                                          current_skeleton_point_information,
                                                                          on='frame', how='outer')

                # Removes frame column from the imported skeleton point information.
                current_data_name_modality_information = current_data_name_modality_information.drop(columns=['frame'])

                # Adds action column to the imported skeleton point information.
                current_data_name_modality_information['action'] = [i for _ in range(len(
                    current_data_name_modality_information))]

                # Appends currently imported & modified skeleton point information to the combined modality skeleton
                # point information
                combined_modality_skeleton_information = combined_modality_skeleton_information.append(
                    current_data_name_modality_information)
    return combined_modality_skeleton_information


def model_training_testing(n_actions: int,
                           n_subjects: int,
                           n_takes: int,
                           skeleton_pose_models: list,
                           modalities: list):
    modality_combinations = list_combinations_generator(modalities)
    train_subject_ids = [i for i in range(1, 7)]
    validation_subject_ids = [7]
    test_subject_ids = [8]
    train_skeleton_point_information = data_combiner(n_actions, train_subject_ids, n_takes, modality_combinations[4],
                                                     skeleton_pose_models[0])
    validation_skeleton_point_information = data_combiner(n_actions, validation_subject_ids, n_takes,
                                                          modality_combinations[4], skeleton_pose_models[0])
    test_skeleton_point_information = data_combiner(n_actions, test_subject_ids, n_takes, modality_combinations[4],
                                                    skeleton_pose_models[0])
    print(len(train_skeleton_point_information), len(validation_skeleton_point_information),
          len(test_skeleton_point_information))
    #data_combiner(n_actions, train_subject_ids, n_takes, modality_combinations, skeleton_pose_models[0)


def main():
    print()
    n_actions = 27
    n_subjects = 8
    n_takes = 4
    skeleton_pose_models = ['coco', 'mpi']
    modalities = ['rgb', 'depth', 'inertial']
    model_training_testing(n_actions, n_subjects, n_takes, skeleton_pose_models, modalities)


if __name__ == '__main__':
    main()
