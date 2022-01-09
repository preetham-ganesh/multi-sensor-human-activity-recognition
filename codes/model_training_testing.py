# authors_name = 'Preetham Ganesh'
# project_title = 'Multi Sensor-based Human Activity Recognition using OpenCV and Sensor Fusion'
# email = 'preetham.ganesh2015@gmail.com'


import numpy as np
import pandas as pd
import itertools
import logging
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import ParameterGrid
from skeleton_points_extraction import choose_caffe_model_files


logging.getLogger('sklearn').setLevel(logging.FATAL)


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
            A pandas dataframe which contains combined skeleton point information for all actions, all takes, given list
            of subject ids and given list of modalities.
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

                # Adds data_name to the imported skeleton point information.
                current_data_name_modality_information['data_name'] = [data_name for _ in range(len(
                    current_data_name_modality_information))]

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


def calculate_metrics(actual_values: np.ndarray,
                      predicted_values: np.ndarray):
    """Using actual_values, predicted_values calculates metrics such as accuracy, balanced accuracy, precision, recall,
    and f1 scores.

        Args:
            actual_values: Actual action labels in the dataset
            predicted_values: Action labels predicted by the currently trained model

        Returns:
            Dictionary contains keys as score names and values as scores which are floating point values.
    """
    return {'accuracy_score': round(accuracy_score(actual_values, predicted_values) * 100, 3),
            'balanced_accuracy_score': round(balanced_accuracy_score(actual_values, predicted_values) * 100, 3),
            'precision_score': round(precision_score(actual_values, predicted_values, average='weighted',
                                                     labels=np.unique(predicted_values)) * 100, 3),
            'recall_score': round(recall_score(actual_values, predicted_values, average='weighted',
                                               labels=np.unique(predicted_values)) * 100, 3),
            'f1_score': round(f1_score(actual_values, predicted_values, average='weighted',
                                       labels=np.unique(predicted_values)) * 100, 3)}


def retrieve_hyperparameters(current_model_name: str):
    """Based on the current_model_name returns a list of hyperparameters used for optimizing the model (if necessary).

        Args:
            current_model_name: Name of the model currently expected to be trained

        Returns:
            A dictionary containing the hyperparameter name and the values that will be used to optimize the model
    """
    # For logistic_regression, the hyperparameter tuned is penalty.
    if current_model_name == 'logistic_regression':
        parameters = {'penalty': ['l1', 'l2', 'elasticnet', 'none']}

    # For gaussian_naive_bayes, none of the hyperparameters are tuned.
    elif current_model_name == 'gaussian_naive_bayes':
        parameters = {'None': ['None']}

    # For support_vector_classifier, the hyperparameter tuned is kernel.
    elif current_model_name == 'support_vector_classifier':
        parameters = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

    # For decision_tree_classifier, the hyperparameters tuned are criterion, splitter, and max_depth.
    elif current_model_name == 'decision_tree_classifier':
        parameters = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'max_depth': [2, 3, 4, 5, 6, 7]}

    # For random_forest_classifier or extra_trees_classifier, the hyperparameters tuned are n_estimators, criterion, and
    # max_depth
    elif current_model_name == 'random_forest_classifier' or current_model_name == 'extra_trees_classifier':
        parameters = {'n_estimators': [i * 10 for i in range(2, 11, 2)], 'criterion': ['gini', 'entropy'],
                      'max_depth': [2, 3, 4, 5, 6, 7]}

    # For gradient_boosting_classifier, the hyperparameters tuned are loss, n_estimators, criterion, and max_depth.
    else:
        parameters = {'loss': ['deviance', 'exponential'], 'n_estimators': [i * 10 for i in range(2, 11, 2)],
                      'criterion': ['friedman_mse', 'squared_error', 'mse', 'mae'], 'max_depth': [2, 3, 4, 5, 6, 7]}

    return parameters


def split_data_input_target(skeleton_data: pd.DataFrame):
    """Splits skeleton_data into input and target datasets by filtering / selecting certain columns.

        Args:
            skeleton_data: Train / Validation / Test dataset used to split / filter certain columns.

        Returns:
            A tuple containing 2 numpy ndarrays for the input and target datasets.
    """
    skeleton_data_input = skeleton_data.drop(columns=['frame', 'data_name', 'action'])
    skeleton_data_target = skeleton_data['action']
    return np.array(skeleton_data_input), np.array(skeleton_data_target)


def model_training_testing(n_actions: int,
                           n_subjects: int,
                           n_takes: int,
                           skeleton_pose_models: list,
                           modalities: list):
    modality_combinations = list_combinations_generator(modalities)
    train_subject_ids = [i for i in range(1, n_subjects - 1)]
    validation_subject_ids = [n_subjects - 1]
    test_subject_ids = [n_subjects]
    """train_skeleton_input_data, train_skeleton_target_data = data_combiner(n_actions, train_subject_ids, n_takes,
                                                                          modality_combinations[0],
                                                                          skeleton_pose_models[0])
    validation_skeleton_input_data, validation_skeleton_target_data = data_combiner(n_actions, validation_subject_ids,
                                                                                    n_takes, modality_combinations[0],
                                                                                    skeleton_pose_models[0])
    test_skeleton_input_data, test_skeleton_target_data = data_combiner(n_actions, test_subject_ids, n_takes,
                                                                        modality_combinations[0],
                                                                        skeleton_pose_models[0])
    print(train_skeleton_input_data.shape, train_skeleton_target_data.shape, validation_skeleton_input_data.shape,
          validation_skeleton_target_data.shape, test_skeleton_input_data.shape, test_skeleton_target_data.shape)
    model = DecisionTreeClassifier()
    model.fit(train_skeleton_input_data, train_skeleton_target_data)
    train_skeleton_predicted_data = model.predict(train_skeleton_input_data)
    validation_skeleton_predicted_data = model.predict(validation_skeleton_input_data)
    test_skeleton_predicted_data = model.predict(test_skeleton_input_data)
    print(calculate_metrics(train_skeleton_target_data, train_skeleton_predicted_data))
    print(calculate_metrics(validation_skeleton_target_data, validation_skeleton_predicted_data))
    print(calculate_metrics(test_skeleton_target_data, test_skeleton_predicted_data))
    print(validation_skeleton_predicted_data)
    print(validation_skeleton_target_data)"""
    train_skeleton_information = data_combiner(n_actions, train_subject_ids, n_takes, modality_combinations[-1],
                                               skeleton_pose_models[1])
    validation_skeleton_information = data_combiner(n_actions, validation_subject_ids, n_takes,
                                                    modality_combinations[-1], skeleton_pose_models[1])
    test_skeleton_information = data_combiner(n_actions, test_subject_ids, n_takes, modality_combinations[-1],
                                              skeleton_pose_models[1])
    print(train_skeleton_information.head())



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
