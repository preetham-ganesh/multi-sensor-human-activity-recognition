# authors_name = 'Preetham Ganesh'
# project_title = 'Multi Sensor-based Human Activity Recognition using OpenCV and Sensor Fusion'
# email = 'preetham.ganesh2015@gmail.com'


import pandas as pd
from model_training_testing import list_combinations_generator


def per_combination_find_best_model(modalities_combination_name: str,
                                    skeleton_pose_model: str,
                                    complete_modality_combination_metrics: pd.DataFrame,
                                    metric_features: list):
    """Finds best model for each modality combination and skeleton pose model. Appends the identified best model to the
    complete best model data.

        Args:
            modalities_combination_name: Name for the current modality combination.
            skeleton_pose_model: Name of the skeleton pose model used for extracting skeleton points from RGB videos.
            complete_modality_combination_metrics: Pandas dataframe which contains the best model for each modality
                                                   combination.
            metric_features: List of metrics used for identifying the best model for each modality combination.

        Returns:
            An updated version of the pandas dataframe which contains the best model for each modality combination.
    """
    # Reads test metrics for the current modality combination and skeleton pose model.
    current_modality_combination_metrics = pd.read_csv('../results/combination_results/{}_{}/test_metrics.csv'.format(
        modalities_combination_name, skeleton_pose_model))

    # Iterates across the list metrics and appends best model and parameters index to the list for each metric.
    best_model_parameter_metrics_index = []
    for i in range(len(metric_features)):
        current_metric = list(current_modality_combination_metrics[metric_features[i]])
        current_metric_maximum_value = max(current_metric)
        current_metric_maximum_value_index = current_metric.index(current_metric_maximum_value)
        best_model_parameter_metrics_index.append(current_metric_maximum_value_index)

    # Identifies the best model based on the maximum count of occurrences in the best_model_parameter_metrics_index.
    # Appends the identified best model to the complete best model information.
    best_model_index = max(best_model_parameter_metrics_index, key=best_model_parameter_metrics_index.count)
    current_combination_best_model = dict(current_modality_combination_metrics.iloc[best_model_index])
    current_combination_best_model['modality_combination'] = modalities_combination_name
    current_combination_best_model['skeleton_pose_model'] = skeleton_pose_model
    complete_modality_combination_metrics = complete_modality_combination_metrics.append(current_combination_best_model,
                                                                                         ignore_index=True)

    return complete_modality_combination_metrics


def find_best_model_all_combinations(modalities_combinations: list,
                                     skeleton_pose_models: list):
    """Finds best model for all the modality combinations and skeleton pose models.

        Args:
            modalities_combinations: List which contains the combinations for all the modalities in the dataset.
            skeleton_pose_models: List of all the skeleton pose models which was used to extract the skeleton pose
                                  information from the RGB videos.

        Returns:
            None.
    """
    # Creating empty dataframe to store best ML model for each modality combination and skeleton pose model.
    metric_features = ['accuracy_score', 'balanced_accuracy_score', 'precision_score', 'recall_score', 'f1_score']
    complete_modality_metrics = pd.DataFrame(columns=['modality_combination', 'skeleton_pose_model',
                                                      'model_names', 'parameters'] + metric_features)

    # Iterates across modalities combinations and skeleton pose models to find the best ML model and parameters.
    for i in range(len(modalities_combinations)):
        modalities_combination_name = '_'.join(modalities_combinations[i])
        for j in range(len(skeleton_pose_models)):
            complete_modality_metrics = per_combination_find_best_model(modalities_combination_name,
                                                                        skeleton_pose_models[j],
                                                                        complete_modality_metrics, metric_features)

    # Exports dataframe which contains best ML model details for each modality combination and skeleton pose model.
    complete_modality_metrics.to_csv('../results/best_model_per_modality_combination.csv', index=False)


def main():
    skeleton_pose_models = ['coco', 'mpi']
    modalities = ['rgb', 'depth', 'inertial']
    modalities_combinations = list_combinations_generator(modalities)
    find_best_model_all_combinations(modalities_combinations, skeleton_pose_models)


if __name__ == '__main__':
    main()
