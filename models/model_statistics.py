"""
functions to Analyse a model and save results
"""
# Package
import os
from typing import Union, List

import numpy as np
import pandas as pd

from utils.logger import StatLogger
from utils.utils import folder_paths


# Main function
def main_stats_model(y_train: pd.DataFrame, y_test: pd.DataFrame, y_pred: np.ndarray,
                     model_name: str = '',
                     model_parameters: dict = None,
                     model_preprocessing: str = '',
                     sequence_origin: str = '',
                     primers_origin: str = '',
                     taxonomy_level: Union[List[int], int] = '',
                     selected_primer: Union[List[str], str] = '',
                     test_size: float = 0.2):
    """
    Compute relevant statistical analysis on classification model
    :param y_train: trained class
    :param y_test: real class for comparison
    :param y_pred: result of model .predict
    :param model_name: Name of the model (NB etc.)
    :param model_parameters: model.get_params()
    :param model_preprocessing: Manual string explanation of the preprocessing applied before the model
    :param sequence_origin:
    :param primers_origin:
    :param taxonomy_level:
    :param selected_primer:
    :param test_size:
    :return: No return, only save analysis in results/models folder
    """
    model_path = folder_paths['model_results'] + model_name + '\\'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    folder_number = get_new_model_folder_number(model_name=model_name)
    analysis_path = model_path + '{:0>5d}_analysis_{}_{}\\'.format(folder_number, selected_primer, taxonomy_level)
    os.makedirs(analysis_path)

    log_path = analysis_path + 'model_results.txt'
    logger = StatLogger(log_path=log_path)

    # Basic information on configuration
    test_size = get_model_info(y_test, model_name, model_parameters, model_preprocessing, sequence_origin,
                               primers_origin, taxonomy_level, selected_primer, test_size, logger)

    # Metrics of model results
    main_class_prop, accuracy = get_metrics_model(y_train, y_test, y_pred, logger, analysis_path=analysis_path)

    return test_size, main_class_prop, accuracy


# Stats functions
def get_model_info(y_test, model_name, model_parameters, model_preprocessing, sequence_origin, primers_origin,
                   taxonomy_level, selected_primer, test_size, logger) -> int:
    """
    Save basics stats on the model parameters and info
    :param logger:
    :param y_test:
    :param model_name:
    :param model_parameters:
    :param model_preprocessing:
    :param sequence_origin:
    :param primers_origin:
    :param taxonomy_level:
    :param selected_primer:
    :param test_size:
    :return:
    """

    # Global information on the model
    logger.log(title='Parameter information for {}'.format(model_name))
    # Data Origins
    logger.log(subtitle='Data Origins')
    logger.log(text='Sequence origin: {}'.format(sequence_origin))
    logger.log(text='Primers origin: {}'.format(primers_origin))
    # Chosen levels for classification
    logger.log(subtitle='Chosen HyperVariable Region and Taxonomy Rank')
    logger.log(text='HyperVariable Region: {}'.format(str(selected_primer)))
    logger.log(text='Taxonomy Rank: {}'.format(str(taxonomy_level)))
    # Applied Preprocessing
    logger.log(subtitle='Preprocessing')
    logger.log(text='Preprocessing description: ' + model_preprocessing)
    # Model parameters
    logger.log(subtitle='Model parameters')
    logger.log(text='Parameter dict: {}'.format(str(model_parameters)))
    logger.log(text='Size of test set: {}'.format(len(y_test)))
    logger.log(text='Part of test size compared to total: {}'.format(test_size))

    return len(y_test)


def get_metrics_model(y_train, y_test, y_pred, logger, analysis_path=''):
    """

    :param analysis_path:
    :param logger:
    :param y_train:
    :param y_test:
    :param y_pred:
    :return:
    """
    # Results on the model
    logger.log(title='Results')
    taxo_column = y_test.columns[0]
    bad_predictions = sum([y_test.iloc[i][taxo_column] != y_pred[i] for i in range(len(y_test))])
    logger.log(text='Number of seen classes in train: {}'.format(len(y_train[taxo_column].value_counts())))
    logger.log(text='Number of predicted classes in pred: {}'.format(len(np.unique(y_pred))))
    logger.log(text='Number of classes waited in test: {}'.format(len(y_test[taxo_column].value_counts())))
    logger.log(text='Number of wrong prediction: {} over {}'.format(bad_predictions, len(y_test)))
    accuracy = (1 - (bad_predictions / len(y_test)))
    logger.log(text='Model Accuracy: {:0.2f}%'.format(accuracy*100))

    # Saving CSV files
    y_test.to_csv(analysis_path + 'y_test.csv', index=False)
    pred_df = pd.DataFrame(y_pred, columns=[taxo_column])
    pred_df.to_csv(analysis_path + 'preds.csv', index=False)

    # Details on classes
    logger.log(subtitle='Main classes in train set')
    value_counts = y_train[taxo_column].value_counts()
    local_size = len(y_train)
    main_class_prop = value_counts[0] / local_size
    for rank in range(min(len(value_counts), 5)):
        logger.log(text='Train - Rank {} - {:0.2f}%: {} with {} occurences'.format(rank + 1,
                                                                                   value_counts[
                                                                                       rank] / local_size * 100,
                                                                                   value_counts.index[rank],
                                                                                   value_counts[rank]))
    logger.log(subtitle='Main classes in test set')
    value_counts = y_test[taxo_column].value_counts()
    local_size = len(y_test)
    for rank in range(min(len(value_counts), 5)):
        logger.log(text='Test - Rank {} - {:0.2f}%: {} with {} occurences'.format(rank + 1,
                                                                                  value_counts[rank] / local_size * 100,
                                                                                  value_counts.index[rank],
                                                                                  value_counts[rank]))
    logger.log(subtitle='Main classes in the predicted classes')
    value_counts = pred_df[taxo_column].value_counts()
    local_size = len(pred_df)
    for rank in range(min(len(value_counts), 5)):
        logger.log(text='Predictions - Rank {} - {:0.2f}%: {} with {} occurences'.format(rank + 1,
                                                                                         value_counts[
                                                                                             rank] / local_size * 100,
                                                                                         value_counts.index[rank],
                                                                                         value_counts[rank]))

    return main_class_prop, accuracy


# Utils
def get_new_model_folder_number(model_name: str = '') -> int:
    """
    Extract folders already used in model_data folder, to give the new folder_number for a given model_name
    :return: (int) folder number
    """
    current_folders = [f for f in os.listdir(folder_paths['model_results'] + model_name + '\\')]
    folder_numbers = [int(f.split('_')[0]) for f in current_folders]
    if len(folder_numbers) == 0:
        return 0
    else:
        return max(folder_numbers) + 1
