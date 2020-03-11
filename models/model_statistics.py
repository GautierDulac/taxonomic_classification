"""
functions to Analyse a model and save results
"""
# Package
import os
import pickle
import random
from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import plot_tree

from processings.sequence_processing import get_all_kmers
from utils.logger import StatLogger
from utils.utils import folder_paths, slash


# Main function
def main_stats_model(y_train: pd.DataFrame, y_test: pd.DataFrame, y_pred: np.ndarray,
                     model_name: str = '',
                     model_parameters: dict = None,
                     model_preprocessing: str = '',
                     sequence_origin: str = '',
                     primers_origin: str = '',
                     taxonomy_level: Union[List[int], int] = '',
                     selected_primer: Union[List[str], str] = '',
                     test_size: float = 0.2,
                     feature_importances: np.ndarray = None,
                     k: int = 4,
                     save_csv: bool = False,
                     xgb_model=None,
                     rf_model=None,
                     save_model=False,
                     save_tree: int = 0):
    """
    Compute relevant statistical analysis on classification model
    :param rf_model:
    :param save_model:
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
    :param feature_importances: For RF models, save in text format the feature_importances
    :param k: Value of k for preprocessing
    :param save_csv: For RF models with Cross Validation and Grid Search, save in csv format the optimal parameters
    :param xgb_model: Model from XGB when saving the first tree
    :param save_tree: Number of random tree to save
    :return: No return, only save analysis in results/models folder
    """
    model_path = folder_paths['model_results'] + model_name + '{}'.format(slash)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    folder_number = get_new_model_folder_number(model_name=model_name)
    analysis_path = model_path + '{:0>5d}_analysis_{}_{}{}'.format(folder_number, selected_primer, taxonomy_level, slash)
    os.makedirs(analysis_path)

    log_path = analysis_path + 'model_results.txt'
    logger = StatLogger(log_path=log_path)

    # Basic information on configuration
    test_size = get_model_info(y_test, model_name, model_parameters, model_preprocessing, sequence_origin,
                               primers_origin, taxonomy_level, selected_primer, test_size, logger)

    # Metrics of model results
    main_class_prop, accuracy = get_metrics_model(y_train, y_test, y_pred, logger, feature_importances, k, save_tree,
                                                  xgb_model,
                                                  analysis_path=analysis_path)

    if save_csv:
        add_optimal_model_params(folder_number, selected_primer, taxonomy_level, accuracy, model_parameters,
                                 model_path=model_path)

    if save_model:
        if xgb_model is not None:
            xgb_model.save_model(analysis_path+'0001.model')
        if rf_model is not None:
            filename = analysis_path+'0001.model'
            pickle.dump(rf_model, open(filename, 'wb'))

    logger.close_file()

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


def get_metrics_model(y_train, y_test, y_pred, logger, feature_importances, k, save_tree, xgb_model, analysis_path=''):
    """

    :param analysis_path:
    :param logger:
    :param y_train:
    :param y_test:
    :param y_pred:
    :param feature_importances:
    :param k:
    :param save_tree:
    :param xgb_model:
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
    logger.log(text='Model Accuracy: {:0.2f}%'.format(accuracy * 100))

    # Saving CSV files
    # y_test.to_csv(analysis_path + 'y_test.csv', index=False)
    pred_df = pd.DataFrame(y_pred, columns=[taxo_column])
    # pred_df.to_csv(analysis_path + 'preds.csv', index=False)

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

    if feature_importances is not None:
        logger.log(subtitle='Feature importances')
        logger.log(text=str(feature_importances))
        feature_plot_path = analysis_path + 'feature_importances_plot.png'
        indices = np.argsort(feature_importances)[-15:]
        features = get_all_kmers(k)
        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.barh(range(len(indices)), feature_importances[indices], color="r", align="center")
        # If you want to define your own labels,
        # change indices to a list of labels on the following line.
        plt.yticks(range(len(indices)), [features[indices_i] for indices_i in indices])
        plt.ylim([-1, len(indices)])
        plt.savefig(feature_plot_path)
        plt.close()

    if xgb_model is not None:
        tree_plot_folder = analysis_path + 'tree_plots{}'.format(slash)
        if not os.path.exists(tree_plot_folder):
            os.makedirs(tree_plot_folder)
        train_classes = np.unique(y_train)
        ids = random.choices(range(len(train_classes)), k=save_tree)
        for index in ids:
            plot_tree(xgb_model, num_trees=index, rankdir='LR')
            fig = plt.gcf()
            fig.set_size_inches(150, 100)
            fig.suptitle("Predicting 'probability' of class {}".format(train_classes[index]), size=200)
            tree_plot_path = tree_plot_folder + 'tree_0_' + train_classes[index] + '.png'

            fig.savefig(tree_plot_path)
            plt.close()

    return main_class_prop, accuracy


# Add a checkpoint for optimal model when gridsearch
def add_optimal_model_params(folder_number, selected_primer, taxonomy_level, accuracy, model_parameters, model_path):
    """

    :param folder_number:
    :param selected_primer:
    :param taxonomy_level:
    :param accuracy:
    :param model_parameters:
    :param model_path:
    :return:
    """
    csv_path = model_path + "optimal_model_parameters.csv"
    columns = ['folder_number', 'selected_primer', 'taxonomy_level', 'accuracy']
    list_of_new_params = [folder_number, str(selected_primer), str(taxonomy_level), accuracy]
    for key, value in model_parameters.items():
        columns.append(key)
        list_of_new_params.append(str(value))

    new_optimal_paramter_df = pd.DataFrame([list_of_new_params], columns=columns)

    if os.path.exists(csv_path):
        saved_optimal_parameters_df = pd.read_csv(csv_path)
        saved_optimal_parameters_df = pd.concat([saved_optimal_parameters_df, new_optimal_paramter_df])
        saved_optimal_parameters_df.to_csv(csv_path, index=False)
    else:
        new_optimal_paramter_df.to_csv(csv_path, index=False)

    return


# Utils
def get_new_model_folder_number(model_name: str = '') -> int:
    """
    Extract folders already used in model_data folder, to give the new folder_number for a given model_name
    :return: (int) folder number
    """
    current_folders = [f for f in os.listdir(folder_paths['model_results'] + model_name + '{}'.format(slash))]
    folder_numbers = [int(f.split('_')[0]) for f in current_folders if f.split('_')[0] != 'optimal']
    if len(folder_numbers) == 0:
        return 0
    else:
        return max(folder_numbers) + 1
