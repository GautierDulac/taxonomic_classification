"""
functions to Analyse a model and save results
"""
# Package
import os
from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from models.cnn_acm import create_activation_map
from models.model_statistics import get_new_model_folder_number
from utils.logger import StatLogger
from utils.utils import folder_paths, taxonomy_levels, slash


# Main function
def main_cnn_stats_model(y_train, y_test_torch, y_pred_torch, dict_id_to_class, loss_train, loss_test, acc_train,
                         acc_test,
                         make_plot: bool = False,
                         save_model: bool = False,
                         model_name: str = '',
                         model_class=None,
                         create_acm=False,
                         acm_parameters=None,
                         parameter_config=None,
                         model_preprocessing: str = '',
                         sequence_origin: str = '',
                         primers_origin: str = '',
                         taxonomy_level: Union[List[int], int] = '',
                         selected_primer: Union[List[str], str] = '',
                         test_size: float = 0.2):
    """
    Compute relevant statistical analysis on classification model
    :param acm_parameters:
    :param create_acm:
    :param parameter_config:
    :param save_model:
    :param y_pred_torch: real class for comparison in tensor format
    :param y_test_torch: result of model in tensor format
    :param make_plot:
    :param acc_test:
    :param acc_train:
    :param loss_test:
    :param loss_train:
    :param dict_id_to_class:
    :param model_class:
    :param y_train: trained class
    :param model_name: Name of the model (NB etc.)
    :param model_preprocessing: Manual string explanation of the preprocessing applied before the model
    :param sequence_origin:
    :param primers_origin:
    :param taxonomy_level:
    :param selected_primer:
    :param test_size:
    :return: No return, only save analysis in results/models folder
    """
    model_path = folder_paths['model_results'] + model_name + '{}'.format(slash)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    folder_number = get_new_model_folder_number(model_name=model_name)
    analysis_path = model_path + '{:0>5d}_analysis_{}_{}{}'.format(folder_number, selected_primer, taxonomy_level, slash)
    os.makedirs(analysis_path)

    if save_model:
        torch.save(model_class.state_dict(), analysis_path + 'model.pt')

    log_path = analysis_path + 'model_results.txt'
    logger = StatLogger(log_path=log_path)

    y_train = list(y_train.iloc[:, 1])
    y_test = [dict_id_to_class[index.item()] for index in y_test_torch]
    y_pred = [dict_id_to_class[index.item()] for index in y_pred_torch]

    # Basic information on configuration
    _ = get_cnn_model_info(y_test, model_name, model_class, parameter_config, model_preprocessing, sequence_origin,
                           primers_origin, taxonomy_level, selected_primer, test_size, logger)

    # Metrics of model results
    main_class_prop, accuracy = get_cnn_metrics_model(y_train, y_test, y_pred,
                                                      loss_train, loss_test, acc_train, acc_test, logger)

    if make_plot:
        add_cnn_plot(folder_number, selected_primer, taxonomy_level, accuracy, loss_train, loss_test, acc_train,
                     acc_test, analysis_path=analysis_path)

    if create_acm:
        X_test = acm_parameters[0]
        y_test = acm_parameters[1]
        n = acm_parameters[2]
        create_activation_map(X_test, y_test, dict_id_to_class, parameter_config, n=n, analysis_path=analysis_path)

    logger.close_file()

    return  # test_size, main_class_prop, accuracy


# Stats functions
def get_cnn_model_info(y_test, model_name, model_class, parameter_config, model_preprocessing, sequence_origin,
                       primers_origin, taxonomy_level, selected_primer, test_size, logger) -> int:
    """
    Save basics stats on the model parameters and info
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
    logger.log(text='Size of test set: {}'.format(len(y_test)))
    logger.log(text='Part of test size compared to total: {}'.format(test_size))
    logger.log(text='Parameter config raw: {}'.format(parameter_config))
    for attribute, value in parameter_config.items():
        logger.log(text='Parameter config: {} = {}'.format(attribute, value))
    for attribute, value in model_class.__dict__.items():
        if attribute[0] != '_':
            logger.log(text='Parameter dict: {} = {}'.format(attribute, value))

    return len(y_test)


def get_cnn_metrics_model(y_train, y_test, y_pred, loss_train, loss_test,
                          acc_train, acc_test, logger):
    """

    :return:
    """
    # Results on the model
    logger.log(title='Results')
    bad_predictions = sum([y_test[i] != y_pred[i] for i in range(len(y_test))])
    logger.log(text='Number of seen classes in train: {}'.format(len(np.unique(np.array(y_train)))))
    logger.log(text='Number of predicted classes in pred: {}'.format(len(np.unique(np.array(y_pred)))))
    logger.log(text='Number of classes waited in test: {}'.format(len(np.unique(np.array(y_test)))))
    logger.log(text='Number of wrong prediction: {} over {}'.format(bad_predictions, len(y_test)))
    accuracy = (1 - (bad_predictions / len(y_test)))
    logger.log(text='Model Accuracy: {:0.2f}%'.format(accuracy * 100))

    # Details on classes
    logger.log(subtitle='Main classes in train set')
    unique_unsorted, counts_unsorted = np.unique(np.array(y_train), return_counts=True)
    value_counts = {}
    for index, classe in enumerate(unique_unsorted):
        value_counts[classe] = counts_unsorted[index]
    unique = list({k: v for k, v in sorted(value_counts.items(), key=lambda item: item[1], reverse=True)}.keys())
    counts = list({k: v for k, v in sorted(value_counts.items(), key=lambda item: item[1], reverse=True)}.values())
    local_size = len(y_train)
    main_class_prop = counts[0] / local_size
    for rank in range(min(len(unique), 5)):
        logger.log(text='Train - Rank {} - {:0.2f}%: {} with {} occurences'.format(rank + 1,
                                                                                   counts[
                                                                                       rank] / local_size * 100,
                                                                                   unique[rank],
                                                                                   counts[rank]))
    logger.log(subtitle='Main classes in test set')
    unique_unsorted, counts_unsorted = np.unique(np.array(y_test), return_counts=True)
    value_counts = {}
    for index, classe in enumerate(unique_unsorted):
        value_counts[classe] = counts_unsorted[index]
    unique = list({k: v for k, v in sorted(value_counts.items(), key=lambda item: item[1], reverse=True)}.keys())
    counts = list({k: v for k, v in sorted(value_counts.items(), key=lambda item: item[1], reverse=True)}.values())
    local_size = len(y_test)
    for rank in range(min(len(unique), 5)):
        logger.log(text='Test - Rank {} - {:0.2f}%: {} with {} occurences'.format(rank + 1,
                                                                                  counts[
                                                                                      rank] / local_size * 100,
                                                                                  unique[rank],
                                                                                  counts[rank]))
    logger.log(subtitle='Main classes in predictions')
    unique_unsorted, counts_unsorted = np.unique(np.array(y_pred), return_counts=True)
    value_counts = {}
    for index, classe in enumerate(unique_unsorted):
        value_counts[classe] = counts_unsorted[index]
    unique = list({k: v for k, v in sorted(value_counts.items(), key=lambda item: item[1], reverse=True)}.keys())
    counts = list({k: v for k, v in sorted(value_counts.items(), key=lambda item: item[1], reverse=True)}.values())
    local_size = len(y_pred)
    for rank in range(min(len(unique), 5)):
        logger.log(text='Predictions - Rank {} - {:0.2f}%: {} with {} occurences'.format(rank + 1,
                                                                                         counts[
                                                                                             rank] / local_size * 100,
                                                                                         unique[rank],
                                                                                         counts[rank]))

    logger.log(title='Loss and accuracies over epochs')
    logger.log(text='Number of epochs: {}'.format(len(loss_train)))
    logger.log(text='Train loss: {}'.format(loss_train))
    logger.log(text='Test loss: {}'.format(loss_test))
    logger.log(text='Train accuracy: {}'.format(acc_train))
    logger.log(text='Test accuracy: {}'.format(acc_test))

    return main_class_prop, accuracy


def add_cnn_plot(folder_number, selected_primer, taxonomy_level, accuracy, loss_train, loss_test, acc_train, acc_test,
                 analysis_path=''):
    """
    Plot evolution over epochs.
    :return:
    """
    feature_plot_path = analysis_path + 'epochs_evolution.png'
    sns.set()
    n_epochs = len(loss_train)
    X = range(1, n_epochs + 1)

    fig, ax1 = plt.subplots(figsize=(12, 12))

    color = 'tab:orange'
    ax1.set_xlabel('Epochs', fontsize=20)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.set_ylabel('Loss', color=color, fontsize=20)
    ln1 = ax1.plot(X, loss_train, color=color, linewidth=2.5, linestyle='--', label='Train Loss')
    ln2 = ax1.plot(X, loss_test, color=color, linewidth=2.5, label='Test Loss')
    ax1.tick_params(axis='y', labelcolor=color, labelsize=16)
    ax1.set_ylim(0, 0.1)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color, fontsize=20)  # we already handled the x-label with ax1
    ln3 = ax2.plot(X, acc_train, color=color, linewidth=2.5, linestyle='--', label='Train Accuracy')
    ln4 = ax2.plot(X, acc_test, color=color, linewidth=2.5, label='Test Accuracy')
    ax2.tick_params(axis='y', labelcolor=color, labelsize=16)
    ax2.set_ylim(0, 1)

    ax1.set_title(
        "Experiments number: {} - Final accuracy: {:.2f}\nEvolution over {} epochs to classify {} with {}".format(
            folder_number, accuracy * 100, n_epochs, taxonomy_levels[taxonomy_level], selected_primer), fontsize=22)

    lns = ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='center right', fontsize=16)

    fig.tight_layout()

    plt.savefig(feature_plot_path)
    plt.close()

    return
