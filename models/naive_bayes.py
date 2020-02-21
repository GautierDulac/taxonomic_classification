"""
Naive Bayes implementation to classify using a given HVR and a given taxonomic rank
"""
# Packages
from os import makedirs
from os.path import isdir

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

from models.loading_model_data import main_loading_model_data, get_saved_folder_number
from processings.sequence_processing import get_ATCG_proportion_in_seq
from utils.utils import taxonomy_levels, folder_paths


# Main function
def naive_bayes():
    """
    Apply Naive Bayes model on a set of sequence preprocessed data.
    :return:
    """
    X_train, X_test, y_train, y_test = ETL_NB()
    GNB = GaussianNB()
    y_pred = GNB.fit(X_train, y_train).predict(X_test)
    num_test = X_test.shape[0]
    bad_predictions = sum([y_test[i] != y_pred[i] for i in range(y_test)])
    print("Number of mislabeled points out of a total %d points : %d" % (num_test, bad_predictions))


# Function
def ETL_NB(sequence_origin='DairyDB', primers_origin='DairyDB', taxonomy_level: int = 1, selected_primer: str = 'V4',
           test_size: float = 0.2):
    """
    Extract Load and Transform data for NB usage
    :return:
    """
    X_train, X_test, y_train, y_test = main_loading_model_data(sequence_origin=sequence_origin,
                                                               primers_origin=primers_origin,
                                                               taxonomy_level=taxonomy_level,
                                                               selected_primer=selected_primer,
                                                               test_size=test_size)
    folder_number = get_saved_folder_number(sequence_origin=sequence_origin,
                                            primers_origin=primers_origin,
                                            taxonomy_level=taxonomy_level,
                                            selected_primer=selected_primer,
                                            test_size=test_size)

    folder_name = '{:0>3d}_data\\'.format(folder_number)
    preprocessed_folder_path = folder_paths['model_data'] + folder_name + 'preprocessed_NB\\'

    if isdir(preprocessed_folder_path):
        # Already preprocessed data
        processed_X_train = pd.read_csv(preprocessed_folder_path + 'processed_X_train.csv')
        processed_X_test = pd.read_csv(preprocessed_folder_path + 'processed_X_test.csv')
        processed_y_train = pd.read_csv(preprocessed_folder_path + 'processed_y_train.csv')
        processed_y_test = pd.read_csv(preprocessed_folder_path + 'processed_y_test.csv')

    else:
        # Not already preprocessed data
        makedirs(preprocessed_folder_path)
        # Processing X_train
        processed_train_list = []
        for seq in X_train[selected_primer]:
            processed_train_list.append(get_ATCG_proportion_in_seq(seq))
        processed_X_train = pd.DataFrame(processed_train_list)
        # Processing X_test
        processed_test_list = []
        for seq in X_test[selected_primer]:
            processed_test_list.append(get_ATCG_proportion_in_seq(seq))
        processed_X_test = pd.DataFrame(processed_test_list)
        # Keeping only relevant columns for y
        processed_y_train = y_train[taxonomy_levels[taxonomy_level]]
        processed_y_test = y_test[taxonomy_levels[taxonomy_level]]
        # Saving to relevant folder
        processed_X_train.to_csv(preprocessed_folder_path + 'processed_X_train.csv', index=False)
        processed_X_test.to_csv(preprocessed_folder_path + 'processed_X_test.csv', index=False)
        processed_y_train.to_csv(preprocessed_folder_path + 'processed_y_train.csv', index=False)
        processed_y_test.to_csv(preprocessed_folder_path + 'processed_y_test.csv', index=False)

    return processed_X_train, processed_X_test, processed_y_train, processed_y_test