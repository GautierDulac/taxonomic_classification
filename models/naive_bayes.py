"""
Naive Bayes implementation to classify using a given HVR and a given taxonomic rank
"""
# Packages
from os import makedirs
from os.path import isdir

import pandas as pd
from sklearn.naive_bayes import GaussianNB

from models.loading_model_data import main_loading_model_data, get_saved_folder_number
from models.model_statistics import main_stats_model
from processings.sequence_processing import get_ATCG_proportion_in_seq
from utils.utils import taxonomy_levels, folder_paths


# Main function
def naive_bayes(sequence_origin='DairyDB', primers_origin='DairyDB', taxonomy_level: int = 1,
                selected_primer: str = 'V4',
                model_preprocessing='Computing frequency of 1-mer (ATCG) in every sequence', test_size=0.2):
    """
    Apply Naive Bayes model on a set of sequence preprocessed data.
    :return:
    """
    X_train, X_test, y_train, y_test = ETL_NB(sequence_origin=sequence_origin,
                                              primers_origin=primers_origin,
                                              taxonomy_level=taxonomy_level,
                                              selected_primer=selected_primer)
    GNB = GaussianNB()
    y_pred = GNB.fit(X_train, y_train).predict(X_test)

    main_stats_model(y_train=y_train,
                     y_test=y_test,
                     y_pred=y_pred,
                     model_name='Naive Bayes - NB(0)',
                     model_parameters=GNB.get_params(),
                     model_preprocessing=model_preprocessing,
                     sequence_origin=sequence_origin,
                     primers_origin=primers_origin,
                     taxonomy_level=taxonomy_level,
                     selected_primer=selected_primer,
                     test_size=test_size)

    return y_pred


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

        # Processing X_train
        processed_train_list = []
        for i, seq in enumerate(X_train[selected_primer]):
            print('Train set: Processing seq {} / {} - size {}'.format(i, len(X_train), len(seq)), end='\r')
            processed_train_list.append(get_ATCG_proportion_in_seq(seq))
        processed_X_train = pd.DataFrame(processed_train_list)
        # Processing X_test
        processed_test_list = []
        for i, seq in enumerate(X_test[selected_primer]):
            print('Test set: Processing seq {} / {} - size {}'.format(i, len(X_test), len(seq)), end='\r')
            processed_test_list.append(get_ATCG_proportion_in_seq(seq))
        processed_X_test = pd.DataFrame(processed_test_list)
        # Keeping only relevant columns for y
        processed_y_train = y_train[[taxonomy_levels[taxonomy_level]]]
        processed_y_test = y_test[[taxonomy_levels[taxonomy_level]]]
        # Not already preprocessed data
        # Saving to relevant folder
        makedirs(preprocessed_folder_path)
        processed_X_train.to_csv(preprocessed_folder_path + 'processed_X_train.csv', index=False)
        processed_X_test.to_csv(preprocessed_folder_path + 'processed_X_test.csv', index=False)
        processed_y_train.to_csv(preprocessed_folder_path + 'processed_y_train.csv', index=False)
        processed_y_test.to_csv(preprocessed_folder_path + 'processed_y_test.csv', index=False)

    return processed_X_train, processed_X_test, processed_y_train, processed_y_test
