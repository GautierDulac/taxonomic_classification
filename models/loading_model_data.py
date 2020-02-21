"""
Loading function to get usable data for models - changing format etc.
"""
# Packages
from os import listdir, makedirs
from os.path import isdir, isfile
from typing import Union, List, Tuple

import numpy as np
import pandas as pd

from hvr_loading.main_hvr_loading import main_hvr_loading
from loading.main_sequence_loading import main_sequence_loading
from utils.utils import folder_paths


# Main function
def main_loading_model_data(force_rewrite: bool = False, test_size: float = 0.2, sequence_origin: str = '',
                            primers_origin: str = 'DairyDB',
                            taxonomy_level: Union[int, List[int]] = None) -> \
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For a given choice of sequence and primer origins, and a taxonomy level, we retrieve the relevant databases,
    merge them and keep only the chosen taxonomy levels.
    We create a dictionnary for ID of classes, and a log_parameter file, within a new folder
    Finally return the usable data for models with X_train, X_test, y_train and y_test, also saved in the data folder.
    X part of the data is not preprocessed (sequences are complete with all hvrs), meaning only 1 + num hvr dimension in X.
    :param test_size: part of data that is left in test df
    :param force_rewrite: if True, data is rewritten if already existed for the given parameters
    :param sequence_origin:
    :param primers_origin:
    :param taxonomy_level:
    :return:
    """
    model_data_folder = folder_paths['model_data']
    alread_existing_data, folder_number = update_loaded_data(sequence_origin, primers_origin, taxonomy_level, test_size)
    folder_name = '{:0>3d}_data\\'.format(folder_number)
    folder_path = model_data_folder + folder_name
    if not force_rewrite and alread_existing_data:
        X_train = pd.read_csv(folder_path + 'x_train.csv')
        X_test = pd.read_csv(folder_path + 'x_test.csv')
        y_train = pd.read_csv(folder_path + 'y_train.csv')
        y_test = pd.read_csv(folder_path + 'y_test.csv')
    else:
        sequence_df = main_sequence_loading(desired_df=sequence_origin)
        taxonomy_levels = sequence_df.columns[2:]
        hvr_df = main_hvr_loading(desired_sequences=sequence_origin, desired_primers=primers_origin)
        hvr_columns = hvr_df.columns[1:]

        merged_df = pd.merge(sequence_df, hvr_df, on='index', how='left')

        X_columns = ['index', 'sequence'] + list(hvr_columns)
        if isinstance(taxonomy_level, int):
            y_columns = taxonomy_levels[taxonomy_level]
        else:
            y_columns = [taxonomy_levels[taxonomy_level[i]] for i in range(len(taxonomy_level))]

        all_columns = X_columns + y_columns

        selected_merged_df = merged_df[all_columns]

        test_indexes = np.random.rand(len(selected_merged_df)) < test_size
        selected_merged_df_test = selected_merged_df[test_indexes]
        selected_merged_df_train = selected_merged_df[~test_indexes]

        X_train = selected_merged_df_train[X_columns]
        X_test = selected_merged_df_test[X_columns]
        y_train = selected_merged_df_test[['index'] + y_columns]
        y_test = selected_merged_df_test[['index'] + y_columns]

        if not isdir(folder_path):
            makedirs(folder_path)

        X_train.to_csv(folder_path + 'x_train.csv', index=False)
        X_test.to_csv(folder_path + 'x_test.csv', index=False)
        y_train.to_csv(folder_path + 'y_train.csv', index=False)
        y_test.to_csv(folder_path + 'y_test.csv', index=False)

    return X_train, X_test, y_train, y_test


# Functions
def get_folder_number() -> int:
    """
    Extract folders already used in model_data folder, to give the new folder_number
    :return: (int) folder number
    """
    current_folders = [f for f in listdir(folder_paths['model_data']) if isdir(f)]
    folder_numbers = [int(f.split('_')[0]) for f in current_folders]

    return max(folder_numbers) + 1


def update_loaded_data(sequence_origin: str, primers_origin: str, taxonomy_level: Union[List[int], int],
                       test_size: float) -> Tuple[bool, int]:
    """
    For a given set of data parameters, update the already loaded parameters, to identify which folder_number to target
    :param: folder_number: the number for the folder where is going to be stored the new loaded data.
    :return: True if data already exist, and related folder_number, False is not existing, with a new folder_number
    where to write the data
    """
    csv_path = folder_paths['model_data'] + 'loaded_data_parameters.csv'
    if isfile(csv_path):
        loaded_data = pd.read_csv(csv_path)
        asked_parameter_df = loaded_data \
            .loc[loaded_data.sequence_origin == sequence_origin] \
            .loc[loaded_data.primers_origin == primers_origin] \
            .loc[loaded_data.taxonomy_level == str(taxonomy_level)] \
            .loc[loaded_data.test_size == test_size]
        if len(asked_parameter_df) > 0:
            return True, int(asked_parameter_df.folder_number[0])
        else:
            folder_number = get_folder_number()
            new_loaded_data = pd.DataFrame(
                [[folder_number, sequence_origin, primers_origin, taxonomy_level, test_size]],
                columns=['folder_number', 'sequence_origin', 'primers_origin',
                         'taxonomy_level', 'test_size'])
            loaded_data = pd.concat(loaded_data, new_loaded_data)
            loaded_data.to_csv(csv_path, index=False)
            return False, folder_number
    else:
        new_loaded_data = pd.DataFrame([[0, sequence_origin, primers_origin, taxonomy_level, test_size]],
                                       columns=['folder_number', 'sequence_origin', 'primers_origin',
                                                'taxonomy_level', 'test_size'])
        new_loaded_data.to_csv(csv_path, index=False)
        return False, 0
