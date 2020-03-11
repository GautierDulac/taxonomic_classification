"""
XGBoost implementation to classify using a given HVR and a given taxonomic rank
"""
# Packages
from os import makedirs
from os.path import isdir

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from models.loading_model_data import main_loading_model_data, get_saved_folder_number
from models.model_statistics import main_stats_model
from processings.sequence_processing import get_ATCG_k_mer_proportion_in_seq
from utils.utils import taxonomy_levels, folder_paths, slash


def xgboost_k_grid_search_cv(k=4, param_grid=None, sequence_origin='DairyDB', primers_origin='DairyDB', taxonomy_level: int = 1,
                             selected_primer: str = 'V4',
                             model_preprocessing='Computing frequency of {}-mer (ATCG) in every sequence',
                             test_size=0.2):
    """
    Apply Random Forest model on a set of sequence preprocessed data.
    :return:
    """
    model_preprocessing = model_preprocessing.format(k)

    X_train, X_test, y_train, y_test = ETL_k_mer(k=k,
                                                 sequence_origin=sequence_origin,
                                                 primers_origin=primers_origin,
                                                 taxonomy_level=taxonomy_level,
                                                 selected_primer=selected_primer)

    XGB = XGBClassifier()

    grid_search = GridSearchCV(estimator=XGB, param_grid=param_grid, cv=3, n_jobs=2, verbose=2)
    grid_search.fit(X_train, y_train)
    XGB_opt = grid_search.best_estimator_
    y_pred = XGB_opt.fit(X_train, y_train).predict(X_test)

    test_size, prop_main_class, accuracy = main_stats_model(y_train=y_train,
                                                            y_test=y_test,
                                                            y_pred=y_pred,
                                                            model_name='XGB_CV_{}'.format(k),
                                                            model_parameters=grid_search.best_params_,
                                                            model_preprocessing=model_preprocessing,
                                                            sequence_origin=sequence_origin,
                                                            primers_origin=primers_origin,
                                                            taxonomy_level=taxonomy_level,
                                                            selected_primer=selected_primer,
                                                            test_size=test_size,
                                                            feature_importances=XGB_opt.feature_importances_,
                                                            k=k,
                                                            save_csv=True,
                                                            xgb_model=XGB_opt,
                                                            save_model=True)

    return test_size, prop_main_class, accuracy


# Main function without CV and Grid search - Now parameters are chosen thanks to previous function
def xgboost_k_default(k=4, sequence_origin='DairyDB', primers_origin='DairyDB', taxonomy_level: int = 1,
                      selected_primer: str = 'V4',
                      model_preprocessing='Computing frequency of {}-mer (ATCG) in every sequence',
                      test_size=0.2):
    """
    Apply Random Forest model on a set of sequence preprocessed data.
    :return:
    """
    model_preprocessing = model_preprocessing.format(k)
    X_train, X_test, y_train, y_test = ETL_k_mer(k=k,
                                                 sequence_origin=sequence_origin,
                                                 primers_origin=primers_origin,
                                                 taxonomy_level=taxonomy_level,
                                                 selected_primer=selected_primer)

    if taxonomy_level >= 5:
        n_estimators = 50
    else:
        n_estimators = 100

    XGB = XGBClassifier(silent=0, eta=0.3, max_depth=4, n_estimators=n_estimators)
    y_pred = XGB.fit(X_train, y_train).predict(X_test)

    test_size, prop_main_class, accuracy = main_stats_model(y_train=y_train,
                                                            y_test=y_test,
                                                            y_pred=y_pred,
                                                            model_name='XGB_{}'.format(k),
                                                            model_parameters=XGB.get_params(),
                                                            model_preprocessing=model_preprocessing,
                                                            sequence_origin=sequence_origin,
                                                            primers_origin=primers_origin,
                                                            taxonomy_level=taxonomy_level,
                                                            selected_primer=selected_primer,
                                                            test_size=test_size,
                                                            k=k,
                                                            feature_importances=XGB.feature_importances_,
                                                            xgb_model=XGB,
                                                            save_model=True,
                                                            save_tree=20)

    del XGB, X_train, X_test, y_train, y_test, y_pred

    return test_size, prop_main_class, accuracy


# Function
def ETL_k_mer(k, sequence_origin='DairyDB', primers_origin='DairyDB', taxonomy_level: int = 1,
              selected_primer: str = 'V4', test_size: float = 0.2):
    """
    Extract Load and Transform data for model usage
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

    folder_name = '{:0>3d}_data{}'.format(folder_number, slash)
    # Data used for NB is the same as the one for RF (when 4-mers are used)
    preprocessed_folder_path = folder_paths['model_data'] + folder_name + 'preprocessed_NB_{}{}'.format(k, slash)

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
            # print('Train set: Processing seq {} / {} - size {}'.format(i, len(X_train), len(seq)), end='\r')
            processed_train_list.append(get_ATCG_k_mer_proportion_in_seq(seq, k))
        processed_X_train = pd.DataFrame(processed_train_list)
        # Processing X_test
        processed_test_list = []
        for i, seq in enumerate(X_test[selected_primer]):
            # print('Test set: Processing seq {} / {} - size {}'.format(i, len(X_test), len(seq)), end='\r')
            processed_test_list.append(get_ATCG_k_mer_proportion_in_seq(seq, k))
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
