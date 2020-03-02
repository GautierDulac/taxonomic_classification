"""
Random Forest implementation to classify using a given HVR and a given taxonomic rank
"""
# Packages
from os import makedirs
from os.path import isdir

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from models.loading_model_data import main_loading_model_data, get_saved_folder_number
from models.model_statistics import main_stats_model
from processings.sequence_processing import get_ATCG_k_mer_proportion_in_seq
from utils.utils import taxonomy_levels, folder_paths


def random_forest_k_grid_search_cv(k=4, sequence_origin='DairyDB', primers_origin='DairyDB', taxonomy_level: int = 1,
                                   selected_primer: str = 'V4',
                                   model_preprocessing='Computing frequency of {}-mer (ATCG) in every sequence',
                                   test_size=0.2):
    """
    Apply Random Forest model on a set of sequence preprocessed data.
    :return:
    """
    model_preprocessing = model_preprocessing.format(k)

    # Number of trees in random forest
    # n_estimators = [200]  # Checked as often the best option
    # Number of features to consider at every split
    # max_features = ['auto']  # Checked as best option
    # Maximum number of levels in tree
    # max_depth = [None]  # Checked as best option -> Due to memory errors, limiting at 30
    # Minimum number of samples required to split a node
    # min_samples_split = [2]  # Instead of 2, 5, 10 because of unbalanced classes
    # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1]  # Instead of 1, 2, 4 because of unbalanced classes
    # Method of selecting samples for training each tree
    # bootstrap = [False]  # Checked as best option
    # Create the random grid
    param_grid = {}

    X_train, X_test, y_train, y_test = ETL_RF_k_mer(k=k,
                                                    sequence_origin=sequence_origin,
                                                    primers_origin=primers_origin,
                                                    taxonomy_level=taxonomy_level,
                                                    selected_primer=selected_primer)
    if taxonomy_level == 0:
        bootstrap = True
    else:
        bootstrap = False
    RF = RandomForestClassifier(bootstrap=bootstrap, min_samples_leaf=1, min_samples_split=2, max_features='auto',
                                n_estimators=200, max_depth=30)
    grid_search = GridSearchCV(estimator=RF, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    RF_opt = grid_search.best_estimator_
    y_pred = RF_opt.fit(X_train, y_train).predict(X_test)

    test_size, prop_main_class, accuracy = main_stats_model(y_train=y_train,
                                                            y_test=y_test,
                                                            y_pred=y_pred,
                                                            model_name='RF_CV_{}'.format(k),
                                                            model_parameters=grid_search.best_params_,
                                                            model_preprocessing=model_preprocessing,
                                                            sequence_origin=sequence_origin,
                                                            primers_origin=primers_origin,
                                                            taxonomy_level=taxonomy_level,
                                                            selected_primer=selected_primer,
                                                            test_size=test_size,
                                                            feature_importances=RF_opt.feature_importances_,
                                                            k=k,
                                                            save_csv=True)

    return RF_opt, test_size, prop_main_class, accuracy


# Main function without CV and Grid search - Now parameters are chosen thanks to previous function
def random_forest_k_default(k=4, sequence_origin='DairyDB', primers_origin='DairyDB', taxonomy_level: int = 1,
                            selected_primer: str = 'V4',
                            model_preprocessing='Computing frequency of {}-mer (ATCG) in every sequence',
                            test_size=0.2):
    """
    Apply Random Forest model on a set of sequence preprocessed data.
    :return:
    """
    model_preprocessing = model_preprocessing.format(k)
    X_train, X_test, y_train, y_test = ETL_RF_k_mer(k=k,
                                                    sequence_origin=sequence_origin,
                                                    primers_origin=primers_origin,
                                                    taxonomy_level=taxonomy_level,
                                                    selected_primer=selected_primer)
        
    if taxonomy_level >= 5:
        max_depth = 10
    elif taxonomy_level >= 3 and selected_primer == 'sequence' and sequence_origin == '':
        max_depth = 20
    else:
        max_depth = 30
    RF = RandomForestClassifier(bootstrap=False, min_samples_leaf=1, min_samples_split=2, max_features=min(50, 4**k),
                                n_estimators=200, max_depth=max_depth, n_jobs=-1)  # 30 for max_depth is not backed-up
    y_pred = RF.fit(X_train, y_train).predict(X_test)

    test_size, prop_main_class, accuracy = main_stats_model(y_train=y_train,
                                                            y_test=y_test,
                                                            y_pred=y_pred,
                                                            model_name='RF_{}'.format(k),
                                                            model_parameters=RF.get_params(),
                                                            model_preprocessing=model_preprocessing,
                                                            sequence_origin=sequence_origin,
                                                            primers_origin=primers_origin,
                                                            taxonomy_level=taxonomy_level,
                                                            selected_primer=selected_primer,
                                                            test_size=test_size,
                                                            k=k,
                                                            feature_importances=RF.feature_importances_)

    return test_size, prop_main_class, accuracy


# Function
def ETL_RF_k_mer(k, sequence_origin='DairyDB', primers_origin='DairyDB', taxonomy_level: int = 1,
                 selected_primer: str = 'V4', test_size: float = 0.2):
    """
    Extract Load and Transform data for RF usage
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
    # Data used for NB is the same as the one for RF (when 4-mers are used)
    preprocessed_folder_path = folder_paths['model_data'] + folder_name + 'preprocessed_NB_{}\\'.format(k)

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
