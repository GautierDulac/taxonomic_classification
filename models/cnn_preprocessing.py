"""
CNN preprocessing implementation to classify using a given HVR and a given taxonomic rank
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Constants
from models.loading_model_data import main_loading_model_data

bs = 64  # batchsize
vectorized_dict = {
    'A': [1, 0, 0, 0],
    'T': [0, 1, 0, 0],
    'C': [0, 0, 1, 0],
    'G': [0, 0, 0, 1],
    'U': [0, 1, 0, 0],
    'R': [1, 0, 0, 1],
    'Y': [0, 1 / 2, 1 / 2, 0],
    'K': [0, 1 / 2, 0, 1 / 2],
    'M': [1 / 2, 0, 1 / 2, 0],
    'S': [0, 0, 1 / 2, 1 / 2],
    'W': [1 / 2, 1 / 2, 0, 0],
    'B': [0, 1 / 3, 1 / 3, 1 / 3],
    'D': [1 / 3, 1 / 3, 0, 1 / 3],
    'H': [1 / 3, 1 / 3, 1 / 3, 0],
    'V': [1 / 3, 0, 1 / 3, 1 / 3],
    'N': [1 / 4, 1 / 4, 1 / 4, 1 / 4],
    'X': [1 / 4, 1 / 4, 1 / 4, 1 / 4],
}


def main_preprocessing_cnn(sequence_origin='DairyDB', primers_origin='DairyDB', selected_primer='V4', taxonomy_level=3,
                           max_size=300, k_mer: int = 1):
    """

    :param sequence_origin: GreenGenes or DairyDB
    :param primers_origin: DairyDB or Chaudhary
    :param taxonomy_level: list of int within 0 - 6
    :param selected_primer: Chosen primer on which we want to classify, if None, all columns are returned
    :param max_size: Size of vectors after processing (must be closed to the max size of sequence)
    :param k_mer: What is the size of the vectorized k_mers - Only k=1 implemented for now # TODO IMPLEMENT k>1
    :return: two DataLoaders for train and test, and the mapping dicts for classes
    """
    X_train, X_test, y_train, y_test = main_loading_model_data(sequence_origin=sequence_origin,
                                                               primers_origin=primers_origin,
                                                               selected_primer=selected_primer,
                                                               taxonomy_level=taxonomy_level)
    y_train_col = y_train.iloc[:, 1]
    y_test_col = y_test.iloc[:, 1]

    all_classes = pd.concat([y_train_col, y_test_col]).unique()
    dict_class_to_id = {}
    dict_id_to_class = {}
    for index, class_index in enumerate(all_classes):
        dict_class_to_id[class_index] = index
        dict_id_to_class[index] = class_index

    new_y_train = np.array([dict_class_to_id[class_index] for class_index in y_train_col])
    new_y_test = np.array([dict_class_to_id[class_index] for class_index in y_test_col])

    new_X_train = np.array([get_homogenous_vector(X_train.iloc[i, 1], max_size).transpose() for i in range(len(X_train))])
    new_X_test = np.array([get_homogenous_vector(X_test.iloc[i, 1], max_size).transpose() for i in range(len(X_test))])

    train_dataset = [[torch.from_numpy(seq.astype(np.float32)),
                      torch.from_numpy(np.array(new_y_train[index]).astype(np.int64))]
                     for index, seq in enumerate(new_X_train)]

    test_dataset = [[torch.from_numpy(seq.astype(np.float32)),
                     torch.from_numpy(np.array(new_y_test[index]).astype(np.int64))]
                    for index, seq in enumerate(new_X_test)]

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=True)

    return train_loader, test_loader, dict_class_to_id, dict_id_to_class


def get_homogenous_vector(seq: str, max_size: int) -> np.ndarray:
    """
    For a given sequence, apply the transformation to (n,4) list and change n to be max_size
    :param seq:
    :param max_size:
    :return:
    """
    list_of_vectorized_seq = transform_sequence_to_n_times_4_vector(seq)
    if len(list_of_vectorized_seq) > max_size:
        return np.array(list_of_vectorized_seq[:max_size])
    else:
        return np.array(list_of_vectorized_seq + [[0, 0, 0, 0]] * (max_size - len(list_of_vectorized_seq)))


def transform_sequence_to_n_times_4_vector(seq: str) -> list:
    """
    From a given ATCG(+) sequence, return a n times 4 vector, the image of the sequence
    :param seq: (str)
    :return: (n, 4) list
    """
    final_list_of_vectorized_based = []
    for letter in seq:
        if letter not in vectorized_dict.keys():
            continue
        else:
            final_list_of_vectorized_based.append(vectorized_dict[letter])
    return final_list_of_vectorized_based
