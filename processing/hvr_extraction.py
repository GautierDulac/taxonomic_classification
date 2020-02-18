"""
Loading functions to retrieve HVR sequences from GreenGene loaded data
"""
# Package
from re import finditer
from typing import List

import pandas as pd

from loading.loading_primers import get_dict_of_primers

# Constant
dict_of_primers = get_dict_of_primers()


# Main Function


# Functions

def get_all_occurences(sequence_df: pd.DataFrame = None) -> pd.DataFrame:
    """

    :param sequence_df: DataFrame with all sequences
    :return:
    """
    columns_primers = [['index']] + [[hvr + '_forward', hvr + '_reverse'] for hvr in dict_of_primers.keys()]
    columns_primers_flat = [item for sublist in columns_primers for item in sublist]
    list_of_all_occurences = []
    for i in sequence_df.index:
        list_of_occurences = [i] + get_occurences_hvr_by_sequence(sequence_df.sequence.iloc[i])
        list_of_all_occurences.append(list_of_occurences)
    return pd.DataFrame(list_of_all_occurences, columns=columns_primers_flat)


def get_occurences_hvr_by_sequence(seq: str = '') -> List[List[int]]:
    """

    :param seq: sequence in which to look for primers
    :return: a list corresponding to 28 sublists of first occurence of start primer and end primer for the 14 HVR
    """

    list_of_occurences = []
    for hvr in dict_of_primers.keys():
        occurences_of_start = [m.start() + len(dict_of_primers[hvr]['forward']) for m in finditer(dict_of_primers[hvr]['forward'], seq)]
        occurences_of_end = [m.start() for m in finditer(dict_of_primers[hvr]['reverse'][::-1], seq)]
        list_of_occurences.append(occurences_of_start)
        list_of_occurences.append(occurences_of_end)
    return list_of_occurences
