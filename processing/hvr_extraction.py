"""
Loading functions to retrieve HVR sequences from loaded data
"""
# Package
from typing import List, Dict, Union, Tuple

import pandas as pd

from loading.loading_primers import get_dict_of_primers
from utils.utils import get_list_of_related_primers, get_searchable_reverse_primer

# Constant
dict_of_primers = get_dict_of_primers(article='DairyDB')


# Main Function
def get_all_hvr(sequence_df: pd.DataFrame) -> pd.DataFrame:
    """
    for a given dataframe of sequences, extract all hvr defined by primers in dict_of_primers above.
    :return: a dataframe with column being hvr and values being subsequences of hvr or None, an index column is kept as
    the one in the sequence_df
    """
    hvr_df = pd.DataFrame()
    for index in sequence_df.index:
        current_seq = sequence_df.sequence[index]
        positions_dict = get_all_hvr_positions_by_sequence(current_seq)
        current_hvr_dict = {"index": [index]}
        for key, value in positions_dict.items():
            if value is None:
                current_hvr_dict[key] = None
            else:
                current_hvr_dict[key] = current_seq[value[0][0]:value[0][1]]
        hvr_df = pd.concat([hvr_df, pd.DataFrame.from_dict(current_hvr_dict)])
    return hvr_df


# Functions
def get_all_hvr_positions_by_sequence(current_seq: str) -> Dict[str, Union[List[Tuple[int, int]], None]]:
    """
    For a given sequence and hvr to look for, return the dictionnary with hvr as keys and list (just a element if one)
    with starting and ending positions of the given hvr
    :param current_seq: seq in which to look for
    :return: dictionnary with hvr as keys with starting and ending positions of the given hvr in a list
    """
    dict_of_observed_hvr = {}
    for hvr in dict_of_primers.keys():
        dict_of_observed_hvr[hvr] = get_hvr_position(current_seq, hvr)
    return dict_of_observed_hvr


def get_hvr_position(sequence: str, hvr: str) -> Union[List[Tuple[int, int]], None]:
    """
    For a given sequence and hvr, look for the starting and ending points in the sequence, or return None
    :param sequence: current sequence
    :param hvr: current hvr (has to be in dict_of_primers.keys())
    :return: A list with one element (a tuple (start, end)) or None
    """
    start_primer = dict_of_primers[hvr]['forward']
    end_primer = dict_of_primers[hvr]['reverse']
    list_of_potential_start = get_list_of_related_primers(start_primer)
    list_of_potential_end = get_searchable_reverse_primer(get_list_of_related_primers(end_primer))
    starting_point = get_first_occurence(seq=sequence, list_of_primes=list_of_potential_start,
                                         forward_or_reverse='forward')
    ending_point = get_first_occurence(seq=sequence, list_of_primes=list_of_potential_end, forward_or_reverse='reverse')
    if starting_point is None or ending_point is None:
        return None
    else:
        return [(starting_point, ending_point)]


def get_first_occurence(seq: str, list_of_primes: List[str], forward_or_reverse: str) -> Union[int, None]:
    """
    For a given sequence and a list of potential primes to look for, return the relevant index of the sequence string
    where the HVR region starts or end (depending on forward_or_reverse parameter)
    :param seq: current sequence
    :param list_of_primes: list of all acceptable primers
    :param forward_or_reverse: eather 'forward' or 'reverse' (to know if we add the size of the primer to the found index)
    :return:
    """
    if len(list_of_primes) == 0:
        raise ValueError('No prime given - Check dict_of_primers ?')
    first_occu = -1
    for potential_prime in list_of_primes:
        first_occu = seq.find(potential_prime)
        if first_occu >= 0:
            break
    if first_occu == -1:
        return None
    else:
        if forward_or_reverse == 'forward':
            return first_occu + len(list_of_primes[0])
        else:
            return first_occu
