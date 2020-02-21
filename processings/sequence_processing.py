"""
Repository of all processing on sequences
Every fonction takes a seq (str) as input, and must returns a dictionary with keys as future columns
and values as preprocessed metrics
"""
# Packages
from typing import List

from utils.utils import get_list_of_related_primers_strict_ATCG


# Global Main

# Main by processing
def get_ATCG_proportion_in_seq(seq):
    """
    Return the proportion of each ATCG base in a given sequence -
    Takes into account the potential other letters
    :param seq:
    :return: (dict) keys are 'A' etc. and values are float summing to 1
    """
    l_seq = get_list_of_related_primers_strict_ATCG(seq.replace('N','').replace('X','').upper())
    counts = occurences_of_characters_in_list_of_sequence(l_seq)
    total_occurences = sum(counts.values())
    props = {}
    for key, value in counts.items():
        props[key] = value / total_occurences
    return props


# Support functions
def occurences_of_characters_in_sequence(seq: str) -> dict:
    """
    Counts characters in a sequence
    """
    res = {i: seq.count(i) for i in set(seq)}
    return res


def occurences_of_characters_in_list_of_sequence(l_seq: List[str]) -> dict:
    """
    Same but applied to list of sequences
    """
    global_seq = ''
    for seq in l_seq:
        global_seq += seq
    return occurences_of_characters_in_sequence(global_seq)
