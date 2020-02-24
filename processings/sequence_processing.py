"""
Repository of all processing on sequences
Every fonction takes a seq (str) as input, and must returns a dictionary with keys as future columns
and values as preprocessed metrics
"""
# Packages
from typing import List

from utils.utils import nomenclature_dict, target_alphabet


# Global Main

# Main by processing
def get_ATCG_proportion_in_seq(seq):
    """
    Return the proportion of each ATCG base in a given sequence -
    Takes into account the potential other letters
    :param seq:
    :return: (dict) keys are 'A' etc. and values are float summing to 1
    """
    # Clean sequence or there would be exponential results in memory and time
    seq = seq.upper()
    for letter in nomenclature_dict.keys():
        if letter not in target_alphabet:
            seq = seq.replace(letter, '')
    # Not looking for all related primers for complexity reasons
    # l_seq = get_list_of_related_primers_strict_ATCG(seq)
    # counts = occurences_of_characters_in_list_of_sequence(l_seq)
    counts = occurences_of_characters_in_sequence(seq)
    total_occurences = sum(counts.values())
    props = {}
    for key, value in counts.items():
        props[key] = value / total_occurences
    return props


def get_ATCG_k_mer_proportion_in_seq(seq, k):
    """
    Return the proportion of each ATCG base in a given sequence -
    Takes into account the potential other letters
    :param k: size of kmers
    :param seq:
    :return: (dict) keys are 'AB' etc. and values are float summing to 1
    """
    # Clean sequence or there would be exponential results in memory and time
    seq = seq.upper()
    for letter in nomenclature_dict.keys():
        if letter not in target_alphabet:
            seq = seq.replace(letter, '')
    # Not looking for all related primers for complexity reasons
    # l_seq = get_list_of_related_primers_strict_ATCG(seq)
    k_mer_to_consider = get_all_kmers(k)
    counts = occurences_of_k_mer_in_sequence(seq, k_mer_to_consider)
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


def occurences_of_k_mer_in_sequence(seq: str, k_mers: list) -> dict:
    """
    Counts k_mers in a sequence
    """
    res = {i: seq.count(i) for i in k_mers}
    return res


def get_all_kmers(k):
    """
    Get all interesting kmers for a given k
    """
    list_of_kmers = [a for a in target_alphabet]
    for i in range(1, k):
        new_list = []
        for old_kmer in list_of_kmers:
            for letter in target_alphabet:
                new_list.append(old_kmer + letter)
        list_of_kmers = new_list
    return list_of_kmers
