# Packages
from typing import Union, List

# Constants
folder_paths = {'Sequence': 'D:\\0 - Boulot\\5 - X4\\16. Research Paper\\data\\gg_13_5_otus\\rep_set\\',
                'Taxonomy': 'D:\\0 - Boulot\\5 - X4\\16. Research Paper\\data\\gg_13_5_otus\\taxonomy\\',
                'data': 'D:\\0 - Boulot\\5 - X4\\16. Research Paper\\data\\',
                'model_data': 'D:\\0 - Boulot\\5 - X4\\16. Research Paper\\model_data\\',
                'Chaudhary': 'D:\\0 - Boulot\\5 - X4\\16. Research Paper\\data\\primers\\Chaudhary\\',
                'DairyDB': 'D:\\0 - Boulot\\5 - X4\\16. Research Paper\\data\\primers\\DairyDB\\',
                'stats_results': 'D:\\0 - Boulot\\5 - X4\\16. Research Paper\\taxonomic_classification\\results\\statistics\\',
                'model_results': 'D:\\0 - Boulot\\5 - X4\\16. Research Paper\\taxonomic_classification\\results\\models\\'}

target_alphabet = 'ATCG'
taxonomy_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
nomenclature_dict = {
    'A': ['A'],
    'T': ['T'],
    'C': ['C'],
    'G': ['G'],
    'U': ['T'],
    'R': ['A', 'G'],
    'Y': ['C', 'T'],
    'K': ['G', 'T'],
    'M': ['A', 'C'],
    'S': ['C', 'G'],
    'W': ['A', 'T'],
    'B': ['T', 'C', 'G'],
    'D': ['A', 'T', 'G'],
    'H': ['A', 'T', 'C'],
    'V': ['A', 'C', 'G'],
    'N': ['A', 'T', 'C', 'G'],
    'X': ['A', 'T', 'C', 'G'],
}

nomenclature_dict_large = {
    'A': ['A'],
    'T': ['T'],
    'C': ['C'],
    'G': ['G'],
    'U': ['U', 'T'],
    'R': ['R', 'A', 'G'],
    'Y': ['Y', 'C', 'T'],
    'K': ['K', 'G', 'T'],
    'M': ['M', 'A', 'C'],
    'S': ['S', 'C', 'G'],
    'W': ['W', 'A', 'T'],
    'B': ['B', 'T', 'C', 'G'],
    'D': ['D', 'A', 'T', 'G'],
    'H': ['H', 'A', 'T', 'C'],
    'V': ['V', 'A', 'C', 'G'],
    'N': ['N', 'A', 'T', 'C', 'G'],
    'X': ['X', 'A', 'T', 'C', 'G'],
}

complementary_dict = {
    'A': 'T',
    'T': 'A',
    'C': 'G',
    'G': 'C',
    'U': 'A',
    'R': 'Y',
    'Y': 'R',
    'K': 'M',
    'M': 'K',
    'S': 'W',
    'W': 'S',
    'B': 'V',
    'D': 'H',
    'H': 'D',
    'V': 'A',
    'N': 'N',
    'X': 'X'
}


# Getting the list of primers in ATCG format given an original primer in the usual nomenclature format
def get_list_of_related_primers_strict_ATCG(primer: str) -> List[str]:
    """
    Getting the list of primers in ATCG format given an original primer in the usual nomenclature format
    We still give the primer with the wrong character in the list of result, as they can also be used in real db
    :param primer: nomenclatured primer with potential ATCG + RYKMS...
    :return: List of str with all potential primers
    """
    list_of_primers = ['']
    # for all letters in the given primers
    for letter in primer:
        new_list_of_primers = []
        list_of_letter = nomenclature_dict[letter]
        # for all potential ATCG letter related to the given one
        for substitution_letter in list_of_letter:
            # for all the current primers reconstructed
            for current_primers in list_of_primers:
                # We create all the new potential ones
                new_list_of_primers.append(current_primers + substitution_letter)
        list_of_primers = new_list_of_primers
    return list_of_primers


# Getting the list of primers in all nomenclature format given an original primer in the usual nomenclature format
def get_list_of_related_primers(primer: str) -> List[str]:
    """
    Getting the list of primers in ATCG format given an original primer in the usual nomenclature format
    We still give the primer with the wrong character in the list of result, as they can also be used in real db
    :param primer: nomenclatured primer with potential ATCG + RYKMS...
    :return: List of str with all potential primers
    """
    list_of_primers = ['']
    # for all letters in the given primers
    for letter in primer:
        new_list_of_primers = []
        list_of_letter = nomenclature_dict_large[letter]
        # for all potential ATCG letter related to the given one
        for substitution_letter in list_of_letter:
            # for all the current primers reconstructed
            for current_primers in list_of_primers:
                # We create all the new potential ones
                new_list_of_primers.append(current_primers + substitution_letter)
        list_of_primers = new_list_of_primers
    return list_of_primers


# Getting complementary reverse in the good order
def get_searchable_reverse_primer(rev_prim: Union[str, List]) -> Union[str, List]:
    """

    :param rev_prim: eather a rev prime or a list of potential rev_prime
    :return: the correspondant complementary versions
    """
    if isinstance(rev_prim, str):
        re_ordered = rev_prim[::-1]
        complementary = ''
        for letter in re_ordered:
            complementary += complementary_dict[letter]
        return complementary
    elif isinstance(rev_prim, list):
        list_of_complementary = []
        for potential_reverse in rev_prim:
            list_of_complementary.append(get_searchable_reverse_primer(potential_reverse))
        return list_of_complementary
