# Packages
import os
from typing import Union, List

import pandas as pd
from IPython.display import display_html

# Constants
folder_paths = {'Sequence': 'D:\\0 - Boulot\\5 - X4\\16. Research Paper\\data\\gg_13_5_otus\\rep_set\\',
                'Taxonomy': 'D:\\0 - Boulot\\5 - X4\\16. Research Paper\\data\\gg_13_5_otus\\taxonomy\\',
                'data': 'D:\\0 - Boulot\\5 - X4\\16. Research Paper\\data\\',
                'model_data': 'D:\\0 - Boulot\\5 - X4\\16. Research Paper\\model_data\\',
                'Chaudhary': 'D:\\0 - Boulot\\5 - X4\\16. Research Paper\\data\\primers\\Chaudhary\\',
                'DairyDB': 'D:\\0 - Boulot\\5 - X4\\16. Research Paper\\data\\primers\\DairyDB\\',
                'stats_results': 'D:\\0 - Boulot\\5 - X4\\16. Research Paper\\taxonomic_classification\\results\\statistics\\',
                'model_results': 'D:\\0 - Boulot\\5 - X4\\16. Research Paper\\model_results\\',
                'model_final_results': 'D:\\0 - Boulot\\5 - X4\\16. Research Paper\\taxonomic_classification\\results\\models\\'}

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


def time_difference_good_format(t1: float, t2: float) -> str:
    """
    From two seconds time, compute the difference and give a relevant string of that time delta
    :param t1: first time
    :param t2: second time, higher than first
    :return: string with 'hours', 'minutes', 'secondes'
    """
    delta_t = int(t2 - t1)
    if delta_t < 60:
        if delta_t <= 1:
            return '{} second'.format(delta_t)
        else:
            return '{} seconds'.format(delta_t)
    elif delta_t < 3600:
        minutes = int(delta_t / 60)
        sec = delta_t % 60
        if minutes <= 1:
            if sec <= 1:
                return '{} minute and {} second'.format(minutes, sec)
            else:
                return '{} minute and {} seconds'.format(minutes, sec)
        else:
            if sec <= 1:
                return '{} minutes and {} second'.format(minutes, sec)
            else:
                return '{} minutes and {} seconds'.format(minutes, sec)
    elif delta_t < 3600 * 24:
        hours = int(delta_t / 3600)
        if hours <= 1:
            hours_s = ''
        else:
            hours_s = 's'
        minutes = int((delta_t % 3600) / 60)
        if minutes <= 1:
            minutes_s = ''
        else:
            minutes_s = 's'
        sec = delta_t % 60
        if sec <= 1:
            sec_s = ''
        else:
            sec_s = 's'
        return '{} hour{}, {} minute{} and {} second{}'.format(hours, hours_s, minutes, minutes_s, sec, sec_s)
    else:
        days = int(delta_t / (3600 * 24))
        if days <= 1:
            days_s = ''
        else:
            days_s = 's'
        hours = int((delta_t % (3600 * 24)) / 3600)
        if hours <= 1:
            hours_s = ''
        else:
            hours_s = 's'
        minutes = int((delta_t % 3600) / 60)
        if minutes <= 1:
            minutes_s = ''
        else:
            minutes_s = 's'
        return '{} day{}, {} hour{} and {} minute{}'.format(days, days_s, hours, hours_s, minutes, minutes_s)


def save_update(file_path, k, selected_primer, taxonomy_level, test_size, main_class_prop, accuracy):
    """

    :param file_path: where results are stored during the handling of a model for all situations
    :param selected_primer:
    :param taxonomy_level:
    :param test_size:
    :param main_class_prop:
    :param accuracy:
    :return: None, only update the file
    """
    if not os.path.exists(file_path):
        pd.DataFrame(columns=['HyperVariable Region', 'Taxonomy Rank to be classified', 'Test Size',
                              'Main Class prop', 'XGBoost - XGB({})'.format(k)]) \
            .to_csv(file_path, index=False)

    saving_df = pd.read_csv(file_path)
    saving_df = pd.concat([saving_df,
                           pd.DataFrame([[selected_primer, taxonomy_level, test_size, main_class_prop, accuracy]],
                                        columns=['HyperVariable Region', 'Taxonomy Rank to be classified', 'Test Size',
                                                 'Main Class prop', 'XGBoost - XGB({})'.format(k)])])
    saving_df.to_csv(file_path, index=False)


def restartkernel():
    display_html("<script>Jupyter.notebook.kernel.restart()</script>", raw=True)
