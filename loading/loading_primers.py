"""
Loading functions to load primers from fasta files
"""

from typing import Dict

import pandas as pd

from utils.utils import folder_paths


# Load primers fasta data already saved
def load_primers(forward_or_reverse: str = '', article: str = '') -> pd.DataFrame:
    """
    Load primers from data fasta file
    :param article: Where to look for primers (depending on the ressource used to look for it)
    :param forward_or_reverse: string telling if forward or reverse primers are to be loaded
    :return:
    """
    seq_file = open(folder_paths[article] + '{}_primers.fasta'.format(forward_or_reverse))
    line_to_read = True
    list_of_reads = []
    while line_to_read:
        hvr = seq_file.readline()[1:-1]
        seq = seq_file.readline()[:-2]
        if hvr == '':
            line_to_read = False
        else:
            list_of_reads.append([hvr, seq, forward_or_reverse])
    seq_file.close()
    return pd.DataFrame(list_of_reads, columns=['hvr', 'sequence', 'forward_or_reverse'])


def get_dict_of_primers(article: str = '') -> Dict[str, Dict[str, str]]:
    """
    Return a dict format with keys as hvr, and a dict with start and end keys for each hvr and the relevant sequence
    :param article: where to look for primers (depending on the ressource used to look for it)
    :return: Dict format
    """
    forward_df = load_primers(forward_or_reverse='forward', article=article)
    reverse_df = load_primers(forward_or_reverse='reverse', article=article)
    primers_dict = {}
    for i in range(len(forward_df.hvr)):
        hvr = forward_df.hvr[i]
        primers_dict[hvr] = {'forward': forward_df.loc[forward_df.hvr == hvr].sequence[i],
                             'reverse': reverse_df.loc[reverse_df.hvr == hvr].sequence[i]}
    return primers_dict
