"""
Loading functions to load primers from fasta files
"""

from utils.utils import folder_paths
import pandas as pd
from typing import Dict


# Load primers fasta data already saved
def load_primers(forward_or_reverse: str = '') -> pd.DataFrame:
    """
    Load primers from data fasta file
    :param forward_or_reverse: string telling if forward or reverse primers are to be loaded
    :return:
    """
    seq_file = open(folder_paths['data'] + '{}_primers.fasta'.format(forward_or_reverse))
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


def get_dict_of_primers(start_df: pd.DataFrame = load_primers('forward'),
                        end_df: pd.DataFrame = load_primers('reverse')) -> \
        Dict[str, Dict[str, str]]:
    """
    Return a dict format with keys as hvr, and a dict with start and end keys for each hvr and the relevant sequence
    :param start_df: forward_primers df
    :param end_df: reverse_primers df
    :return: Dict format
    """
    primers_dict = {}
    for i in range(len(start_df.hvr)):
        hvr = start_df.hvr[i]
        primers_dict[hvr] = {'forward': start_df.loc[start_df.hvr == hvr].sequence[i],
                             'reverse': end_df.loc[end_df.hvr == hvr].sequence[i]}
    return primers_dict
